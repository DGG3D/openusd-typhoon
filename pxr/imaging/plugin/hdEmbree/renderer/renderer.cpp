//
// Copyright 2018 Pixar
//
// Licensed under the terms set forth in the LICENSE.txt file available at
// https://openusd.org/license.
//
// Modified by DGG3D 2026: throttle the per-pass display resolve (with a
// guaranteed final resolve) and mark single-sampled AOVs converged after
// the first pass. Also: adaptive stopping is checkpointed and retires a
// pixel only when its 3x3 neighbourhood agrees (_RetireAgreedAdaptivePixels).
// Also: per-pass timing for a remaining-time estimate of the current frame
// (_UpdateTimeEstimate, GetEstimatedSecondsRemaining); bookkeeping only.
//
// Frame orchestration and renderer configuration.

#include "renderer.h"
#include "rayUtil.h"
#include "renderBuffer.h"

#include <renderer/aov/adaptiveConvergence.h>
#include <renderer/materials/MaterialXCpp/materials/bsdf.h>
#include <renderer/materials/oiioTextureSystem.h>

#include "pxr/base/work/loops.h"
#include "pxr/base/work/threadLimits.h"
#include "pxr/base/work/workTBB/tbb_version.h"
#include "pxr/imaging/hd/perfLog.h"
#include "pxr/imaging/hd/renderBuffer.h"

#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>

// -------------------------------------------------------------------------
// Old TBB workaround - we plan to remove this once OpenUSD adopts
// oneTBB as a min spec. This applies the "Work" thread limit to the
// render thread if "Work" is using old TBB, but won't affect other "Work"
// implementations.  Note that it may affect Embree TBB usage as well.
//
// Note: The TBB version macro is located in different headers in legacy TBB.
// -------------------------------------------------------------------------
#if TBB_INTERFACE_VERSION_MAJOR < 12

#include <tbb/task_scheduler_init.h>

#include <optional>

namespace {

PXR_NAMESPACE_USING_DIRECTIVE

// Make the calling context respect PXR_WORK_THREAD_LIMIT, if run from a thread
// other than the main thread (ie, the renderThread)
class _ScopedThreadScheduler {
public:
    _ScopedThreadScheduler() {
        unsigned int limit = WorkGetConcurrencyLimitSetting();
        if (limit != 0) {
            _tbbTaskSchedInit.emplace(limit);
        }
    }

    std::optional<tbb::task_scheduler_init> _tbbTaskSchedInit;
};

}  // anonymous namespace

#endif  // TBB_INTERFACE_VERSION_MAJOR < 12


PXR_NAMESPACE_OPEN_SCOPE

namespace ty {
TF_DEFINE_PUBLIC_TOKENS(AovTokens, HDEMBREE_AOV_TOKENS);
}

ty::Renderer::Renderer()
    : _aovBindings()
    , _aovNames()
    , _aovBindingsVersion(0)
    , _width(0)
    , _height(0)
    , _viewMatrix(1.0f) // == identity
    , _projMatrix(1.0f) // == identity
    , _inverseViewMatrix(1.0f) // == identity
    , _inverseProjMatrix(1.0f) // == identity
    , _cameraExposureScale(1.0f)
    , _cameraDepthOfField()
    , _scene(nullptr)
    , _settings()
    , _lightingEnabled(true)
    , _wireframeColor(0.0f)
    , _wireframeLineWidth(1.0f)
    , _textureSystem(std::make_unique<ty::OiioTextureSystem>())
    , _renderColorSpace(ty::RenderColorSpace::LinearRec709)
    , _materialEvalServices{
        _textureSystem.get(),
        0.0f,
        0.0f,
        _renderColorSpace,
        ty::GetLuminanceCoefficients(_renderColorSpace)}
    , _completedSamples(0)
    , _sssCallCount(0)
    , _sssSuccessCount(0)
    , _sssWalkStepCount(0)
    , _sssIntersectionCount(0)
    , _ambientOcclusionRayCount(0)
{
}

ty::Renderer::~Renderer() = default;

void
ty::Renderer::SetScene(RTCScene scene)
{
    _scene = scene;
}

void
ty::Renderer::SetRenderSettings(
    ty::RenderSettings const& settings)
{
    // Normalize values required by renderer loops and work partitioning.
    ty::RenderSettings normalized = settings;
    normalized.maxBounces = std::max(0, normalized.maxBounces);
    normalized.lightSamplesPerHit =
        std::max(1, normalized.lightSamplesPerHit);
    normalized.tileSize = std::max(1, normalized.tileSize);

    // MaterialXCpp layer-throughput policy and OIIO's shared texture cache are
    // process-wide. Reapply them because another renderer or caller may have
    // changed them.
    // Keep the renderer-settings enum independent from the shading API enum.
    mxcpp::Bsdf::SetDielectricLayerThroughputMode(
        normalized.dielectricLayerThroughputMode ==
                ty::DielectricLayerThroughputMode::MaterialXGlsl
            ? mxcpp::Bsdf::DielectricLayerThroughputMode::MaterialXGlsl
            : mxcpp::Bsdf::DielectricLayerThroughputMode::Bsdl);
    if (ty::OiioTextureSystem* oiio =
            dynamic_cast<ty::OiioTextureSystem*>(
                _textureSystem.get())) {
        oiio->SetCacheSizeMB(normalized.textureCacheSizeMB);
    }

    _settings = normalized;
}

void
ty::Renderer::SetLightingEnabled(bool lightingEnabled)
{
    _lightingEnabled = lightingEnabled;
}

void
ty::Renderer::SetWireframeStyle(
    GfVec4f const& color,
    float lineWidth)
{
    _wireframeColor = color;
    _wireframeLineWidth = std::max(1.0f, lineWidth);
}

void
ty::Renderer::SetDataWindow(const GfRect2i& dataWindow)
{
    _dataWindow = dataWindow;
}

void
ty::Renderer::SetCamera(const GfMatrix4d& viewMatrix,
                            const GfMatrix4d& projMatrix)
{
    _viewMatrix = viewMatrix;
    _projMatrix = projMatrix;
    _inverseViewMatrix = viewMatrix.GetInverse();
    _inverseProjMatrix = projMatrix.GetInverse();
}

void
ty::Renderer::SetCameraExposureScale(float cameraExposureScale)
{
    _cameraExposureScale = cameraExposureScale;
}

void
ty::Renderer::SetCameraDepthOfField(
    ty::CameraDepthOfField const& cameraDepthOfField)
{
    _cameraDepthOfField = cameraDepthOfField;
}

void
ty::Renderer::SetSceneFrameAndTime(float frame, float time)
{
    _materialEvalServices.frame = frame;
    _materialEvalServices.time = time;
}

void
ty::Renderer::SetRenderColorSpace(ty::RenderColorSpace colorSpace)
{
    _renderColorSpace = colorSpace;
    _materialEvalServices.renderColorSpace = colorSpace;
    _materialEvalServices.luminanceCoefficients =
        ty::GetLuminanceCoefficients(colorSpace);
    if (ty::OiioTextureSystem* oiio =
            dynamic_cast<ty::OiioTextureSystem*>(_textureSystem.get())) {
        oiio->SetRenderColorSpace(colorSpace);
    }
}

int
ty::Renderer::GetCompletedSamples() const
{
    return _completedSamples.load();
}

void
ty::Renderer::MarkFramePending()
{
    _frameStatus.store(_FrameStatus::Pending, std::memory_order_release);
}

bool
ty::Renderer::DidLastFrameProduceValidPixels() const
{
    return _frameStatus.load(std::memory_order_acquire) == _FrameStatus::Valid;
}

float
ty::Renderer::GetRenderElapsedSeconds() const
{
    std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now();
    return std::chrono::duration<float>(now - _renderStartTime).count();
}

uint64_t
ty::Renderer::GetSssCallCount() const
{
    return _sssCallCount.load();
}

uint64_t
ty::Renderer::GetSssSuccessCount() const
{
    return _sssSuccessCount.load();
}

uint64_t
ty::Renderer::GetSssWalkStepCount() const
{
    return _sssWalkStepCount.load();
}

uint64_t
ty::Renderer::GetSssIntersectionCount() const
{
    return _sssIntersectionCount.load();
}

uint64_t
ty::Renderer::GetAmbientOcclusionRayCount() const
{
    return _ambientOcclusionRayCount.load();
}

uint64_t
ty::Renderer::GetConvergedPixelCount() const
{
    return _convergedPixelCount.load(std::memory_order_relaxed);
}

uint64_t
ty::Renderer::GetTotalPixels() const
{
    return _totalPixelsInDataWindow.load(std::memory_order_relaxed);
}

uint64_t
ty::Renderer::GetActivePixelCount() const
{
    const uint64_t total = GetTotalPixels();
    const uint64_t converged = GetConvergedPixelCount();
    return (converged < total) ? total - converged : 0;
}

double
ty::Renderer::GetEstimatedSecondsRemaining() const
{
    // Same completion short-circuit as GetPercentDone().
    if (_frameComplete.load(std::memory_order_relaxed)) {
        return 0.0;
    }
    return _estimatedSecondsRemaining.load(std::memory_order_relaxed);
}

void
ty::Renderer::_ResetTimeEstimate()
{
    std::lock_guard<std::mutex> lock(_timeEstimateMutex);
    _adaptiveTimeEstimator.Reset();
    _estimatedSecondsRemaining.store(-1.0, std::memory_order_relaxed);
}

void
ty::Renderer::_UpdateTimeEstimate(
    int completedPasses, uint64_t activeBeforePass, double passSeconds,
    bool timed)
{
    const uint64_t activeAfterPass = GetActivePixelCount();
    const int minSamples = _settings.minSamplesBeforeAdaptive;
    std::lock_guard<std::mutex> lock(_timeEstimateMutex);
    if (timed) {
        _adaptiveTimeEstimator.RecordPass(activeBeforePass, passSeconds);
    }
    // Every active pixel now holds completedPasses samples, so this pass's
    // sweep could only retire pixels if that count is a checkpoint.
    if (completedPasses > 0 &&
        ty::IsAdaptiveCheckpoint(
            static_cast<uint32_t>(completedPasses), minSamples)) {
        _adaptiveTimeEstimator.RecordCheckpointSurvival(
            activeBeforePass, activeAfterPass);
    }
    _estimatedSecondsRemaining.store(
        _adaptiveTimeEstimator.EstimateRemainingSeconds(
            completedPasses, activeAfterPass,
            _settings.samplesToConvergence, minSamples),
        std::memory_order_relaxed);
}

double
ty::Renderer::ComputePercentDone(
    uint64_t convergedPixels,
    uint64_t totalPixels,
    int completedSamples,
    int samplesToConvergence)
{
    // No pixels means the frame has not been set up yet (e.g. polled before
    // the first _PreRenderSetup()), not that it is done: report 0. A finished
    // frame is reported as 100 by GetPercentDone() via _frameComplete.
    if (totalPixels == 0) {
        return 0.0;
    }
    if (samplesToConvergence <= 0) {
        return 100.0;
    }
    const double c =
        static_cast<double>(std::min(convergedPixels, totalPixels)) /
        static_cast<double>(totalPixels);
    const double s = static_cast<double>(std::max(0, completedSamples));
    const double n = static_cast<double>(samplesToConvergence);
    const double sFraction = std::min(s / n, 1.0);
    const double percent = 100.0 * (c + (1.0 - c) * sFraction);
    return std::min(100.0, std::max(0.0, percent));
}

double
ty::Renderer::GetPercentDone() const
{
    // Some exit paths finish the frame without every pixel satisfying
    // adaptive convergence (for example, no multi-sampled AOVs left after
    // the first pass); report 100 whenever the renderer considers the frame
    // complete rather than relying on the formula alone.
    if (_frameComplete.load(std::memory_order_relaxed)) {
        return 100.0;
    }
    return ComputePercentDone(
        GetConvergedPixelCount(),
        GetTotalPixels(),
        _completedSamples.load(std::memory_order_relaxed),
        _settings.samplesToConvergence);
}

bool
ty::Renderer::_PreRenderSetup()
{
    HD_TRACE_FUNCTION();

    // Reset state derived only from this setup invocation so failure cannot
    // expose the previous frame's dimensions or AOV dispatch. Adaptive state
    // persists until ResetAccumulation() or a setup failure owns its reset.
    _width = 0;
    _height = 0;
    _needRadiance = false;
    _needAmbientOcclusion = false;
    _colorClearValue = GfVec4f(0.0f);
    _aovOutputs.clear();
    _completedSamples.store(0);
    // Frame-completion state is derived fresh by this Render() call; a
    // previous frame's completion must not leak into GetPercentDone() while
    // this one is still in progress.
    _frameComplete.store(false, std::memory_order_relaxed);
    // Modified by DGG3D 2026: the remaining-time estimate is per Render()
    // call, like _completedSamples.
    _ResetTimeEstimate();
    _sssCallCount.store(0);
    _sssSuccessCount.store(0);
    _sssWalkStepCount.store(0);
    _sssIntersectionCount.store(0);
    _ambientOcclusionRayCount.store(0);

    // Validate every observable failure before committing the scene or
    // mapping a buffer, so setup failure needs no partial cleanup.
    bool setupValid = true;
    if (_scene == nullptr) {
        TF_WARN("Cannot render without an Embree scene");
        setupValid = false;
    }
    if (!_ValidateAovBindings()) {
        setupValid = false;
    }
    if (!setupValid) {
        // A terminal setup failure must not retain adaptive state from the
        // previous valid frame. Keep this off the successful restart path,
        // where ResetAccumulation() has already cleared the same arrays.
        _pixelMean.clear();
        _pixelM2.clear();
        _pixelSampleCount.clear();
        _pixelConverged.clear();
        _pixelAdaptivePass.clear();
        _convergedPixelCount.store(0, std::memory_order_relaxed);
        _totalPixelsInDataWindow.store(0, std::memory_order_relaxed);

        // Mark usable buffers converged so Hydra parks instead of retrying a
        // terminal setup failure.
        for (size_t i = 0; i < _aovBindings.size(); ++i) {
            ty::RenderBufferInterface *renderBuffer =
                dynamic_cast<ty::RenderBufferInterface*>(
                    _aovBindings[i].renderBuffer);
            if (renderBuffer != nullptr) {
                renderBuffer->SetConverged(true);
            }
        }
        return false;
    }

    {
        HD_TRACE_SCOPE("ty::Renderer::CommitScene");
        // Commit pending changes only after setup is known to be valid.
        rtcCommitScene(_scene);
    }

    // Build all per-frame state before mapping validated buffers.
    if (_width > 0 && _height > 0) {
        const size_t numPixels = _width * _height;
        _pixelMean.resize(numPixels, GfVec3f(0.0f));
        _pixelM2.resize(numPixels, GfVec3f(0.0f));
        _pixelSampleCount.resize(numPixels, 0);
        _pixelConverged.resize(numPixels, uint8_t(0));
        _pixelAdaptivePass.resize(numPixels, uint8_t(0));

        // Only pixels inside the data window are ever sampled or marked
        // converged; it may be smaller than the full buffer resolution.
        const uint64_t totalPixels =
            static_cast<uint64_t>(_dataWindow.GetWidth()) *
            static_cast<uint64_t>(_dataWindow.GetHeight());
        _totalPixelsInDataWindow.store(totalPixels, std::memory_order_relaxed);
    }

    _ClassifyAovOutputs();

    // A validated interface Map is non-failing aside from allocation failure,
    // which is outside this non-exception setup contract.
    for (size_t i = 0; i < _aovBindings.size(); ++i) {
        ty::RenderBufferInterface *renderBuffer =
            dynamic_cast<ty::RenderBufferInterface*>(
                _aovBindings[i].renderBuffer);
        renderBuffer->Map();
    }
    return true;
}

void
ty::Renderer::Render(HdRenderThread *renderThread)
{
    HD_TRACE_FUNCTION();

#if TBB_INTERFACE_VERSION_MAJOR < 12
    _ScopedThreadScheduler scheduler;
#endif

    _renderStartTime = std::chrono::steady_clock::now();
    if (!_PreRenderSetup()) {
        _frameStatus.store(_FrameStatus::Failed, std::memory_order_release);
        return;
    }
    _frameStatus.store(_FrameStatus::Valid, std::memory_order_release);

    // Compute the OpenQMC frame seed once per Render() call. An explicit
    // render setting or environment seed overrides the scene frame.
    const uint32_t baseSeed =
        ty::ResolveFrameSeed(
            _settings.randomNumberSeed, _materialEvalServices.frame);

    const unsigned int tileSize =
        static_cast<unsigned int>(_settings.tileSize);
    const unsigned int numTilesX =
        (_dataWindow.GetWidth() + tileSize - 1) / tileSize;
    const unsigned int numTilesY =
        (_dataWindow.GetHeight() + tileSize - 1) / tileSize;

    // ---- Coarse preview passes ----
    // Render a sparse subset of pixels and block-fill the display buffer
    // so the user sees a mosaic preview almost immediately, then refine.
    {
        static const unsigned int kPreviewStrides[] = {8, 4, 2};
        for (unsigned int stride : kPreviewStrides) {
            if (renderThread->IsStopRequested()) {
                break;
            }

            // Only run coarse passes that are coarser than a single pixel.
            if (stride >= static_cast<unsigned int>(_dataWindow.GetWidth()) &&
                stride >= static_cast<unsigned int>(_dataWindow.GetHeight())) {
                continue;
            }

            {
                HD_TRACE_SCOPE("ty::Renderer::TracePreviewPass");
                WorkParallelForN(numTilesX * numTilesY,
                    std::bind(&ty::Renderer::_RenderTiles, this,
                        renderThread, /*sampleNum=*/0, baseSeed, stride,
                        std::placeholders::_1, std::placeholders::_2));
            }

            if (renderThread->IsStopRequested()) {
                break;
            }

            // Resolve sparse samples into the display buffer and
            // replicate each sampled pixel across its block.
            {
                HD_TRACE_SCOPE("ty::Renderer::ResolvePreviewPass");
                std::unique_lock<std::mutex> lock =
                    renderThread->LockFramebuffer();
                for (size_t i = 0; i < _aovBindings.size(); ++i) {
                    ty::RenderBufferInterface *renderBuffer =
                        dynamic_cast<ty::RenderBufferInterface*>(
                            _aovBindings[i].renderBuffer);
                    renderBuffer->Resolve();
                    renderBuffer->BlockFill(stride);
                }
            }
        }

        // Clear the sample accumulation so the full-resolution passes
        // start from a clean slate, while the display buffer retains
        // the coarse preview for visual continuity.
        if (!renderThread->IsStopRequested()) {
            for (size_t i = 0; i < _aovBindings.size(); ++i) {
                ty::RenderBufferInterface *renderBuffer =
                    dynamic_cast<ty::RenderBufferInterface*>(
                        _aovBindings[i].renderBuffer);
                renderBuffer->ClearSamples();
            }
            // Reset adaptive sampling state that was partially filled
            // by the coarse passes.
            std::fill(_pixelMean.begin(), _pixelMean.end(), GfVec3f(0.0f));
            std::fill(_pixelM2.begin(), _pixelM2.end(), GfVec3f(0.0f));
            std::fill(
                _pixelSampleCount.begin(), _pixelSampleCount.end(), 0);
            std::fill(
                _pixelConverged.begin(), _pixelConverged.end(), uint8_t(0));
            std::fill(
                _pixelAdaptivePass.begin(), _pixelAdaptivePass.end(),
                uint8_t(0));
            _convergedPixelCount.store(0, std::memory_order_relaxed);
            _ResetTimeEstimate();
        }
    }

    // ---- Full-resolution multi-sample rendering ----
    // Each pass adds one sample per pixel.  After every pass we resolve
    // the accumulation buffer so the display shows progressively
    // improving quality.
    //
    // Modified by DGG3D 2026: Resolve() (see HdEmbreeRenderBuffer::Resolve)
    // is now internally parallelized, but it still touches every pixel of
    // every AOV under the framebuffer lock, which idles worker threads
    // between passes. Throttle how often the in-loop resolve runs to about
    // once per kResolveThrottleInterval, except for the first
    // kAlwaysResolvePasses passes (so the viewport still updates promptly
    // right after a render starts) — and always resolve once more,
    // unconditionally, after the loop below exits, so the displayed image
    // always reflects the last completed pass. Resolve() only recomputes
    // averages from the accumulated sample buffers, so skipping
    // intermediate resolves cannot change the final resolved pixel values.
    static const int kAlwaysResolvePasses = 4;
    static const std::chrono::milliseconds kResolveThrottleInterval(250);
    auto lastResolveTime = std::chrono::steady_clock::now();

    bool renderFinished = false;
    for (int i = 0; i < _settings.samplesToConvergence; ++i) {
        // Pause point.
        while (renderThread->IsPauseRequested()) {
            if (renderThread->IsStopRequested()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // Cancellation point.
        if (renderThread->IsStopRequested()) {
            break;
        }

        // Modified by DGG3D 2026: time the pass (after the pause point, so
        // pausing is not counted) for the remaining-time estimate.
        const auto passStartTime = std::chrono::steady_clock::now();
        const uint64_t activeBeforePass = GetActivePixelCount();

        {
            HD_TRACE_SCOPE("ty::Renderer::TraceSamplePass");
            WorkParallelForN(numTilesX * numTilesY,
                std::bind(&ty::Renderer::_RenderTiles, this,
                    renderThread, i, baseSeed, /*stride=*/1u,
                    std::placeholders::_1, std::placeholders::_2));
        }

        // Modified by DGG3D 2026: adaptive retirement is a separate phase.
        // The pass above only recorded, per pixel, whether it passed the
        // convergence test at a checkpoint count; now that every tile has
        // joined, retire the pixels whose whole in-window 3x3 neighbourhood
        // passed, before the next pass decides which pixels to skip and
        // before the all-converged check below. Skipped on cancellation,
        // where some tiles may not have received this pass's sample.
        if (!_pixelConverged.empty() && !renderThread->IsStopRequested()) {
            HD_TRACE_SCOPE("ty::Renderer::RetireAgreedAdaptivePixels");
            _RetireAgreedAdaptivePixels();
        }

        // Resolve intermediate results so the viewport shows progressive
        // refinement instead of staying blank until convergence. Throttled;
        // see the comment above this loop.
        {
            const auto now = std::chrono::steady_clock::now();
            const bool shouldResolveThisPass =
                (i < kAlwaysResolvePasses) ||
                (now - lastResolveTime >= kResolveThrottleInterval);
            if (shouldResolveThisPass) {
                HD_TRACE_SCOPE("ty::Renderer::ResolveSamplePass");
                std::unique_lock<std::mutex> lock =
                    renderThread->LockFramebuffer();
                for (size_t i = 0; i < _aovBindings.size(); ++i) {
                    ty::RenderBufferInterface *renderBuffer =
                        dynamic_cast<ty::RenderBufferInterface*>(
                            _aovBindings[i].renderBuffer);
                    renderBuffer->Resolve();
                }
                lastResolveTime = now;
            }
        }

        // After the first pass, mark the single-sampled attachments (depth,
        // primId, normal, ...) as converged. This is checked by the
        // IsConverged() gate in _EvaluatePixelSample, so they stop being
        // written to on later passes and simply keep this first pass's
        // value instead of being repeatedly overwritten by every later
        // pass (each of which may sample a slightly different sub-pixel
        // jitter position). Actually unmapping them is left to the
        // FinalizeAovs step below, alongside the multisampled attachments,
        // to keep exactly one Map()/Unmap() pair per buffer per frame.
        // If there are no multisampled attachments, we are done.
        if (i == 0) {
            bool moreWork = false;
            for (size_t i = 0; i < _aovBindings.size(); ++i) {
                ty::RenderBufferInterface *renderBuffer =
                    dynamic_cast<ty::RenderBufferInterface*>(
                        _aovBindings[i].renderBuffer);
                if (renderBuffer->IsMultiSampled()) {
                    moreWork = true;
                } else {
                    renderBuffer->SetConverged(true);
                }
            }
            if (!moreWork) {
                _completedSamples.store(i + 1);
                renderFinished = true;
                break;
            }
        }

        // Track the number of completed samples for external consumption.
        _completedSamples.store(i + 1);

        // Stop once every pixel satisfies the adaptive convergence test.
        if (!_pixelConverged.empty()) {
            HD_TRACE_SCOPE("ty::Renderer::CheckConvergence");
            bool allConverged = true;
            for (size_t indexPixel = 0;
                 indexPixel < _pixelConverged.size();
                 ++indexPixel) {
                if (!_pixelConverged[indexPixel]) {
                    allConverged = false;
                    break;
                }
            }
            if (allConverged) {
                renderFinished = true;
                break;
            }
        }

        // Modified by DGG3D 2026: feed the remaining-time estimate. A pass
        // cut short by a stop request is not representative and is skipped.
        // The first pass is not used for the pass-time fit: it carries
        // one-off warm-up costs (texture and cache population, the
        // single-sampled AOVs) that later passes do not repeat.
        if (!renderThread->IsStopRequested()) {
            const double passSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - passStartTime).count();
            _UpdateTimeEstimate(
                i + 1, activeBeforePass, passSeconds, /*timed=*/i > 0);
        }

        // Cancellation point.
        if (renderThread->IsStopRequested()) {
            break;
        }

        // If this is the last iteration, rendering completed naturally.
        if (i == _settings.samplesToConvergence - 1) {
            renderFinished = true;
        }
    }

    // GetPercentDone() reports 100 once the frame is complete, regardless
    // of which exit path above reached that state (for example, no
    // multi-sampled AOVs left after the first pass, not just every pixel
    // satisfying adaptive convergence).
    _frameComplete.store(renderFinished, std::memory_order_relaxed);

    // Unconditional final resolve: the in-loop resolve above is throttled,
    // so the last completed pass may not have been resolved into the
    // display buffer yet. Always resolve once more here, however the loop
    // exited (converged, stopped, or sample cap), so the final image is
    // always up to date with the last pass that was actually rendered.
    {
        HD_TRACE_SCOPE("ty::Renderer::FinalResolve");
        std::unique_lock<std::mutex> lock = renderThread->LockFramebuffer();
        for (size_t i = 0; i < _aovBindings.size(); ++i) {
            ty::RenderBufferInterface *renderBuffer =
                dynamic_cast<ty::RenderBufferInterface*>(
                    _aovBindings[i].renderBuffer);
            renderBuffer->Resolve();
        }
    }

    // Mark the multisampled attachments as converged and unmap all buffers.
    {
        HD_TRACE_SCOPE("ty::Renderer::FinalizeAovs");
        for (size_t i = 0; i < _aovBindings.size(); ++i) {
            ty::RenderBufferInterface *renderBuffer =
                dynamic_cast<ty::RenderBufferInterface*>(
                    _aovBindings[i].renderBuffer);
            renderBuffer->Unmap();
            renderBuffer->SetConverged(true);
        }
    }

    // Print render statistics only when rendering completed (not interrupted).
    if (renderFinished) {
        const float elapsedSec = GetRenderElapsedSeconds();
        const int completedSamples = _completedSamples.load();
        const int imageWidth = _dataWindow.GetWidth();
        const int imageHeight = _dataWindow.GetHeight();
        const long long totalSamples =
            static_cast<long long>(imageWidth) *
            imageHeight * completedSamples;

        std::printf("\n");
        std::printf("===== hdEmbree Render Statistics =====\n");
        std::printf(
            "  Resolution       : %d x %d\n", imageWidth, imageHeight);
        std::printf("  Samples/pixel    : %d / %d\n",
                    completedSamples, _settings.samplesToConvergence);
        std::printf("  Total samples    : %lld\n", totalSamples);
        std::printf("  Render time      : %.3f s\n", elapsedSec);
        if (elapsedSec > 0.0f) {
            std::printf("  Samples/sec      : %.0f\n",
                        totalSamples / static_cast<double>(elapsedSec));
            std::printf("  Pixels/sec       : %.0f\n",
                (static_cast<double>(imageWidth) *
                 imageHeight * completedSamples)
                            / elapsedSec);
        }
        std::printf("  Max bounces      : %d\n", _settings.maxBounces);
        std::printf("  Light samples    : %d\n", _settings.lightSamplesPerHit);
        const uint64_t sssCalls = _sssCallCount.load();
        if (sssCalls > 0) {
            const uint64_t sssSuccesses = _sssSuccessCount.load();
            const uint64_t sssWalkSteps = _sssWalkStepCount.load();
            const uint64_t sssIntersections = _sssIntersectionCount.load();
            const double successRate =
                100.0 * static_cast<double>(sssSuccesses)
                / static_cast<double>(sssCalls);
            const double avgStepsPerCall =
                static_cast<double>(sssWalkSteps)
                / static_cast<double>(sssCalls);
            const double avgStepsPerSuccess =
                (sssSuccesses > 0)
                    ? static_cast<double>(sssWalkSteps)
                        / static_cast<double>(sssSuccesses)
                    : 0.0;
            const double avgIntersectionsPerCall =
                static_cast<double>(sssIntersections)
                / static_cast<double>(sssCalls);

            std::printf(
                "  SSS walks        : %llu calls, %llu success (%.1f%%)\n",
                static_cast<unsigned long long>(sssCalls),
                static_cast<unsigned long long>(sssSuccesses),
                successRate);
            std::printf(
                "  SSS avg steps    : %.2f / call, %.2f / success\n",
                avgStepsPerCall,
                avgStepsPerSuccess);
            std::printf(
                "  SSS intersects   : %llu total, %.2f / call\n",
                static_cast<unsigned long long>(sssIntersections),
                avgIntersectionsPerCall);
        }

        if (!_pixelConverged.empty()) {
            size_t convergedCount = 0;
            double avgSamples = 0.0;
            for (size_t indexPixel = 0;
                 indexPixel < _pixelConverged.size();
                 ++indexPixel) {
                if (_pixelConverged[indexPixel]) {
                    ++convergedCount;
                }
                avgSamples += _pixelSampleCount[indexPixel];
            }
            avgSamples /= _pixelConverged.size();
            const double convergedPct =
                100.0 * convergedCount / _pixelConverged.size();
            std::printf("  Adaptive sampling: on (threshold=%.4f)\n",
                        _settings.adaptiveThreshold);
            std::printf("  Converged pixels : %zu / %zu (%.1f%%)\n",
                        convergedCount, _pixelConverged.size(), convergedPct);
            std::printf("  Avg samples/pixel: %.1f\n", avgSamples);
        }

        std::printf("======================================\n");
        std::fflush(stdout);
    }
}

// Modified by DGG3D 2026: new. See the declaration in renderer.h and
// renderer/aov/adaptiveConvergence.h.
void
ty::Renderer::_RetireAgreedAdaptivePixels()
{
    if (_pixelConverged.empty() ||
        _pixelAdaptivePass.size() != _pixelConverged.size() ||
        _pixelSampleCount.size() != _pixelConverged.size() ||
        _width == 0 || _height == 0) {
        return;
    }

    // Same buffer-space data window as _RenderTiles: the data window is
    // y-down but the image rows are bottom-to-top, so flip it. Clamped to
    // the buffer so neighbour lookups can never leave it.
    unsigned int minY = _dataWindow.GetMinY();
    unsigned int maxY = _dataWindow.GetMaxY() + 1;
    std::swap(minY, maxY);
    minY = _height - minY;
    maxY = _height - maxY;

    ty::AdaptiveWindow window;
    window.minX = std::min(
        static_cast<unsigned int>(_dataWindow.GetMinX()), _width);
    window.maxX = std::min(
        static_cast<unsigned int>(_dataWindow.GetMaxX() + 1), _width);
    window.minY = std::min(minY, _height);
    window.maxY = std::min(maxY, _height);
    if (window.minX >= window.maxX || window.minY >= window.maxY) {
        return;
    }

    const int minSamples = _settings.minSamplesBeforeAdaptive;
    // Each worker reads only pass flags / sample counts (not written during
    // this sweep) and writes only _pixelConverged entries of its own rows,
    // so the retirement decision is independent of sweep order.
    WorkParallelForN(window.maxY - window.minY,
        [this, &window, minSamples](size_t rowBegin, size_t rowEnd) {
            const uint64_t retired = ty::RetireAgreedAdaptivePixels(
                _pixelAdaptivePass.data(),
                _pixelSampleCount.data(),
                _pixelConverged.data(),
                _width,
                window,
                minSamples,
                window.minY + static_cast<unsigned int>(rowBegin),
                window.minY + static_cast<unsigned int>(rowEnd));
            if (retired != 0) {
                _convergedPixelCount.fetch_add(
                    retired, std::memory_order_relaxed);
            }
        });
}

void
ty::Renderer::_EvaluatePixelSample(
    unsigned int x, unsigned int y,
    GfVec3f const& posRayOrgWld, GfVec3f const& dirRayWld,
    ty::Sampler const& sampler,
    ty::RayDifferential const& diffRay)
{
    _PixelSampleResult result;
    if (_needRadiance) {
        result = _lightingEnabled
            ? _IntegratePath(
                  posRayOrgWld, dirRayWld, diffRay, sampler.RootDomain())
            : _IntegrateUnlit(
                  posRayOrgWld, dirRayWld);
        _ApplyWireframe(result.primaryHit, diffRay, &result.color);
    } else {
        // Geometric and ambient-occlusion AOV-only renders need the primary
        // hit but no radiance.
        result.primaryHit.ray.flags = 0;
        ty::PopulateRayHit(
            &result.primaryHit, posRayOrgWld, dirRayWld, 0.0f,
            std::numeric_limits<float>::max(),
            ty::RayMask::Camera);
        rtcIntersect1(_scene, &result.primaryHit);
    }

    if (_needAmbientOcclusion) {
        result.ambientVisibility = _ComputeAmbientOcclusion(
            result.primaryHit,
            sampler.RootDomain().Fork(
                ty::SampleDomainKey::AmbientOcclusion));
    }

    if (!_pixelConverged.empty()) {
        const GfVec3f convergenceSample = _needRadiance
            ? GfVec3f(result.color[0], result.color[1], result.color[2])
            : GfVec3f(result.ambientVisibility);
        _UpdateVariance(x, y, convergenceSample);
    }

    for (_AovOutput const& aov : _aovOutputs) {
        if (!aov.buffer->IsConverged()) {
            _WriteAov(aov, result, x, y);
        }
    }
}

void
ty::Renderer::_RenderTiles(HdRenderThread *renderThread, int sampleNum,
                               uint32_t baseSeed, unsigned int stride,
                               size_t tileStart, size_t tileEnd)
{
    const unsigned int minX = _dataWindow.GetMinX();
    unsigned int minY = _dataWindow.GetMinY();
    const unsigned int maxX = _dataWindow.GetMaxX() + 1;
    unsigned int maxY = _dataWindow.GetMaxY() + 1;

    // If a client does not use AOVs and we have no render buffers,
    // _height is 0 and we shouldn't use it to flip the data window.
    if (_height > 0) {
        // The data window is y-Down but the image line order
        // is from bottom to top, so we need to flip it.
        std::swap(minY, maxY);
        minY = _height - minY;
        maxY = _height - maxY;
    }

    const unsigned int tileSize =
        static_cast<unsigned int>(_settings.tileSize);
    const unsigned int numTilesX =
        (_dataWindow.GetWidth() + tileSize - 1) / tileSize;

    // _RenderTiles gets a range of tiles; iterate through them.
    for (unsigned int tile = tileStart; tile < tileEnd; ++tile) {
        // Cancellation point.
        if (renderThread && renderThread->IsStopRequested()) {
            break;
        }

        // Compute the pixel location of tile boundaries.
        const unsigned int tileY = tile / numTilesX;
        const unsigned int tileX = tile - tileY * numTilesX;
        const unsigned int x0 = tileX * tileSize + minX;
        const unsigned int y0 = tileY * tileSize + minY;
        // Clamp to data window, in case tileSize doesn't
        // neatly divide its with and height.
        const unsigned int x1 = std::min(x0 + tileSize, maxX);
        const unsigned int y1 = std::min(y0 + tileSize, maxY);

        // Loop over pixels casting rays.
        for (unsigned int y = y0; y < y1; ++y) {
            // For coarse preview passes, only render sparse pixels
            // whose data-window-relative coordinates are multiples
            // of the stride.
            if (stride > 1 && ((y - minY) % stride != 0)) {
                continue;
            }
            for (unsigned int x = x0; x < x1; ++x) {
                if (stride > 1 && ((x - minX) % stride != 0)) {
                    continue;
                }

                // Skip pixels that already satisfy adaptive convergence.
                const size_t idx = y * _width + x;
                if (!_pixelConverged.empty() && _pixelConverged[idx]) {
                    continue;
                }

                // Create a per-pixel OpenQMC sampler.
                ty::Sampler sampler(
                    baseSeed,
                    x,
                    y,
                    sampleNum);

                GfVec3f posRayOrgWld;
                GfVec3f dirRayWld;
                ty::RayDifferential diffRay;
                _SampleCameraRay(
                    x, y, minX, minY, sampler,
                    posRayOrgWld, dirRayWld, diffRay);

                // Evaluate and write this pixel sample.
                _EvaluatePixelSample(
                    x, y,
                    posRayOrgWld, dirRayWld,
                    sampler, diffRay);
            }
        }
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
