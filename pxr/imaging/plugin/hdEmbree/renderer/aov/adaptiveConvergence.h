//
// Copyright 2026 Pixar
//
// Licensed under the terms set forth in the LICENSE.txt file available at
// https://openusd.org/license.
//
// Modified by DGG3D 2026: new file; factors the adaptive-sampling stopping
// decision out of aovOutput.cpp and adds checkpointed stopping plus 3x3
// neighbourhood agreement, so the renderer and its unit test share one
// implementation.
//
// Adaptive-sampling convergence decision.
//
// A pixel's per-sample estimator is never altered here: samples are
// accumulated with Welford's update exactly as before, and the per-channel
// test (varOfMean <= absoluteFloor + threshold * mean^2) is unchanged. What
// changed is *when* a pixel may be retired:
//
//  1. Checkpointed stopping. The test is only evaluated at sample counts
//     minSamples * 2^k (k >= 0). Testing after every sample is "optional
//     stopping": in dim regions dominated by rare bright samples, a pixel is
//     retired right after a run without spikes, i.e. exactly when its running
//     mean is too low, leaving persistent dark pixels.
//  2. Neighbourhood agreement. A pixel is only retired when it and all of its
//     in-data-window 8-neighbours passed the test at their latest checkpoint.
//     The per-pixel pass flag is written by the sampling workers; retirement
//     happens in a separate sweep between sample passes, so the decision
//     never reads a flag that another thread is writing.
//
// Both rules only add conditions to the old one (retire at the first count
// >= minSamples that passes), and the per-pixel sample sequence does not
// depend on when neighbours stop, so a pixel can never stop earlier than it
// did before: it stops at the same count or later.

#ifndef PXR_IMAGING_PLUGIN_HD_EMBREE_ADAPTIVE_CONVERGENCE_H
#define PXR_IMAGING_PLUGIN_HD_EMBREE_ADAPTIVE_CONVERGENCE_H

#include "pxr/base/gf/vec3f.h"
#include "pxr/pxr.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

PXR_NAMESPACE_OPEN_SCOPE
namespace ty {

// Absolute standard-error floor of the adaptive test (pre-exposure linear
// radiance). The variance-of-mean floor is its square.
constexpr float kAdaptiveAbsoluteStdError = 0.001f;
constexpr float kAdaptiveAbsoluteVarianceOfMean =
    kAdaptiveAbsoluteStdError * kAdaptiveAbsoluteStdError;

/// Per-channel convergence test on the variance of the mean.
inline bool
IsPerChannelVarianceConverged(
    GfVec3f const& varOfMean,
    GfVec3f const& mean,
    float relativeVarianceThreshold)
{
    const float threshold = std::max(0.0f, relativeVarianceThreshold);
    for (int c = 0; c < 3; ++c) {
        const float meanMagnitude = std::abs(mean[c]);
        const float varianceLimit =
            kAdaptiveAbsoluteVarianceOfMean
            + threshold * meanMagnitude * meanMagnitude;
        if (varOfMean[c] > varianceLimit) {
            return false;
        }
    }
    return true;
}

/// True exactly when \p count is minSamples * 2^k for some k >= 0.
/// A non-positive \p minSamples is treated as 1.
inline bool
IsAdaptiveCheckpoint(uint32_t count, int minSamples)
{
    const uint32_t base =
        static_cast<uint32_t>(std::max(1, minSamples));
    if (count < base || count % base != 0) {
        return false;
    }
    const uint32_t multiple = count / base;
    return (multiple & (multiple - 1)) == 0;
}

/// Adds one sample to a pixel's Welford statistics.
/// \return The new sample count.
inline uint32_t
AccumulateAdaptiveSample(
    GfVec3f* mean, GfVec3f* m2, uint32_t* count, GfVec3f const& rgb)
{
    const uint32_t n = ++(*count);
    const GfVec3f delta = rgb - *mean;
    *mean += delta / static_cast<float>(n);
    const GfVec3f delta2 = rgb - *mean;
    *m2 += GfCompMult(delta, delta2);
    return n;
}

/// Evaluates the (unchanged) adaptive test on a pixel's statistics.
inline bool
PassesAdaptiveTest(
    GfVec3f const& mean, GfVec3f const& m2, uint32_t count,
    float relativeVarianceThreshold)
{
    if (count == 0) {
        return false;
    }
    const float fCount = static_cast<float>(count);
    const GfVec3f varOfMean = m2 / (fCount * fCount);
    return IsPerChannelVarianceConverged(
        varOfMean, mean, relativeVarianceThreshold);
}

/// The per-sample step used by the render workers: accumulates the sample
/// and, only at a checkpoint count, overwrites the pixel's pass flag with
/// the current test result. Between checkpoints the flag keeps the latest
/// checkpoint's result. Never retires the pixel itself.
inline void
UpdateAdaptivePixel(
    GfVec3f* mean, GfVec3f* m2, uint32_t* count, uint8_t* passFlag,
    GfVec3f const& rgb, int minSamples, float relativeVarianceThreshold)
{
    const uint32_t n = AccumulateAdaptiveSample(mean, m2, count, rgb);
    if (IsAdaptiveCheckpoint(n, minSamples)) {
        *passFlag = PassesAdaptiveTest(
            *mean, *m2, n, relativeVarianceThreshold) ? 1 : 0;
    }
}

/// Pixel-buffer rectangle [minX, maxX) x [minY, maxY) that is sampled.
struct AdaptiveWindow {
    unsigned int minX = 0;
    unsigned int minY = 0;
    unsigned int maxX = 0;
    unsigned int maxY = 0;
};

/// True when (x, y), which must lie in \p window, may be retired: it sits at
/// a checkpoint count, and its own pass flag and the flags of all its
/// 8-neighbours inside \p window are set. Neighbours outside the window are
/// never sampled and are ignored.
inline bool
ShouldRetireAdaptivePixel(
    uint8_t const* passFlags, uint32_t const* sampleCounts,
    size_t width, AdaptiveWindow const& window,
    unsigned int x, unsigned int y, int minSamples)
{
    const size_t idx = static_cast<size_t>(y) * width + x;
    if (!passFlags[idx] ||
        !IsAdaptiveCheckpoint(sampleCounts[idx], minSamples)) {
        return false;
    }
    const unsigned int x0 = (x > window.minX) ? x - 1 : x;
    const unsigned int y0 = (y > window.minY) ? y - 1 : y;
    const unsigned int x1 = std::min(x + 2, window.maxX);
    const unsigned int y1 = std::min(y + 2, window.maxY);
    for (unsigned int ny = y0; ny < y1; ++ny) {
        const size_t row = static_cast<size_t>(ny) * width;
        for (unsigned int nx = x0; nx < x1; ++nx) {
            if (!passFlags[row + nx]) {
                return false;
            }
        }
    }
    return true;
}

/// Retirement sweep over window rows [rowBegin, rowEnd) (absolute buffer
/// rows, clipped to the window). Reads only \p passFlags / \p sampleCounts
/// and writes only \p converged entries of its own rows, so disjoint row
/// ranges can be swept concurrently. Already-retired pixels are never
/// un-retired.
/// \return Number of pixels newly retired (false -> true) by this call.
inline uint64_t
RetireAgreedAdaptivePixels(
    uint8_t const* passFlags, uint32_t const* sampleCounts,
    uint8_t* converged, size_t width, AdaptiveWindow const& window,
    int minSamples, unsigned int rowBegin, unsigned int rowEnd)
{
    uint64_t retired = 0;
    const unsigned int yBegin = std::max(rowBegin, window.minY);
    const unsigned int yEnd = std::min(rowEnd, window.maxY);
    for (unsigned int y = yBegin; y < yEnd; ++y) {
        for (unsigned int x = window.minX; x < window.maxX; ++x) {
            const size_t idx = static_cast<size_t>(y) * width + x;
            if (converged[idx]) {
                continue;
            }
            if (ShouldRetireAdaptivePixel(
                    passFlags, sampleCounts, width, window, x, y,
                    minSamples)) {
                converged[idx] = 1;
                ++retired;
            }
        }
    }
    return retired;
}

} // namespace ty
PXR_NAMESPACE_CLOSE_SCOPE

#endif // PXR_IMAGING_PLUGIN_HD_EMBREE_ADAPTIVE_CONVERGENCE_H
