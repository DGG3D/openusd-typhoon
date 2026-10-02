//
// Copyright 2026 Pixar
//
// Licensed under the terms set forth in the LICENSE.txt file available at
// https://openusd.org/license.
//
// Modified by DGG3D 2026: new file; regression tests for checkpointed,
// neighbourhood-agreed adaptive stopping (renderer/aov/adaptiveConvergence.h).
// Also: unit tests for the remaining-time estimator
// (renderer/aov/adaptiveTimeEstimate.h), including its work-prediction
// model.
//
#include <renderer/aov/adaptiveConvergence.h>
#include <renderer/aov/adaptiveTimeEstimate.h>

#include "pxr/base/gf/vec3f.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

// ---------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------

bool
TestCheckpointsAreMinSamplesTimesPowersOfTwo()
{
    bool ok = true;
    auto expect = [&ok](uint32_t count, int minSamples, bool expected) {
        if (ty::IsAdaptiveCheckpoint(count, minSamples) != expected) {
            std::printf("    IsAdaptiveCheckpoint(%u, %d) != %s\n",
                        count, minSamples, expected ? "true" : "false");
            ok = false;
        }
    };

    // Exhaustive against a reference for a few bases.
    const int bases[] = {1, 3, 64, 128, 1000};
    for (int base : bases) {
        for (uint32_t count = 0; count <= 70000; ++count) {
            bool reference = false;
            for (uint64_t c = static_cast<uint64_t>(base); c <= count;
                 c *= 2) {
                if (c == count) {
                    reference = true;
                }
            }
            if (ty::IsAdaptiveCheckpoint(count, base) != reference) {
                std::printf("    IsAdaptiveCheckpoint(%u, %d) != %s\n",
                            count, base, reference ? "true" : "false");
                return false;
            }
        }
    }

    expect(128, 128, true);
    expect(256, 128, true);
    expect(8192, 128, true);
    expect(127, 128, false);
    expect(129, 128, false);
    expect(384, 128, false);
    expect(0, 128, false);
    // minSamples <= 0 is treated as 1.
    expect(0, 0, false);
    expect(1, 0, true);
    expect(2, -5, true);
    expect(3, 0, false);
    expect(1024, -1, true);
    // Large counts do not overflow.
    expect(0x80000000u, 1, true);
    expect(0x80000000u, 128, true);
    expect(0xFFFFFFFFu, 1, false);
    return ok;
}

// ---------------------------------------------------------------------------
// Per-sample update
// ---------------------------------------------------------------------------

// The per-sample update and test exactly as they were in
// ty::Renderer::_UpdateVariance before checkpointing was added.
void
_LegacyAccumulate(GfVec3f* mean, GfVec3f* m2, uint32_t* count,
                  GfVec3f const& rgb)
{
    uint32_t n = ++(*count);
    GfVec3f delta = rgb - *mean;
    *mean += delta / static_cast<float>(n);
    GfVec3f delta2 = rgb - *mean;
    *m2 += GfCompMult(delta, delta2);
}

bool
TestUpdateKeepsEstimatorAndOnlyTestsAtCheckpoints()
{
    const int minSamples = 4;
    const float threshold = 0.002f;

    GfVec3f mean(0.0f), m2(0.0f), legacyMean(0.0f), legacyM2(0.0f);
    uint32_t count = 0, legacyCount = 0;
    const uint8_t kSentinel = 7;
    uint8_t pass = kSentinel;

    for (uint32_t i = 0; i < 40; ++i) {
        const GfVec3f rgb(
            0.25f * static_cast<float>(i % 3),
            0.1f + 0.01f * static_cast<float>(i),
            (i % 5 == 0) ? 3.0f : 0.0f);
        const uint8_t before = pass;
        ty::UpdateAdaptivePixel(
            &mean, &m2, &count, &pass, rgb, minSamples, threshold);
        _LegacyAccumulate(&legacyMean, &legacyM2, &legacyCount, rgb);

        if (count != legacyCount || mean != legacyMean || m2 != legacyM2) {
            std::printf("    Welford statistics diverged at sample %u\n",
                        count);
            return false;
        }
        if (ty::IsAdaptiveCheckpoint(count, minSamples)) {
            const bool expected = ty::PassesAdaptiveTest(
                mean, m2, count, threshold);
            if (pass != (expected ? 1 : 0)) {
                std::printf("    pass flag wrong at checkpoint %u\n", count);
                return false;
            }
        } else if (pass != before) {
            std::printf("    pass flag changed at non-checkpoint %u\n",
                        count);
            return false;
        }
    }
    return true;
}

bool
TestVarianceTestIsUnchanged()
{
    // varOfMean = M2 / n^2 against 1e-6 + threshold * mean^2 per channel.
    const float threshold = 0.002f;
    const uint32_t n = 1000;
    const float limitAtZeroMean = 1e-6f;
    const float nf = static_cast<float>(n);

    const GfVec3f mean(0.0f);
    const GfVec3f m2Pass(0.99f * limitAtZeroMean * nf * nf);
    const GfVec3f m2Fail(1.01f * limitAtZeroMean * nf * nf);
    if (!ty::PassesAdaptiveTest(mean, m2Pass, n, threshold) ||
        ty::PassesAdaptiveTest(mean, m2Fail, n, threshold)) {
        std::printf("    absolute floor changed\n");
        return false;
    }

    // With mean 1 the relative term (0.002) dominates.
    const GfVec3f meanOne(1.0f);
    const float limit = limitAtZeroMean + threshold;
    const GfVec3f m2RelPass(0.99f * limit * nf * nf);
    const GfVec3f m2RelFail(1.01f * limit * nf * nf);
    if (!ty::PassesAdaptiveTest(meanOne, m2RelPass, n, threshold) ||
        ty::PassesAdaptiveTest(meanOne, m2RelFail, n, threshold)) {
        std::printf("    relative threshold changed\n");
        return false;
    }

    // One failing channel fails the pixel.
    const GfVec3f m2Mixed(m2Pass[0], m2Fail[1], m2Pass[2]);
    if (ty::PassesAdaptiveTest(mean, m2Mixed, n, threshold)) {
        std::printf("    per-channel test changed\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Neighbourhood sweep
// ---------------------------------------------------------------------------

struct _Grid {
    _Grid(unsigned int w, unsigned int h, uint32_t initialCount)
        : width(w), height(h)
        , pass(static_cast<size_t>(w) * h, uint8_t(0))
        , counts(static_cast<size_t>(w) * h, initialCount)
        , converged(static_cast<size_t>(w) * h, uint8_t(0))
    {
        window.minX = 0;
        window.minY = 0;
        window.maxX = w;
        window.maxY = h;
    }

    uint8_t& Pass(unsigned int x, unsigned int y)
    {
        return pass[static_cast<size_t>(y) * width + x];
    }

    bool Converged(unsigned int x, unsigned int y) const
    {
        return converged[static_cast<size_t>(y) * width + x] != 0;
    }

    void SetBlock(unsigned int x0, unsigned int y0,
                  unsigned int x1, unsigned int y1)
    {
        for (unsigned int y = y0; y < y1; ++y) {
            for (unsigned int x = x0; x < x1; ++x) {
                Pass(x, y) = 1;
            }
        }
    }

    uint64_t Sweep(int minSamples)
    {
        return ty::RetireAgreedAdaptivePixels(
            pass.data(), counts.data(), converged.data(), width, window,
            minSamples, 0, height);
    }

    size_t CountConverged() const
    {
        size_t n = 0;
        for (uint8_t c : converged) {
            n += c ? 1 : 0;
        }
        return n;
    }

    unsigned int width;
    unsigned int height;
    ty::AdaptiveWindow window;
    std::vector<uint8_t> pass;
    std::vector<uint32_t> counts;
    std::vector<uint8_t> converged;
};

bool
TestSinglePassingPixelWithFailingNeighbourIsNotRetired()
{
    const int minSamples = 16;

    // Only the centre passes.
    _Grid lone(5, 5, minSamples);
    lone.Pass(2, 2) = 1;
    if (lone.Sweep(minSamples) != 0 || lone.CountConverged() != 0) {
        std::printf("    isolated passing pixel was retired\n");
        return false;
    }

    // Whole 3x3 block passes except one diagonal neighbour.
    _Grid almost(5, 5, minSamples);
    almost.SetBlock(1, 1, 4, 4);
    almost.Pass(3, 1) = 0;
    if (almost.Sweep(minSamples) != 0 || almost.Converged(2, 2)) {
        std::printf("    pixel with one failing neighbour was retired\n");
        return false;
    }
    return true;
}

bool
TestPassingBlockRetiresOnlyItsCentre()
{
    const int minSamples = 16;
    _Grid grid(5, 5, minSamples);
    grid.SetBlock(1, 1, 4, 4);

    const uint64_t retired = grid.Sweep(minSamples);
    if (retired != 1 || !grid.Converged(2, 2) || grid.CountConverged() != 1) {
        std::printf("    expected exactly the 3x3 centre to retire "
                    "(retired=%llu)\n",
                    static_cast<unsigned long long>(retired));
        return false;
    }

    // A second sweep must not count the same pixel again.
    if (grid.Sweep(minSamples) != 0 || grid.CountConverged() != 1) {
        std::printf("    retired pixel counted twice\n");
        return false;
    }

    // A later failing flag never un-retires a pixel.
    grid.Pass(2, 2) = 0;
    grid.Pass(1, 1) = 0;
    grid.Sweep(minSamples);
    if (!grid.Converged(2, 2)) {
        std::printf("    retired pixel was un-retired\n");
        return false;
    }
    return true;
}

bool
TestOnlyCheckpointCountsRetire()
{
    const int minSamples = 16;
    _Grid grid(3, 3, minSamples + 1);
    grid.SetBlock(0, 0, 3, 3);
    if (grid.Sweep(minSamples) != 0) {
        std::printf("    pixel retired at a non-checkpoint count\n");
        return false;
    }
    grid.counts.assign(grid.counts.size(), 2 * minSamples);
    if (grid.Sweep(minSamples) != 9) {
        std::printf("    pixels did not retire at a checkpoint count\n");
        return false;
    }
    return true;
}

bool
TestBufferAndDataWindowBordersIgnoreOutsideNeighbours()
{
    const int minSamples = 16;

    // Buffer border: a fully passing buffer retires every pixel, including
    // corners and edges, which have fewer neighbours.
    _Grid full(4, 3, minSamples);
    full.SetBlock(0, 0, 4, 3);
    if (full.Sweep(minSamples) != 12) {
        std::printf("    fully passing buffer did not fully retire\n");
        return false;
    }

    // Data window [1,4) x [1,4) inside a 6x6 buffer. Pixels outside the
    // window are never sampled (flag 0) and must be ignored.
    _Grid win(6, 6, minSamples);
    win.window.minX = 1;
    win.window.minY = 1;
    win.window.maxX = 4;
    win.window.maxY = 4;
    win.SetBlock(1, 1, 4, 4);
    if (win.Sweep(minSamples) != 9) {
        std::printf("    window-border pixels not retired\n");
        return false;
    }
    for (unsigned int y = 0; y < 6; ++y) {
        for (unsigned int x = 0; x < 6; ++x) {
            const bool inside = x >= 1 && x < 4 && y >= 1 && y < 4;
            if (win.Converged(x, y) != inside) {
                std::printf("    wrong retirement at (%u, %u)\n", x, y);
                return false;
            }
        }
    }

    // Passing flags outside the window never retire anything there, and a
    // failing in-window neighbour still blocks a window-corner pixel.
    _Grid win2(6, 6, minSamples);
    win2.window = win.window;
    win2.SetBlock(0, 0, 6, 6);
    win2.Pass(2, 1) = 0;
    win2.Sweep(minSamples);
    if (win2.Converged(1, 1) || win2.Converged(0, 0) ||
        win2.Converged(5, 5) || !win2.Converged(3, 3)) {
        std::printf("    window border handling wrong\n");
        return false;
    }
    return true;
}

bool
TestRowRangeSweepsMatchFullSweep()
{
    const int minSamples = 8;
    _Grid a(7, 6, minSamples);
    // Irregular pattern.
    for (unsigned int y = 0; y < 6; ++y) {
        for (unsigned int x = 0; x < 7; ++x) {
            a.Pass(x, y) = ((x * 7 + y * 3) % 5 != 0) ? 1 : 0;
        }
    }
    _Grid b = a;
    const uint64_t full = a.Sweep(minSamples);
    uint64_t split = 0;
    const unsigned int cuts[] = {0, 1, 4, 6};
    for (int i = 0; i < 3; ++i) {
        split += ty::RetireAgreedAdaptivePixels(
            b.pass.data(), b.counts.data(), b.converged.data(), b.width,
            b.window, minSamples, cuts[i], cuts[i + 1]);
    }
    if (full != split || a.converged != b.converged) {
        std::printf("    row-split sweep differs from full sweep\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Statistical regression: rare-spike estimator
// ---------------------------------------------------------------------------

uint64_t
_SplitMix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Deterministic per-(pixel, sample index) Bernoulli spike, mirroring the
// renderer's sampler keyed by (seed, x, y, sampleNum): a pixel's sample
// sequence does not depend on when other pixels stop.
float
_SpikeSample(uint64_t seed, size_t pixel, uint32_t sampleIndex, double p)
{
    const uint64_t h = _SplitMix64(
        _SplitMix64(seed ^ (static_cast<uint64_t>(pixel) << 20)) +
        sampleIndex);
    const double u = static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0);
    return (u < p) ? 1.0f : 0.0f;
}

struct _SpikeConfig {
    unsigned int width = 64;
    unsigned int height = 64;
    double p = 0.005;
    uint32_t sampleCap = 8192;
    int minSamples = 128;
    float threshold = 0.002f;
    uint64_t seed = 0x5EEDu;
};

struct _SpikeResult {
    std::vector<float> mean;
    std::vector<uint32_t> count;
};

// Previous rule: test after every sample from minSamples on, retire at the
// first pass.
_SpikeResult
_RunPerSampleRule(_SpikeConfig const& cfg)
{
    const size_t numPixels = static_cast<size_t>(cfg.width) * cfg.height;
    _SpikeResult result;
    result.mean.resize(numPixels);
    result.count.resize(numPixels);
    for (size_t pixel = 0; pixel < numPixels; ++pixel) {
        GfVec3f mean(0.0f), m2(0.0f);
        uint32_t count = 0;
        for (uint32_t s = 0; s < cfg.sampleCap; ++s) {
            const float v = _SpikeSample(cfg.seed, pixel, s, cfg.p);
            ty::AccumulateAdaptiveSample(&mean, &m2, &count, GfVec3f(v));
            if (count >= static_cast<uint32_t>(cfg.minSamples) &&
                ty::PassesAdaptiveTest(mean, m2, count, cfg.threshold)) {
                break;
            }
        }
        result.mean[pixel] = mean[0];
        result.count[pixel] = count;
    }
    return result;
}

// Current rule, driven exactly like ty::Renderer::Render: one sample per
// unretired pixel per pass through ty::UpdateAdaptivePixel, then the
// retirement sweep between passes.
_SpikeResult
_RunCheckpointNeighbourhoodRule(_SpikeConfig const& cfg)
{
    const size_t numPixels = static_cast<size_t>(cfg.width) * cfg.height;
    std::vector<GfVec3f> mean(numPixels, GfVec3f(0.0f));
    std::vector<GfVec3f> m2(numPixels, GfVec3f(0.0f));
    std::vector<uint32_t> count(numPixels, 0);
    std::vector<uint8_t> pass(numPixels, 0);
    std::vector<uint8_t> converged(numPixels, 0);
    ty::AdaptiveWindow window;
    window.maxX = cfg.width;
    window.maxY = cfg.height;

    uint64_t convergedCount = 0;
    for (uint32_t s = 0; s < cfg.sampleCap; ++s) {
        for (size_t pixel = 0; pixel < numPixels; ++pixel) {
            if (converged[pixel]) {
                continue;
            }
            const float v = _SpikeSample(cfg.seed, pixel, s, cfg.p);
            ty::UpdateAdaptivePixel(
                &mean[pixel], &m2[pixel], &count[pixel], &pass[pixel],
                GfVec3f(v), cfg.minSamples, cfg.threshold);
        }
        convergedCount += ty::RetireAgreedAdaptivePixels(
            pass.data(), count.data(), converged.data(), cfg.width, window,
            cfg.minSamples, 0, cfg.height);
        if (convergedCount == numPixels) {
            break;
        }
    }

    _SpikeResult result;
    result.mean.resize(numPixels);
    result.count = count;
    for (size_t pixel = 0; pixel < numPixels; ++pixel) {
        result.mean[pixel] = mean[pixel][0];
    }
    return result;
}

struct _SpikeSummary {
    double darkFraction = 0.0;
    double populationMean = 0.0;
};

_SpikeSummary
_Summarize(_SpikeResult const& r, double p)
{
    _SpikeSummary s;
    size_t dark = 0;
    double sum = 0.0;
    for (float m : r.mean) {
        if (m < 0.25 * p) {
            ++dark;
        }
        sum += m;
    }
    s.darkFraction = static_cast<double>(dark) / r.mean.size();
    s.populationMean = sum / r.mean.size();
    return s;
}

// Shared checks for one configuration. maxNewDarkFraction and
// minOldDarkFraction encode the expected behaviour of the two rules.
bool
_CheckSpikeConfig(_SpikeConfig const& cfg,
                  double maxNewDarkFraction,
                  double minOldDarkFraction)
{
    const _SpikeResult oldRule = _RunPerSampleRule(cfg);
    const _SpikeResult newRule = _RunCheckpointNeighbourhoodRule(cfg);
    const _SpikeSummary oldSummary = _Summarize(oldRule, cfg.p);
    const _SpikeSummary newSummary = _Summarize(newRule, cfg.p);

    std::printf("    minSamples=%d: per-sample rule dark=%.4f%% mean/p=%.4f;"
                " checkpoint+neighbourhood dark=%.4f%% mean/p=%.4f\n",
                cfg.minSamples,
                100.0 * oldSummary.darkFraction,
                oldSummary.populationMean / cfg.p,
                100.0 * newSummary.darkFraction,
                newSummary.populationMean / cfg.p);

    bool ok = true;

    // The new rule may only delay stopping: with identical per-pixel
    // sample sequences, no pixel stops earlier than under the old rule.
    for (size_t i = 0; i < newRule.count.size(); ++i) {
        if (newRule.count[i] < oldRule.count[i]) {
            std::printf("    pixel %zu stopped earlier (%u < %u)\n",
                        i, newRule.count[i], oldRule.count[i]);
            ok = false;
            break;
        }
    }

    if (newSummary.darkFraction >= maxNewDarkFraction) {
        std::printf("    too many dark pixels with the new rule\n");
        ok = false;
    }
    const double relErr =
        std::abs(newSummary.populationMean - cfg.p) / cfg.p;
    if (relErr > 0.02) {
        std::printf("    population mean off by %.2f%%\n", 100.0 * relErr);
        ok = false;
    }

    // Documents the bug: the per-sample rule leaves clearly more pixels
    // stuck dark.
    if (oldSummary.darkFraction < minOldDarkFraction ||
        oldSummary.darkFraction <
            10.0 * std::max(newSummary.darkFraction, 1e-4)) {
        std::printf("    per-sample rule unexpectedly not worse\n");
        ok = false;
    }
    return ok;
}

// The configuration from the bug report's regression plan: minSamples 128,
// threshold 0.002, p = 0.005. At the first checkpoint a pixel whose nine
// neighbourhood pixels all saw zero spikes has exactly zero variance and
// legitimately passes: probability 0.995^(128*9) ~= 0.31% for an interior
// pixel (more at borders), so ~0.43% of a 64x64 grid is expected to retire
// black. The per-sample rule retires every pixel with zero spikes in its
// first 128 samples: 0.995^128 ~= 53%.
bool
TestSpikeEstimatorMinSamples128()
{
    _SpikeConfig cfg;
    cfg.minSamples = 128;
    return _CheckSpikeConfig(cfg,
                             /*maxNewDarkFraction=*/0.01,
                             /*minOldDarkFraction=*/0.20);
}

// With a larger minSamples the all-zero neighbourhood is negligible and the
// new rule leaves essentially no dark pixels (< 0.1%), while the per-sample
// rule still retires every pixel with <= 1 spike at 1024 samples (~3.7%).
bool
TestSpikeEstimatorMinSamples1024()
{
    _SpikeConfig cfg;
    cfg.minSamples = 1024;
    cfg.seed = 0xC0FFEEu;
    return _CheckSpikeConfig(cfg,
                             /*maxNewDarkFraction=*/0.001,
                             /*minOldDarkFraction=*/0.01);
}

// ---------------------------------------------------------------------------
// Remaining-time estimate
// ---------------------------------------------------------------------------

bool
IsNear(double actual, double expected, double relTol)
{
    const double scale = std::max(1.0, std::abs(expected));
    return std::abs(actual - expected) <= relTol * scale;
}

bool
ExpectNear(char const* what, double actual, double expected,
           double relTol = 1e-9)
{
    if (!IsNear(actual, expected, relTol)) {
        std::printf("    %s: got %.12g, expected %.12g\n",
                    what, actual, expected);
        return false;
    }
    return true;
}

bool
TestEtaConstantPassTimeWithoutRetirement()
{
    bool ok = true;
    const double t = 0.02;
    const int n = 256;
    const int minSamples = 64;

    // Pure function: b = 0, no retirement -> (N - s) * t for any s,
    // including across checkpoints; nothing remains at or past N.
    ty::AdaptivePassTimeModel model;
    model.fixedSeconds = t;
    const int completed[] = {0, 1, 10, 63, 64, 65, 128, 200, 255};
    for (int s : completed) {
        ok &= ExpectNear("constant pass time",
            ty::PredictAdaptiveRemainingSeconds(
                model, s, 1000.0, n, minSamples, 1.0),
            (n - s) * t);
    }
    ok &= ExpectNear("s == N",
        ty::PredictAdaptiveRemainingSeconds(
            model, n, 1000.0, n, minSamples, 1.0), 0.0);
    ok &= ExpectNear("s > N",
        ty::PredictAdaptiveRemainingSeconds(
            model, n + 5, 1000.0, n, minSamples, 1.0), 0.0);

    // Estimator: every pass at the same active count and duration, no
    // checkpoint survival recorded (default ratio 1) -> the fit degenerates
    // to the mean pass time and the estimate is (N - s) * t.
    ty::AdaptiveTimeEstimator estimator;
    estimator.Reset();
    for (int pass = 1; pass <= 10; ++pass) {
        estimator.RecordPass(1000, t);
    }
    ok &= ExpectNear("default survival ratio",
                     estimator.GetSurvivalRatio(), 1.0);
    const ty::AdaptivePassTimeModel fit = estimator.Fit();
    ok &= ExpectNear("degenerate fit a", fit.fixedSeconds, t);
    ok &= ExpectNear("degenerate fit b", fit.secondsPerPixel, 0.0);
    ok &= ExpectNear("estimator constant pass time",
        estimator.EstimateRemainingSeconds(10, 1000, n, minSamples),
        (n - 10) * t);
    return ok;
}

bool
TestEtaFitRecoversLinearModel()
{
    bool ok = true;
    const double a = 0.005;
    const double b = 2.0e-6;

    // Synthetic passes at three active levels, fed pass by pass.
    struct Run { uint64_t active; int passes; };
    const Run runs[] = {{1000000, 63}, {400000, 64}, {100000, 128}};
    ty::AdaptiveTimeEstimator estimator;
    estimator.Reset();
    for (Run const& run : runs) {
        for (int pass = 0; pass < run.passes; ++pass) {
            estimator.RecordPass(
                run.active, a + b * static_cast<double>(run.active));
        }
    }
    if (estimator.GetTimedPassCount() != 63 + 64 + 128) {
        std::printf("    timed pass count %llu\n",
            static_cast<unsigned long long>(estimator.GetTimedPassCount()));
        ok = false;
    }
    ty::AdaptivePassTimeModel fit = estimator.Fit();
    ok &= ExpectNear("fit a", fit.fixedSeconds, a, 1e-7);
    ok &= ExpectNear("fit b", fit.secondsPerPixel / b, 1.0, 1e-7);

    // Symmetric +/- noise around each level's true time leaves the
    // least-squares fit unchanged.
    estimator.Reset();
    for (Run const& run : runs) {
        for (int pass = 0; pass < 2 * run.passes; ++pass) {
            const double noise = (pass % 2 == 0) ? 1e-3 : -1e-3;
            estimator.RecordPass(
                run.active,
                a + b * static_cast<double>(run.active) + noise);
        }
    }
    fit = estimator.Fit();
    ok &= ExpectNear("noisy fit a", fit.fixedSeconds, a, 1e-7);
    ok &= ExpectNear("noisy fit b", fit.secondsPerPixel / b, 1.0, 1e-7);

    // More levels than the ring holds: the oldest are dropped, and the
    // survivors still describe the same line.
    estimator.Reset();
    for (uint64_t level = 0; level < 2 * ty::AdaptiveTimeEstimator::kMaxLevels;
         ++level) {
        const uint64_t active = 1000 + 1000 * level;
        estimator.RecordPass(active, a + b * static_cast<double>(active));
        estimator.RecordPass(active, a + b * static_cast<double>(active));
    }
    fit = estimator.Fit();
    ok &= ExpectNear("ring fit a", fit.fixedSeconds, a, 1e-7);
    ok &= ExpectNear("ring fit b", fit.secondsPerPixel / b, 1.0, 1e-7);

    // Clamping. Time falling as active grows: slope clamped to 0, the
    // intercept is the mean pass time.
    {
        ty::AdaptivePassTimeLevel levels[2];
        levels[0].activePixels = 100.0;
        levels[0].passes = 2;
        levels[0].totalSeconds = 2.0 * 3.0;
        levels[1].activePixels = 300.0;
        levels[1].passes = 2;
        levels[1].totalSeconds = 2.0 * 1.0;
        fit = ty::FitAdaptivePassTime(levels, 2);
        ok &= ExpectNear("negative slope a", fit.fixedSeconds, 2.0);
        ok &= ExpectNear("negative slope b", fit.secondsPerPixel, 0.0);
    }
    // Negative intercept (x = 100: t = 0.5, x = 200: t = 2.0, one pass
    // each; the free fit has intercept -1): refit through the origin,
    // b = sum(x t) / sum(x^2).
    {
        ty::AdaptivePassTimeLevel levels[2];
        levels[0].activePixels = 100.0;
        levels[0].passes = 1;
        levels[0].totalSeconds = 0.5;
        levels[1].activePixels = 200.0;
        levels[1].passes = 1;
        levels[1].totalSeconds = 2.0;
        fit = ty::FitAdaptivePassTime(levels, 2);
        ok &= ExpectNear("negative intercept a", fit.fixedSeconds, 0.0);
        ok &= ExpectNear("negative intercept b", fit.secondsPerPixel,
            (100.0 * 0.5 + 200.0 * 2.0) / (100.0 * 100.0 + 200.0 * 200.0));
    }
    // No data: zero model.
    fit = ty::FitAdaptivePassTime(nullptr, 0);
    ok &= ExpectNear("empty fit a", fit.fixedSeconds, 0.0);
    ok &= ExpectNear("empty fit b", fit.secondsPerPixel, 0.0);
    return ok;
}

bool
TestEtaHalfRetirementPerCheckpoint()
{
    bool ok = true;
    ty::AdaptivePassTimeModel model;
    model.fixedSeconds = 0.1;
    model.secondsPerPixel = 0.001;

    // minSamples 4, N 40, s = 5 (just past checkpoint 4), 800 active,
    // half retired at each later checkpoint (8, 16, 32; 40 is not one):
    //   passes  6.. 8: 3  * (0.1 + 0.800) = 2.7
    //   passes  9..16: 8  * (0.1 + 0.400) = 4.0
    //   passes 17..32: 16 * (0.1 + 0.200) = 4.8
    //   passes 33..40: 8  * (0.1 + 0.100) = 1.6    total 13.1
    ok &= ExpectNear("half retirement",
        ty::PredictAdaptiveRemainingSeconds(model, 5, 800.0, 40, 4, 0.5),
        13.1);

    // Before the first checkpoint every pixel stays active until it:
    //   passes  1.. 4: 4  * (0.1 + 0.800) = 3.6
    //   passes  5.. 8: 4  * (0.1 + 0.400) = 2.0
    //   passes  9..16: 8  * (0.1 + 0.200) = 2.4
    //   passes 17..32: 16 * (0.1 + 0.100) = 3.2
    //   passes 33..40: 8  * (0.1 + 0.050) = 1.2    total 12.4
    ok &= ExpectNear("half retirement from start",
        ty::PredictAdaptiveRemainingSeconds(model, 0, 800.0, 40, 4, 0.5),
        12.4);

    // Sitting exactly on a checkpoint: its retirement is already in the
    // active count, so the next drop is at the following checkpoint.
    //   passes  9..16: 8  * (0.1 + 0.400) = 4.0
    //   passes 17..32: 16 * (0.1 + 0.200) = 4.8
    //   passes 33..40: 8  * (0.1 + 0.100) = 1.6    total 10.4
    ok &= ExpectNear("on a checkpoint",
        ty::PredictAdaptiveRemainingSeconds(model, 8, 400.0, 40, 4, 0.5),
        10.4);

    // The frame ends once every pixel is predicted retired (active < 0.5):
    // minSamples 1, 2 active, a = 1, b = 0, N = 100, s = 1:
    //   pass 2 (2 active), passes 3..4 (1), passes 5..8 (0.5), then 0.25.
    {
        ty::AdaptivePassTimeModel unit;
        unit.fixedSeconds = 1.0;
        ok &= ExpectNear("all retired",
            ty::PredictAdaptiveRemainingSeconds(unit, 1, 2.0, 100, 1, 0.5),
            7.0);
        ok &= ExpectNear("nothing active",
            ty::PredictAdaptiveRemainingSeconds(unit, 1, 0.0, 100, 1, 0.5),
            0.0);
    }

    // Through the estimator, with the ratio observed at checkpoint 4 and
    // the model fitted from the timings (a = 0.1, b = 0.001 exactly):
    //   passes  6.. 8: 3  * (0.1 + 0.5000) = 1.8
    //   passes  9..16: 8  * (0.1 + 0.2500) = 2.8
    //   passes 17..32: 16 * (0.1 + 0.1250) = 3.6
    //   passes 33..40: 8  * (0.1 + 0.0625) = 1.3    total 9.5
    ty::AdaptiveTimeEstimator estimator;
    estimator.Reset();
    for (int pass = 1; pass <= 4; ++pass) {
        estimator.RecordPass(1000, 1.1);
    }
    estimator.RecordCheckpointSurvival(1000, 500);
    estimator.RecordPass(500, 0.6);
    ok &= ExpectNear("observed ratio", estimator.GetSurvivalRatio(), 0.5);
    ok &= ExpectNear("estimator half retirement",
        estimator.EstimateRemainingSeconds(5, 500, 40, 4), 9.5, 1e-9);

    // The most recent observation wins.
    estimator.RecordCheckpointSurvival(500, 400);
    ok &= ExpectNear("latest ratio", estimator.GetSurvivalRatio(), 0.8);
    estimator.RecordCheckpointSurvival(0, 0);
    ok &= ExpectNear("ignored empty checkpoint",
                     estimator.GetSurvivalRatio(), 0.8);
    return ok;
}

bool
TestEtaUnknownUntilEnoughPassesTimed()
{
    bool ok = true;
    ty::AdaptiveTimeEstimator estimator;
    estimator.Reset();
    if (estimator.EstimateRemainingSeconds(0, 1000, 256, 64) >= 0.0) {
        std::printf("    estimate known with no passes timed\n");
        ok = false;
    }
    for (uint64_t pass = 1;
         pass < ty::AdaptiveTimeEstimator::kMinTimedPasses; ++pass) {
        estimator.RecordPass(1000, 0.02);
        if (estimator.EstimateRemainingSeconds(
                static_cast<int>(pass), 1000, 256, 64) >= 0.0) {
            std::printf("    estimate known after %llu timed passes\n",
                        static_cast<unsigned long long>(pass));
            ok = false;
        }
    }
    // Invalid durations are not counted.
    estimator.RecordPass(1000, -1.0);
    estimator.RecordPass(1000, std::nan(""));
    if (estimator.EstimateRemainingSeconds(3, 1000, 256, 64) >= 0.0) {
        std::printf("    invalid durations were counted\n");
        ok = false;
    }
    estimator.RecordPass(1000, 0.02);
    const int s = static_cast<int>(ty::AdaptiveTimeEstimator::kMinTimedPasses);
    ok &= ExpectNear("known after enough passes",
        estimator.EstimateRemainingSeconds(s, 1000, 256, 64),
        (256 - s) * 0.02);

    estimator.Reset();
    if (estimator.EstimateRemainingSeconds(s, 1000, 256, 64) >= 0.0) {
        std::printf("    estimate known after Reset()\n");
        ok = false;
    }
    ok &= ExpectNear("ratio after Reset()", estimator.GetSurvivalRatio(), 1.0);
    return ok;
}

bool
TestEtaFromWorkPrediction()
{
    bool ok = true;

    // Levels end at the checkpoints 64, 128 and at the cap 256.
    ty::AdaptiveWorkPrediction work;
    work.numLevels = ty::BuildAdaptiveLevels(64, 256, work.levelEndPass);
    ok &= ExpectNear("level count", work.numLevels, 3.0);
    ok &= ExpectNear("level 0 end", double(work.levelEndPass[0]), 64.0);
    ok &= ExpectNear("level 2 end", double(work.levelEndPass[2]), 256.0);
    ok &= ExpectNear("level of pass 65", ty::AdaptiveLevelOfPass(
        65, work.levelEndPass, work.numLevels), 1.0);

    // Passing count: the variance of the mean falls as 1 / n, so the test
    // limit is reached at m2 / (count * limit) samples.
    const GfVec3f mean(0.5f);
    const double limit = 1e-6 + 0.01 * 0.25;
    const GfVec3f m2(static_cast<float>(100.0 * 16.0 * limit), 0.0f, 0.0f);
    ok &= ExpectNear("passing count",
        ty::PredictAdaptivePassingCount(mean, m2, 16, 0.01f), 100.0, 1e-6);
    ok &= ExpectNear("constant pixel passes now",
        ty::PredictAdaptivePassingCount(mean, GfVec3f(0.0f), 16, 0.01f),
        0.0);

    // Before the makespan scale is measured each level costs overhead +
    // max(w, 1 / tiles) * traceCpuSeconds per pass: w = 0.25 for level 0
    // (10 tiles), 1/2 and 1/1 for the others.
    work.completedPasses = 32;
    work.currentLevel = 0;
    const uint64_t pixels[] = {100, 10, 1};
    const uint64_t tiles[] = {10, 2, 1};
    const double cpu[] = {0.1, 0.02, 0.01};
    const double makespan[] = {0.02, 0.01, 0.005};
    for (int level = 0; level < 3; ++level) {
        work.activePixels[level] = pixels[level];
        work.activeTiles[level] = tiles[level];
        work.traceCpuSeconds[level] = cpu[level];
        work.traceWallSeconds[level] = makespan[level];
    }
    ty::AdaptivePassCostRates rates;
    rates.overheadSeconds = 0.001;
    rates.traceWallPerCpuSecond = 0.25;
    const double expected = 32 * (0.001 + 0.25 * 0.1)
        + 64 * (0.001 + 0.5 * 0.02) + 128 * (0.001 + 1.0 * 0.01);
    ok &= ExpectNear("work eta",
        ty::PredictAdaptiveRemainingSecondsFromWork(work, 32, rates),
        expected);
    ok &= ExpectNear("work eta in level 1",
        ty::PredictAdaptiveRemainingSecondsFromWork(work, 100, rates),
        28 * (0.001 + 0.5 * 0.02) + 128 * (0.001 + 1.0 * 0.01));
    ok &= ExpectNear("work eta at cap",
        ty::PredictAdaptiveRemainingSecondsFromWork(work, 256, rates),
        0.0);
    ty::AdaptiveWorkPrediction retired = work;
    retired.activePixels[2] = 0;
    ok &= ExpectNear("work eta stops when nothing remains",
        ty::PredictAdaptiveRemainingSecondsFromWork(retired, 32, rates),
        32 * (0.001 + 0.25 * 0.1) + 64 * (0.001 + 0.5 * 0.02));

    // Once measured, the makespan scale replaces w.
    ty::AdaptivePassCostRates scaled = rates;
    scaled.traceWallScale = 2.0;
    ok &= ExpectNear("work eta with makespan scale",
        ty::PredictAdaptiveRemainingSecondsFromWork(work, 32, scaled),
        32 * (0.001 + 2.0 * 0.02) + 64 * (0.001 + 2.0 * 0.01)
        + 128 * (0.001 + 2.0 * 0.005));

    // Longest-first makespan: {5, 4, 3, 3, 3} on 2 threads -> 8 and 10.
    std::vector<double> jobs = {3.0, 5.0, 3.0, 4.0, 3.0};
    ok &= ExpectNear("makespan", ty::AdaptiveScheduleMakespan(&jobs, 2), 10.0);
    std::vector<double> oneJob = {7.0};
    ok &= ExpectNear("makespan bounded by largest job",
        ty::AdaptiveScheduleMakespan(&oneJob, 8), 7.0);

    // The estimator prefers the work prediction once a pass cost exists,
    // and scales the predicted makespan by the measured one: level-0 passes
    // traced in 0.025 s against a predicted 0.02 s -> scale 1.25.
    ty::AdaptiveTimeEstimator estimator;
    estimator.Reset();
    estimator.SetWorkPrediction(work);
    for (int pass = 30; pass < 34; ++pass) {
        estimator.RecordPass(100, 0.026);
        estimator.RecordPassCost(64, pass, 0.026, 0.025, 0.1);
    }
    ok &= ExpectNear("estimator uses work prediction",
        estimator.EstimateRemainingSeconds(32, 100, 256, 64),
        32 * (0.001 + 1.25 * 0.02) + 64 * (0.001 + 1.25 * 0.01)
        + 128 * (0.001 + 1.25 * 0.005));
    estimator.Reset();
    if (estimator.GetWorkPrediction().completedPasses != 0) {
        std::printf("    work prediction kept after Reset()\n");
        ok = false;
    }
    return ok;
}

} // namespace

int
main()
{
    struct Test {
        const char* name;
        bool (*fn)();
    };

    const Test tests[] = {
        {"AdaptiveConvergence.TestCheckpointsAreMinSamplesTimesPowersOfTwo",
         &TestCheckpointsAreMinSamplesTimesPowersOfTwo},
        {"AdaptiveConvergence.TestUpdateKeepsEstimatorAndOnlyTestsAtCheckpoints",
         &TestUpdateKeepsEstimatorAndOnlyTestsAtCheckpoints},
        {"AdaptiveConvergence.TestVarianceTestIsUnchanged",
         &TestVarianceTestIsUnchanged},
        {"AdaptiveConvergence.TestSinglePassingPixelWithFailingNeighbourIsNotRetired",
         &TestSinglePassingPixelWithFailingNeighbourIsNotRetired},
        {"AdaptiveConvergence.TestPassingBlockRetiresOnlyItsCentre",
         &TestPassingBlockRetiresOnlyItsCentre},
        {"AdaptiveConvergence.TestOnlyCheckpointCountsRetire",
         &TestOnlyCheckpointCountsRetire},
        {"AdaptiveConvergence.TestBufferAndDataWindowBordersIgnoreOutsideNeighbours",
         &TestBufferAndDataWindowBordersIgnoreOutsideNeighbours},
        {"AdaptiveConvergence.TestRowRangeSweepsMatchFullSweep",
         &TestRowRangeSweepsMatchFullSweep},
        {"AdaptiveConvergence.TestSpikeEstimatorMinSamples128",
         &TestSpikeEstimatorMinSamples128},
        {"AdaptiveConvergence.TestSpikeEstimatorMinSamples1024",
         &TestSpikeEstimatorMinSamples1024},
        {"AdaptiveConvergence.TestEtaConstantPassTimeWithoutRetirement",
         &TestEtaConstantPassTimeWithoutRetirement},
        {"AdaptiveConvergence.TestEtaFitRecoversLinearModel",
         &TestEtaFitRecoversLinearModel},
        {"AdaptiveConvergence.TestEtaHalfRetirementPerCheckpoint",
         &TestEtaHalfRetirementPerCheckpoint},
        {"AdaptiveConvergence.TestEtaUnknownUntilEnoughPassesTimed",
         &TestEtaUnknownUntilEnoughPassesTimed},
        {"AdaptiveConvergence.TestEtaFromWorkPrediction",
         &TestEtaFromWorkPrediction},
    };

    int failed = 0;
    for (const Test& test : tests) {
        std::printf("  [RUN ] %s\n", test.name);
        if (test.fn()) {
            std::printf("  [PASS] %s\n", test.name);
        } else {
            std::printf("  [FAIL] %s\n", test.name);
            ++failed;
        }
    }

    if (failed != 0) {
        std::printf("%d/%zu tests failed.\n", failed,
                    sizeof(tests) / sizeof(tests[0]));
        return 1;
    }

    std::printf("%zu/%zu tests passed.\n",
                sizeof(tests) / sizeof(tests[0]),
                sizeof(tests) / sizeof(tests[0]));
    return 0;
}
