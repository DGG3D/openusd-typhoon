//
// Copyright 2026 Pixar
//
// Licensed under the terms set forth in the LICENSE.txt file available at
// https://openusd.org/license.
//
// Modified by DGG3D 2026: new file; remaining-time estimate for the current
// frame under checkpointed adaptive stopping (reported as the
// "estimatedSecondsRemaining" render stat). Bookkeeping only: nothing here
// influences which pixels are sampled or what is written to any AOV.
//
// Remaining-time model.
//
// A full-resolution pass renders one sample for every still-active pixel, so
// its wall-clock cost is modelled as
//
//     passSeconds ~= a + b * activePixels,   a, b >= 0,
//
// where a covers the per-pass fixed work (retirement sweep, convergence scan,
// throttled resolve, scheduling) and b the per-pixel path tracing. a and b are
// a least-squares fit over the timed passes of the frame.
//
// Active pixels only change at adaptive checkpoints (minSamples * 2^k, see
// IsAdaptiveCheckpoint), so the future is predicted checkpoint by checkpoint:
// the current active count holds until the next checkpoint, after which it is
// multiplied by the survival ratio r (active after / active before the sweep)
// observed at the most recent checkpoint. Before any checkpoint has been
// observed r = 1 (no further retirement): this is the conservative choice, an
// upper bound on the remaining time given b >= 0, so the estimate only falls
// as retirement actually happens instead of promising an early finish that a
// scene with few easy pixels would never deliver. The frame ends at
// samplesToConvergence passes, or earlier once the predicted active count
// rounds to zero.

#ifndef PXR_IMAGING_PLUGIN_HD_EMBREE_ADAPTIVE_TIME_ESTIMATE_H
#define PXR_IMAGING_PLUGIN_HD_EMBREE_ADAPTIVE_TIME_ESTIMATE_H

#include <renderer/aov/adaptiveConvergence.h>

#include "pxr/pxr.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

PXR_NAMESPACE_OPEN_SCOPE
namespace ty {

/// Pass-time model passSeconds ~= fixedSeconds + secondsPerPixel * active.
struct AdaptivePassTimeModel {
    double fixedSeconds = 0.0;
    double secondsPerPixel = 0.0;
};

/// Timing summary of passes that all rendered the same number of active
/// pixels.
struct AdaptivePassTimeLevel {
    double activePixels = 0.0;
    uint64_t passes = 0;
    double totalSeconds = 0.0;
};

/// Least-squares fit of the pass-time model, clamped to a, b >= 0.
///
/// Each level contributes its passes with weight \c passes; because passes
/// of one level share the same active count, this is exactly ordinary least
/// squares over the individual passes. When the active count has not varied
/// the slope is unidentifiable and the model falls back to the mean pass
/// time (b = 0). A negative slope is clamped to the mean pass time; a
/// negative intercept is clamped to a fit through the origin.
inline AdaptivePassTimeModel
FitAdaptivePassTime(AdaptivePassTimeLevel const* levels, size_t numLevels)
{
    AdaptivePassTimeModel model;
    double n = 0.0;
    double sumX = 0.0;
    double sumT = 0.0;
    for (size_t i = 0; i < numLevels; ++i) {
        const double w = static_cast<double>(levels[i].passes);
        n += w;
        sumX += w * levels[i].activePixels;
        sumT += levels[i].totalSeconds;
    }
    if (n <= 0.0) {
        return model;
    }
    const double meanX = sumX / n;
    const double meanT = std::max(0.0, sumT / n);

    double sxx = 0.0;     // sum over passes of (x - meanX)^2
    double sxt = 0.0;     // sum over passes of (x - meanX) * (t - meanT)
    double sxxRaw = 0.0;  // sum over passes of x^2
    double sxtRaw = 0.0;  // sum over passes of x * t
    for (size_t i = 0; i < numLevels; ++i) {
        const double w = static_cast<double>(levels[i].passes);
        const double x = levels[i].activePixels;
        const double dx = x - meanX;
        sxx += w * dx * dx;
        sxt += dx * (levels[i].totalSeconds - w * meanT);
        sxxRaw += w * x * x;
        sxtRaw += x * levels[i].totalSeconds;
    }

    // Relative test: the active count spans many orders of magnitude.
    if (!(sxx > 1e-12 * sxxRaw)) {
        model.fixedSeconds = meanT;
        return model;
    }

    const double slope = sxt / sxx;
    const double intercept = meanT - slope * meanX;
    if (slope < 0.0) {
        model.fixedSeconds = meanT;
    } else if (intercept < 0.0) {
        model.secondsPerPixel = std::max(0.0, sxtRaw / sxxRaw);
    } else {
        model.fixedSeconds = intercept;
        model.secondsPerPixel = slope;
    }
    return model;
}

/// Smallest adaptive checkpoint count (see IsAdaptiveCheckpoint) strictly
/// greater than \p count. A non-positive \p minSamples is treated as 1.
inline uint64_t
NextAdaptiveCheckpoint(uint64_t count, int minSamples)
{
    uint64_t checkpoint =
        static_cast<uint64_t>(std::max(1, minSamples));
    while (checkpoint <= count) {
        checkpoint *= 2;
    }
    return checkpoint;
}

/// Predicted active count below which every pixel is taken to be retired.
constexpr double kAdaptiveEtaMinActivePixels = 0.5;

/// Predicted wall-clock seconds for the passes that remain in the frame.
///
/// \param model Pass-time model.
/// \param completedPasses Full-resolution passes completed so far (s). Every
/// active pixel holds s samples; if s is a checkpoint its retirement is
/// already reflected in \p activePixels.
/// \param activePixels Pixels the next pass will sample.
/// \param samplesToConvergence Pass cap N of the frame.
/// \param minSamples Adaptive checkpoint base.
/// \param survivalRatio Fraction of active pixels predicted to survive each
/// future checkpoint's retirement sweep; clamped to [0, 1].
/// \return Sum over passes s+1..N of a + b * predictedActive, where
/// predictedActive is multiplied by \p survivalRatio after every checkpoint
/// pass before N and the sum stops once it drops below
/// kAdaptiveEtaMinActivePixels. 0 when s >= N or nothing is active.
/// O(number of checkpoints).
inline double
PredictAdaptiveRemainingSeconds(
    AdaptivePassTimeModel const& model,
    int completedPasses,
    double activePixels,
    int samplesToConvergence,
    int minSamples,
    double survivalRatio)
{
    if (samplesToConvergence <= 0 ||
        completedPasses >= samplesToConvergence) {
        return 0.0;
    }
    const double ratio = std::min(1.0, std::max(0.0, survivalRatio));
    const uint64_t endPass = static_cast<uint64_t>(samplesToConvergence);
    uint64_t done = static_cast<uint64_t>(std::max(0, completedPasses));
    double active = std::max(0.0, activePixels);
    double seconds = 0.0;
    while (done < endPass && active >= kAdaptiveEtaMinActivePixels) {
        // Passes done+1 .. segmentEnd all render the current active count.
        const uint64_t segmentEnd =
            std::min(NextAdaptiveCheckpoint(done, minSamples), endPass);
        seconds += static_cast<double>(segmentEnd - done) *
            (model.fixedSeconds + model.secondsPerPixel * active);
        done = segmentEnd;
        // segmentEnd < endPass is a checkpoint: its sweep retires pixels
        // before the next pass.
        if (done < endPass) {
            active *= ratio;
        }
    }
    return seconds;
}

/// Per-frame accumulator for the remaining-time estimate. Not thread-safe;
/// the owner serializes access.
///
/// Timings are kept as a ring of the most recent kMaxLevels "levels", runs
/// of consecutive passes at the same active count, rather than of individual
/// passes: the active count only changes at checkpoints, which are at least
/// minSamples passes apart, so a ring of recent individual passes would
/// almost never see two active counts and the fit would degenerate to the
/// mean pass time, ignoring predicted retirement.
class AdaptiveTimeEstimator {
public:
    static constexpr size_t kMaxLevels = 32;
    /// Timed passes required before an estimate is reported.
    static constexpr uint64_t kMinTimedPasses = 3;

    /// Forgets all timings and the observed survival ratio.
    void Reset()
    {
        _numLevels = 0;
        _nextLevel = 0;
        _lastLevel = kMaxLevels;
        _timedPasses = 0;
        _survivalRatio = 1.0;
    }

    /// Records one pass that sampled \p activePixels pixels in \p seconds.
    /// Negative or non-finite durations are ignored.
    void RecordPass(uint64_t activePixels, double seconds)
    {
        if (!std::isfinite(seconds) || seconds < 0.0) {
            return;
        }
        const double x = static_cast<double>(activePixels);
        if (_lastLevel < kMaxLevels &&
            _levels[_lastLevel].activePixels == x) {
            _levels[_lastLevel].passes += 1;
            _levels[_lastLevel].totalSeconds += seconds;
        } else {
            // Overwrites the oldest level once the ring is full. The fit is
            // order-independent, so entries [0, _numLevels) are always the
            // live set.
            _lastLevel = _nextLevel;
            _nextLevel = (_nextLevel + 1) % kMaxLevels;
            if (_numLevels < kMaxLevels) {
                ++_numLevels;
            }
            AdaptivePassTimeLevel& level = _levels[_lastLevel];
            level.activePixels = x;
            level.passes = 1;
            level.totalSeconds = seconds;
        }
        ++_timedPasses;
    }

    /// Records the retirement at a checkpoint: \p activeBefore pixels were
    /// sampled by the checkpoint pass and \p activeAfter remain after its
    /// sweep. The most recent observation replaces earlier ones.
    void RecordCheckpointSurvival(uint64_t activeBefore, uint64_t activeAfter)
    {
        if (activeBefore == 0) {
            return;
        }
        _survivalRatio = std::min(1.0,
            static_cast<double>(activeAfter) /
            static_cast<double>(activeBefore));
    }

    uint64_t GetTimedPassCount() const { return _timedPasses; }

    double GetSurvivalRatio() const { return _survivalRatio; }

    AdaptivePassTimeModel Fit() const
    {
        return FitAdaptivePassTime(_levels, _numLevels);
    }

    /// \return Predicted seconds until the frame completes (see
    /// PredictAdaptiveRemainingSeconds), or -1 (unknown) while fewer than
    /// kMinTimedPasses passes have been timed.
    double EstimateRemainingSeconds(
        int completedPasses, uint64_t activePixels,
        int samplesToConvergence, int minSamples) const
    {
        if (_timedPasses < kMinTimedPasses) {
            return -1.0;
        }
        return PredictAdaptiveRemainingSeconds(
            Fit(), completedPasses, static_cast<double>(activePixels),
            samplesToConvergence, minSamples, _survivalRatio);
    }

private:
    AdaptivePassTimeLevel _levels[kMaxLevels];
    size_t _numLevels = 0;
    size_t _nextLevel = 0;
    // Level receiving passes at the current active count; kMaxLevels = none.
    size_t _lastLevel = kMaxLevels;
    uint64_t _timedPasses = 0;
    double _survivalRatio = 1.0;
};

} // namespace ty
PXR_NAMESPACE_CLOSE_SCOPE

#endif // PXR_IMAGING_PLUGIN_HD_EMBREE_ADAPTIVE_TIME_ESTIMATE_H
