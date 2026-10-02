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
//
// Modified by DGG3D 2026: work-prediction model, preferred once available.
// The fit above extrapolates one survival ratio and a fixed + linear pass
// cost, but retirement is not geometric (the last checkpoints retire far more
// than the middle ones) and the pixels that survive longest are the
// expensive ones, so per-pixel cost rises as the active count falls. Instead:
//
//  * Every active pixel's retirement level is predicted from its own
//    statistics (PredictAdaptivePassingCount: the checkpoint test's
//    variance of the mean falls as 1 / n) and then raised to the maximum
//    over its active 3x3 neighbourhood, since a pixel only retires once all
//    in-window neighbours pass too. This treats passing as permanent: a pixel
//    predicted to pass at a checkpoint is assumed to keep passing at later
//    ones, so noise around the threshold is ignored (it makes real
//    retirement somewhat later than predicted).
//  * Each pixel is costed by its own measured tracing time per sample
//    (timed every kAdaptivePixelTimingInterval-th pass; its tile's average
//    until it has one), so a level is costed by the pixels predicted to
//    remain in it (AdaptiveWorkPrediction::traceCpuSeconds). A tile average
//    is not enough: the pixels that survive longest are cheaper per sample
//    than the tile mates that retired before them.
//  * A pass's tile loop is one WorkParallelForN over all tiles in tile-index
//    order, a barrier, so its wall time is the makespan of its schedule, not
//    summed tile time / threads. With TBB's default auto_partitioner the
//    range is split into about one task per thread, each of which splits its
//    own range only to a depth of 5 (__TBB_INIT_DEPTH) unless work is
//    demanded, so contiguous runs of about numTiles / (threads * 32) tiles
//    run on one thread. When few tiles remain active they cluster in a few
//    such runs and parallelism collapses (measured: about one run of
//    active tiles per effective thread). So the tile loop of level j is
//    modelled as the greedy (longest-first) makespan of those runs' predicted
//    tracing time over the worker threads
//    (AdaptiveWorkPrediction::traceWallSeconds, at least the largest run and
//    the summed time / threads).
//  * Wall time per pass of level j is overhead + scale * traceWallSeconds[j],
//    where overhead (sweep, scans, throttled resolve, the prediction itself)
//    and scale = measured tile-loop wall time / predicted makespan of the
//    same passes are measured over the current level's passes. Until the
//    scale is measured, w_j * traceCpuSeconds[j] is used instead, with
//    w = tile-loop wall time / summed tile time and
//    w_j = max(w, 1 / activeTiles[j]).

#ifndef PXR_IMAGING_PLUGIN_HD_EMBREE_ADAPTIVE_TIME_ESTIMATE_H
#define PXR_IMAGING_PLUGIN_HD_EMBREE_ADAPTIVE_TIME_ESTIMATE_H

#include <renderer/aov/adaptiveConvergence.h>

#include "pxr/pxr.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <queue>
#include <vector>

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

/// Upper bound on the adaptive levels of a frame: level j ends at
/// min(minSamples * 2^j, samplesToConvergence), which fits in an int.
constexpr int kMaxAdaptiveLevels = 32;

/// Samples per pixel before the first work prediction: fewer make the
/// per-pixel variance estimate too noisy to predict from.
constexpr int kAdaptivePredictionMinPasses = 16;

/// Every kAdaptivePixelTimingInterval-th full-resolution pass times each
/// pixel sample, so every pixel is costed by its own tracing time: pixels
/// that survive to later levels are measurably cheaper per sample than the
/// tile mates that retired before them, so a tile average over-costs them,
/// increasingly so with depth. Two clock reads per sample on one pass in
/// this many.
constexpr int kAdaptivePixelTimingInterval = 8;

/// Fills \p outLevelEndPass with the last pass of every adaptive level of a
/// frame. Level j covers passes (end(j - 1), end(j)], end(-1) = 0; its end is
/// the checkpoint minSamples * 2^j, and the last level ends at
/// \p samplesToConvergence. \p outLevelEndPass must hold kMaxAdaptiveLevels
/// entries. A non-positive \p minSamples is treated as 1.
/// \return Number of levels; 0 when \p samplesToConvergence <= 0.
inline int
BuildAdaptiveLevels(
    int minSamples, int samplesToConvergence, uint64_t* outLevelEndPass)
{
    if (samplesToConvergence <= 0) {
        return 0;
    }
    const uint64_t endPass = static_cast<uint64_t>(samplesToConvergence);
    uint64_t checkpoint = static_cast<uint64_t>(std::max(1, minSamples));
    int numLevels = 0;
    while (numLevels < kMaxAdaptiveLevels) {
        outLevelEndPass[numLevels++] = std::min(checkpoint, endPass);
        if (checkpoint >= endPass) {
            break;
        }
        checkpoint *= 2;
    }
    return numLevels;
}

/// Level containing 1-based \p pass: the first level whose end is >= pass,
/// clamped to the last level.
inline int
AdaptiveLevelOfPass(
    uint64_t pass, uint64_t const* levelEndPass, int numLevels)
{
    int level = 0;
    while (level + 1 < numLevels && levelEndPass[level] < pass) {
        ++level;
    }
    return level;
}

/// Predicted work left in the current frame (see the model at the top of
/// this file). Levels before currentLevel hold zeros.
struct AdaptiveWorkPrediction {
    /// Passes completed when the prediction was made; 0 = no prediction.
    int completedPasses = 0;
    /// Level of pass completedPasses + 1.
    int currentLevel = 0;
    int numLevels = 0;
    /// Worker threads and tiles per contiguous scheduling run assumed by
    /// traceWallSeconds.
    int threads = 0;
    int runTiles = 0;
    uint64_t levelEndPass[kMaxAdaptiveLevels] = {};
    /// Pixels predicted to be sampled by every pass of level j.
    uint64_t activePixels[kMaxAdaptiveLevels] = {};
    /// Tiles holding at least one of those pixels.
    uint64_t activeTiles[kMaxAdaptiveLevels] = {};
    /// Predicted tracing time of one pass of level j, summed over its tiles
    /// (thread time, not wall time).
    double traceCpuSeconds[kMaxAdaptiveLevels] = {};
    /// Predicted tile-loop makespan of one pass of level j
    /// (AdaptiveScheduleMakespan of its runs' tracing time).
    double traceWallSeconds[kMaxAdaptiveLevels] = {};
    /// Predicted tracing time of the most expensive run of level j.
    double maxRunSeconds[kMaxAdaptiveLevels] = {};
    /// Sum over active pixels of (predicted retirement pass - completed
    /// passes): the pixel samples still to be traced.
    uint64_t remainingPixelSamples = 0;
};

/// Wall-clock cost of the timed passes of one adaptive level.
struct AdaptivePassCost {
    uint64_t passes = 0;
    /// Whole passes, including retirement sweep, scans and resolves.
    double passSeconds = 0.0;
    /// The tile loops of those passes.
    double traceSeconds = 0.0;
    /// Summed per-tile tracing time of those tile loops.
    double traceCpuSeconds = 0.0;
    /// For the passes whose makespan was predicted: the predicted makespans
    /// and the measured tile-loop times.
    double predictedTraceSeconds = 0.0;
    double predictedTraceMeasuredSeconds = 0.0;
};

/// Measured per-pass costs applied to a work prediction. Negative = not
/// measured.
struct AdaptivePassCostRates {
    /// Per-pass wall time outside the tile loop.
    double overheadSeconds = -1.0;
    /// Tile-loop wall time per second of summed tile time.
    double traceWallPerCpuSecond = -1.0;
    /// Tile-loop wall time per second of predicted makespan.
    double traceWallScale = -1.0;
};

/// Greedy longest-first makespan of independent \p jobSeconds over
/// \p threads workers: the wall time of a barrier-terminated parallel loop
/// whose indivisible pieces cost \p jobSeconds. Reorders \p jobSeconds.
/// \return At least max(job) and sum / threads; 0 for no jobs.
inline double
AdaptiveScheduleMakespan(std::vector<double>* jobSeconds, int threads)
{
    if (jobSeconds->empty()) {
        return 0.0;
    }
    std::sort(jobSeconds->begin(), jobSeconds->end(), std::greater<double>());
    std::priority_queue<double, std::vector<double>, std::greater<double>>
        threadLoads;
    for (int i = 0; i < std::max(1, threads); ++i) {
        threadLoads.push(0.0);
    }
    double makespan = 0.0;
    for (double job : *jobSeconds) {
        const double load = threadLoads.top() + job;
        threadLoads.pop();
        threadLoads.push(load);
        makespan = std::max(makespan, load);
    }
    return makespan;
}

/// Predicted wall-clock seconds of one pass of \p level (see the model at the
/// top of this file): overhead + scale * traceWallSeconds once the scale is
/// measured, else overhead + max(w, 1 / activeTiles) * traceCpuSeconds.
/// \return Negative when the needed rates are not measured.
inline double
PredictAdaptivePassSeconds(
    AdaptiveWorkPrediction const& work, int level,
    AdaptivePassCostRates const& rates)
{
    if (level < 0 || level >= work.numLevels || rates.overheadSeconds < 0.0) {
        return -1.0;
    }
    if (rates.traceWallScale > 0.0 && work.traceWallSeconds[level] > 0.0) {
        return rates.overheadSeconds +
            rates.traceWallScale * work.traceWallSeconds[level];
    }
    if (rates.traceWallPerCpuSecond < 0.0) {
        return -1.0;
    }
    const double tiles = static_cast<double>(
        std::max<uint64_t>(1, work.activeTiles[level]));
    return rates.overheadSeconds +
        std::max(rates.traceWallPerCpuSecond, 1.0 / tiles) *
        work.traceCpuSeconds[level];
}

/// Predicted wall-clock seconds for the passes that remain in the frame from
/// a work prediction (see the model at the top of this file).
///
/// \param work Prediction made at or before \p completedPasses.
/// \param completedPasses Full-resolution passes completed so far.
/// \param rates Measured pass costs (see PredictAdaptivePassSeconds).
/// \return Sum over the remaining passes of PredictAdaptivePassSeconds,
/// stopping at the first level predicted to have no active pixels. 0 when
/// no level remains; negative when the rates are not measured.
inline double
PredictAdaptiveRemainingSecondsFromWork(
    AdaptiveWorkPrediction const& work,
    int completedPasses,
    AdaptivePassCostRates const& rates)
{
    if (work.numLevels <= 0) {
        return 0.0;
    }
    const uint64_t done = static_cast<uint64_t>(std::max(0, completedPasses));
    if (done >= work.levelEndPass[work.numLevels - 1]) {
        return 0.0;
    }
    double seconds = 0.0;
    uint64_t passStart = done;
    const int firstLevel = std::max(work.currentLevel,
        AdaptiveLevelOfPass(done + 1, work.levelEndPass, work.numLevels));
    for (int j = firstLevel; j < work.numLevels; ++j) {
        if (work.activePixels[j] == 0) {
            break;
        }
        const double passSeconds = PredictAdaptivePassSeconds(work, j, rates);
        if (passSeconds < 0.0) {
            return -1.0;
        }
        const uint64_t passEnd = work.levelEndPass[j];
        if (passEnd > passStart) {
            seconds += static_cast<double>(passEnd - passStart) * passSeconds;
            passStart = passEnd;
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

    /// Timed passes of the current level before its pass cost replaces the
    /// previous level's.
    static constexpr uint64_t kMinLevelCostPasses = 4;
    /// Bounds of the measured makespan scale: it corrects scheduling detail
    /// the makespan model leaves out, not a different cost model.
    static constexpr double kMinTraceWallScale = 0.5;
    static constexpr double kMaxTraceWallScale = 4.0;

    /// Forgets all timings, the observed survival ratio and the work
    /// prediction.
    void Reset()
    {
        _numLevels = 0;
        _nextLevel = 0;
        _lastLevel = kMaxLevels;
        _timedPasses = 0;
        _survivalRatio = 1.0;
        _work = AdaptiveWorkPrediction();
        _levelCost = AdaptivePassCost();
        _previousLevelCost = AdaptivePassCost();
        _costLevelEndPass = 0;
    }

    /// Records the cost breakdown of timed pass \p completedPasses of the
    /// level ending at pass \p levelEndPass (see AdaptivePassCost), and the
    /// work prediction's makespan for it when the prediction covers that
    /// pass's level. Starting a new level keeps the last one as the fallback
    /// until the new one has kMinLevelCostPasses passes. Non-finite or
    /// negative times are ignored.
    void RecordPassCost(
        uint64_t levelEndPass, int completedPasses, double passSeconds,
        double traceSeconds, double traceCpuSeconds)
    {
        if (!std::isfinite(passSeconds) || !std::isfinite(traceSeconds) ||
            !std::isfinite(traceCpuSeconds) || passSeconds < 0.0 ||
            traceSeconds < 0.0 || traceCpuSeconds < 0.0) {
            return;
        }
        if (levelEndPass != _costLevelEndPass) {
            if (_levelCost.passes > 0) {
                _previousLevelCost = _levelCost;
            }
            _levelCost = AdaptivePassCost();
            _costLevelEndPass = levelEndPass;
        }
        _levelCost.passes += 1;
        _levelCost.passSeconds += passSeconds;
        _levelCost.traceSeconds += traceSeconds;
        _levelCost.traceCpuSeconds += traceCpuSeconds;
        if (_work.completedPasses > 0 && completedPasses > 0) {
            const int level = AdaptiveLevelOfPass(
                static_cast<uint64_t>(completedPasses), _work.levelEndPass,
                _work.numLevels);
            if (level >= _work.currentLevel &&
                _work.traceWallSeconds[level] > 0.0) {
                _levelCost.predictedTraceSeconds +=
                    _work.traceWallSeconds[level];
                _levelCost.predictedTraceMeasuredSeconds += traceSeconds;
            }
        }
    }

    /// Measured pass costs (see AdaptivePassCostRates), from the current
    /// level's passes (the previous level's while the current one has fewer
    /// than kMinLevelCostPasses). The makespan scale is clamped to
    /// [kMinTraceWallScale, kMaxTraceWallScale] and stays unmeasured until
    /// some timed pass had a predicted makespan.
    /// \return false while no timed pass traced anything.
    bool GetMeasuredPassCost(AdaptivePassCostRates* outRates) const
    {
        *outRates = AdaptivePassCostRates();
        AdaptivePassCost const& cost =
            (_levelCost.passes >= kMinLevelCostPasses ||
             _previousLevelCost.passes == 0)
            ? _levelCost : _previousLevelCost;
        if (cost.passes == 0 || !(cost.traceCpuSeconds > 0.0)) {
            return false;
        }
        outRates->overheadSeconds = std::max(0.0,
            (cost.passSeconds - cost.traceSeconds) /
            static_cast<double>(cost.passes));
        outRates->traceWallPerCpuSecond =
            cost.traceSeconds / cost.traceCpuSeconds;
        if (cost.predictedTraceSeconds > 0.0) {
            outRates->traceWallScale = std::min(kMaxTraceWallScale,
                std::max(kMinTraceWallScale,
                    cost.predictedTraceMeasuredSeconds /
                    cost.predictedTraceSeconds));
        }
        return true;
    }

    /// Replaces the work prediction (completedPasses 0 = none).
    void SetWorkPrediction(AdaptiveWorkPrediction const& work)
    {
        _work = work;
    }

    AdaptiveWorkPrediction const& GetWorkPrediction() const { return _work; }

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

    /// \return Predicted seconds until the frame completes, or -1 (unknown)
    /// while fewer than kMinTimedPasses passes have been timed. Uses the work
    /// prediction (PredictAdaptiveRemainingSecondsFromWork) once one exists
    /// and a pass cost has been measured, else the pass-time fit
    /// (PredictAdaptiveRemainingSeconds).
    double EstimateRemainingSeconds(
        int completedPasses, uint64_t activePixels,
        int samplesToConvergence, int minSamples) const
    {
        if (_timedPasses < kMinTimedPasses) {
            return -1.0;
        }
        AdaptivePassCostRates rates;
        if (_work.completedPasses > 0 && GetMeasuredPassCost(&rates)) {
            const double seconds = PredictAdaptiveRemainingSecondsFromWork(
                _work, completedPasses, rates);
            if (seconds >= 0.0) {
                return seconds;
            }
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
    AdaptiveWorkPrediction _work;
    // Pass costs of the level ending at _costLevelEndPass and of the level
    // timed before it.
    AdaptivePassCost _levelCost;
    AdaptivePassCost _previousLevelCost;
    uint64_t _costLevelEndPass = 0;
};

} // namespace ty
PXR_NAMESPACE_CLOSE_SCOPE

#endif // PXR_IMAGING_PLUGIN_HD_EMBREE_ADAPTIVE_TIME_ESTIMATE_H
