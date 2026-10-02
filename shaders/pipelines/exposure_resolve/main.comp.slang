// Single-invocation exposure controller: percentile-trimmed mean of log2 luminance -> key -> clamp ->
// exponential smoothing in EV space. ExposureState keeps temporal state and read-only diagnostics
// (evScene/evTarget/evApplied/clipLowFrac/clipHighFrac) on the GPU. The CPU advances resetSeq when
// scene continuity breaks; a mismatch snaps to this frame's target without a host write or fence.

import display_common;
import bindings;

[[vk::push_constant]] ExposureResolvePush pc;

static const float LOG_MIN = -12.0;
static const float LOG_MAX = 12.0;
static const float LOG_STEP = (LOG_MAX - LOG_MIN) / 256.0;

float binCenter(uint bin) {
    return LOG_MIN + (float(bin) + 0.5) * LOG_STEP;
}

float curveSegment(float x, float x0, float y0, float x1, float y1) {
    return lerp(y0, y1, (x - x0) / (x1 - x0));
}

float curveCompensation(float sceneEv) {
    if (sceneEv <= pc.curveScene0) return pc.curveCompensation0;
    if (sceneEv < pc.curveScene1) {
        return curveSegment(sceneEv, pc.curveScene0, pc.curveCompensation0,
                pc.curveScene1, pc.curveCompensation1);
    }
    if (sceneEv < pc.curveScene2) {
        return curveSegment(sceneEv, pc.curveScene1, pc.curveCompensation1,
                pc.curveScene2, pc.curveCompensation2);
    }
    if (sceneEv < pc.curveScene3) {
        return curveSegment(sceneEv, pc.curveScene2, pc.curveCompensation2,
                pc.curveScene3, pc.curveCompensation3);
    }
    return pc.curveCompensation3;
}

float curveEffectiveSlope(float sceneEv) {
    if (sceneEv <= pc.curveScene0 || sceneEv >= pc.curveScene3) return 1.0;
    float compensationSlope;
    if (sceneEv < pc.curveScene1) {
        compensationSlope = (pc.curveCompensation1 - pc.curveCompensation0)
                / (pc.curveScene1 - pc.curveScene0);
    } else if (sceneEv < pc.curveScene2) {
        compensationSlope = (pc.curveCompensation2 - pc.curveCompensation1)
                / (pc.curveScene2 - pc.curveScene1);
    } else {
        compensationSlope = (pc.curveCompensation3 - pc.curveCompensation2)
                / (pc.curveScene3 - pc.curveScene2);
    }
    return 1.0 - compensationSlope;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    float surfacePopulation = 0.0;
    float skyPopulation = 0.0;
    float emissivePopulation = 0.0;
    for (uint i = 0u; i < 256u; ++i) {
        surfacePopulation += float(histBins[i]);
        skyPopulation += float(histBins[256u + i]);
        emissivePopulation += float(histBins[512u + i]);
    }
    float skyCap = clamp(pc.skyWeightCap, 0.0, 1.0);
    float emissiveCap = clamp(pc.emissiveWeightCap, 0.0, 1.0);
    float skyScale = 1.0;
    float emissiveScale = 1.0;
    // Preserve the previous behavior for views containing no ordinary surface: with no neutral
    // population to meter against, suppressing a sky-only or lava-only frame would leave no signal.
    // Otherwise solve the two final-share constraints exactly. If both raw populations exceed their
    // caps, both constraints are active and ordinary surfaces occupy the remaining share.
    if (surfacePopulation > 0.0) {
        float rawTotal = surfacePopulation + skyPopulation + emissivePopulation;
        bool capSky = skyPopulation > skyCap * rawTotal;
        bool capEmissive = emissivePopulation > emissiveCap * rawTotal;
        float weightedSky = skyPopulation;
        float weightedEmissive = emissivePopulation;
        if (capSky && capEmissive) {
            float ordinaryShare = max(1.0 - skyCap - emissiveCap, 1.0e-6);
            float cappedTotal = surfacePopulation / ordinaryShare;
            weightedSky = skyCap * cappedTotal;
            weightedEmissive = emissiveCap * cappedTotal;
        } else if (capSky) {
            weightedSky = skyCap <= 0.0 ? 0.0
                    : skyCap * (surfacePopulation + emissivePopulation) / max(1.0 - skyCap, 1.0e-6);
        } else if (capEmissive) {
            weightedEmissive = emissiveCap <= 0.0 ? 0.0
                    : emissiveCap * (surfacePopulation + skyPopulation) / max(1.0 - emissiveCap, 1.0e-6);
        }
        skyScale = skyPopulation > 0.0 ? min(1.0, weightedSky / skyPopulation) : 1.0;
        emissiveScale = emissivePopulation > 0.0
                ? min(1.0, weightedEmissive / emissivePopulation) : 1.0;
    }
    float population = surfacePopulation + skyPopulation * skyScale
            + emissivePopulation * emissiveScale;
    float total = max(population, 1.0);
    float lowPercentile = clamp(min(pc.lowPercentile, pc.highPercentile), 0.0, 1.0);
    float highPercentile = clamp(max(pc.lowPercentile, pc.highPercentile), 0.0, 1.0);
    float lowCount = min(floor(total * lowPercentile), total - 1.0);
    float highCount = clamp(ceil(total * highPercentile), lowCount + 1.0, total);

    float cumulative = 0.0;
    float weightedLogLum = 0.0;
    float weightedCount = 0.0;
    for (uint i = 0u; i < 256u; ++i) {
        float count = float(histBins[i]) + float(histBins[256u + i]) * skyScale
                + float(histBins[512u + i]) * emissiveScale;
        float start = cumulative;
        float end = cumulative + count;
        float takeStart = max(start, lowCount);
        float takeEnd = min(end, highCount);
        if (takeEnd > takeStart) {
            float take = takeEnd - takeStart;
            weightedLogLum += binCenter(i) * take;
            weightedCount += take;
        }
        cumulative = end;
    }

    float avgLogLum = weightedCount > 0.0 ? weightedLogLum / weightedCount : 0.0;
    // avgLogLum is log2 of the PRE-EXPOSED stored luminance. evScene is the absolute scene value on
    // the EV100 scale used by the compensation curve's control points and diagnostics.
    float evScene = avgLogLum + pc.evOffset;
    float compensation = curveCompensation(evScene);
    float effectiveSlope = curveEffectiveSlope(evScene);
    // Residual: what the display pass must still multiply the stored (already pre-exposed) value by.
    // Deriving it from avgLogLum rather than evScene is deliberate and exact -- stored * residual =
    // key * 2^(evBias + compensation) regardless of preExposure, so the pre-exposure cancels.
    float residualTarget = pc.key * exp2(pc.evBias - avgLogLum + compensation);
    // Clamp and smooth in ABSOLUTE exposure space. The residual is near 1.0 by construction (it is
    // whatever preExposure failed to anticipate), so clamping or smoothing it would bound the wrong
    // quantity and would fight preExposure drifting between frames.
    // Fail safe, not to white: a non-positive/non-finite value here can only mean the push layout
    // desynced, and clamping it to a tiny epsilon would divide the residual up to infinity and blow
    // the whole frame out. 1.0 safely disables pre-exposure for this frame.
    float preExposure = (isfinite(pc.preExposure) && pc.preExposure > 0.0) ? pc.preExposure : 1.0;
    float target = clamp(residualTarget * preExposure, exp2(pc.minEv), exp2(pc.maxEv));

    bool initialized = stateBuf[0].initialized != 0u;
    bool reset = !initialized || stateBuf[0].resetSeq != pc.resetSeq;
    float targetEv = log2(max(target, 1.0e-12));
    if (reset) {
        stateBuf[0].resetSeq = pc.resetSeq;
    }
    float prevEv = reset ? targetEv : log2(max(stateBuf[0].previous, 1.0e-12));

    // Smooth in EV space, not in linear exposure space. Linear smoothing is
    // wildly asymmetric for reasons that have nothing to do with the time constants: lerping toward a
    // much LARGER target crosses most of the ratio in the first frame, while lerping toward a much
    // SMALLER one decays through it geometrically. Measured in game, that made day->night snap in a
    // frame and night->day take ~3.5s of white screen -- the exact opposite of how eyes work, and it
    // survived any choice of adaptUp/adaptDown because the asymmetry was in the interpolation itself.
    // In EV space a 15 EV swing takes the same 3.4*tau either way, so the time constants below are the
    // only thing that sets the asymmetry, and they can express the real one: light adaptation is quick,
    // dark adaptation is slow.
    float exposureEv = prevEv;
    if (!reset) {
        float deltaEv = targetEv - prevEv;
        float rate = deltaEv > 0.0 ? pc.adaptDarken : pc.adaptBrighten;
        float alpha = 1.0 - exp(-pc.frameTimeSeconds / max(rate, 1.0e-4));
        exposureEv = prevEv + deltaEv * clamp(alpha, 0.0, 1.0);
    }
    float exposure = exp2(exposureEv);

    // previous stays ABSOLUTE: it is both the smoothing history and the value RtExposure reads back
    // to choose next frame's preExposure.
    stateBuf[0].previous = exposure;
    stateBuf[0].initialized = 1u;
    stateBuf[0].evScene = evScene;
    stateBuf[0].evTarget = targetEv;
    stateBuf[0].evApplied = exposureEv;
    stateBuf[0].clipLowFrac = (float(histBins[0]) + float(histBins[256]) * skyScale
            + float(histBins[512]) * emissiveScale) / total;
    stateBuf[0].clipHighFrac = (float(histBins[255]) + float(histBins[511]) * skyScale
            + float(histBins[767]) * emissiveScale) / total;
    stateBuf[0].meteringSkyScale = skyScale;
    stateBuf[0].meteringSkyFrac = skyPopulation * skyScale / total;
    stateBuf[0].curveCompensation = compensation;
    stateBuf[0].effectiveSlope = effectiveSlope;
    stateBuf[0].meteringEmissiveScale = emissiveScale;
    stateBuf[0].meteringEmissiveFrac = emissivePopulation * emissiveScale / total;
    // Autofocus: the histogram pass accumulated a centre-weighted distance sum and its weight. A frame with
    // no metered surface (all sky, or nothing sampled) reports 0, which is held rather than read as a focal
    // plane at the lens.
    //
    // The measurement is smoothed in TIME, not space. A single frame's metered depth is a noisy, brittle
    // estimate: a nearer surface crossing the centre of the frame moves it by metres at once, and the blur
    // that follows reads as the lens snapping rather than racking. A half-life is the right shape for that
    // because it is frame-rate independent by construction -- the fraction covered per frame follows the
    // frame's own duration -- so the same setting feels identical at 30 and at 240 fps, which a fixed
    // per-frame blend factor would not.
    float focusWeight = focusScratch[1];
    float measuredFocus = focusWeight > 0.0 ? focusScratch[0] / focusWeight : 0.0;
    // The auto-focus filter, applied to the centre sample: an exponential approach with a HALF-LIFE in seconds.
    // Expressed as exp(log(0.5) * frameTime * 10 / centerDepthHalflife) rather than as a per-frame blend factor,
    // so a stalled or doubled frame changes the time constant instead of the result.
    //
    // The history is the smoothed value itself, not the raw measurement, so this is an approach to whatever the
    // centre is showing now rather than an average of everything it has shown. Before the first measurement --
    // or after a reset -- it takes the measurement as it stands instead of racking the lens in from zero.
    stateBuf[0].measuredFocus = measuredFocus;
    float previousFocus = stateBuf[0].smoothCenterDepth;
    if (measuredFocus > 0.0) {
        float halflife = max(pc.focusHalflifeSeconds, 1.0e-4);
        float fadeFactor = exp(log(0.5) * max(pc.frameTimeSeconds, 0.0) * 10.0 / halflife);
        bool first = reset || !(previousFocus > 0.0);
        stateBuf[0].smoothCenterDepth = first
                ? measuredFocus
                : lerp(measuredFocus, previousFocus, clamp(fadeFactor, 0.0, 1.0));
    }
    // The reference's minFocalLength: the focal plane is never allowed inside the lens.
    stateBuf[0].focusDistance = stateBuf[0].smoothCenterDepth > 0.0
            ? max(stateBuf[0].smoothCenterDepth, 0.25) : 0.0;
    // The display pass consumes stored radiance, which raygen already scaled by preExposure, so it
    // needs the residual rather than the absolute exposure. stored * residual == L * exposure. The
    // camera's own aperture and shutter time scale that same number: the lens is part of the exposure,
    // not a second correction layered on top of it. Manual exposure overrides the resolved value here
    // rather than skipping this pass, so the metering the lens programmes read stays live in that mode.
    float absoluteExposure = pc.manualScale > 0.0 ? pc.manualScale : exposure;
    exposureImage[int2(0, 0)] = absoluteExposure * pc.physicalScale / preExposure;
}
