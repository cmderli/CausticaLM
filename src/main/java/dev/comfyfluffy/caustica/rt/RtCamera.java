package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.CausticaConfig;
import org.joml.Matrix4fc;

/**
 * The virtual camera's lens, solved once per frame.
 *
 * <p>Everything here follows from one physical model: the primary rays pass through a thin lens of focal
 * length {@code f} whose entrance pupil has diameter {@code f / N} for an f-number {@code N}, focused at
 * {@code Zf}. A point at distance {@code Z} then lands on the film as a disc of diameter
 * {@code (f / N) * |Zf / Z - 1|}, which is the circle of confusion this renderer blurs with — so the field
 * of view matters: it sets {@code f}, and a narrow FOV focuses tighter at the same aperture, exactly as a
 * telephoto lens does.
 *
 * <p>Exposure rides on the same two quantities. A frame's exposure value is
 * {@code EV100 = log2(N^2 / t)}, so the aperture and the shutter time together decide how bright the image
 * is; {@link #physicalExposureScale} turns that into the multiplier the display pass applies.
 */
public final class RtCamera {
    /**
     * Full-frame sensor height in metres. Only the focal length derived from the FOV depends on this, and the
     * FOV is what actually shapes the image: the sensor fixes the unit the lens is expressed in, which is why
     * a given FOV and f-number defocus by the same amount whatever the render resolution is.
     */
    public static final float SENSOR_HEIGHT_METRES = 0.024f;

    /** Focus distances at or beyond this are treated as infinity; 1 km is past every world border. */
    private static final float FOCUS_INFINITY_BLOCKS = 1024.0f;

    /**
     * Largest circle of confusion the lens may produce, as a fraction of the render height.
     *
     * <p>A fast lens focused close has an enormous geometric blur: a 45 mm f/1.4 focused at 1 m puts a subject
     * 3 m away through a disc 64 mm across on a 24 mm sensor, 267% of the frame. Everything from about a metre
     * out is then past any sane bound at the SAME value, which is what makes a frame read as uniformly hazy
     * instead of as a sharp subject against a soft background — the separation between in and out of focus is
     * the whole effect, and capping everything to one number removes it. A smaller cap keeps the near field
     * distinguishable: the blur still grows with distance, it just stops before it swallows the frame.
     *
     * <p>It is a fraction of the frame rather than a block size, so the look does not change with resolution or
     * field of view. {@code DEPTH_OF_FIELD_ENABLED} scales it, which is the artistic dial.
     */
    public static final float MAX_DEFOCUS_FRAME_FRACTION = 0.02f;
    /** The setting value {@link #MAX_DEFOCUS_FRAME_FRACTION} corresponds to, in percent. */
    public static final float DEFAULT_MAX_BLUR_PERCENT = 2.0f;

    /**
     * Time constant of the auto programme, in seconds. What the frame still needs on top of the exposure the
     * controller has already applied is {@code evTarget - evApplied}, and this is how fast the lens moves when
     * that correction changes. An aperture is a mechanical part, so it does not re-rack on every flicker of the
     * histogram.
     */
    private static final float AUTO_ADAPT_SECONDS = 0.75f;

    /**
     * Disc of confusion, in render pixels, that the primary rays' own aperture samples are allowed to cover.
     *
     * <p>One primary ray per pixel samples one point of the pupil, so the traced disc is only resolved while it
     * stays a few pixels wide. Past that a pixel's single sample stops reading as blur and becomes speckle, with
     * an in/out flicker along every silhouette that no amount of temporal history removes, because the geometry
     * the sample lands on keeps changing. Capping the traced disc here is what leaves the depth-of-field gather
     * the remainder of the aperture: below the cap the frame gets a physically traced near-field bokeh, above it
     * the gather supplies the rest, and the two together are always the aperture the camera was set to.
     *
     * <p>Three pixels, not six, because the samples have to cover the disc rather than merely fall inside it.
     * One sample per pixel at two aperture points puts its nearest neighbours about
     * {@code radius * sqrt(pi / 2)} pixels away, which is six at a three-pixel disc and eleven at a six-pixel
     * one -- and a gap that wide is a dotted ring along a silhouette, not a bokeh. The gather's 96 taps are what
     * make a wide disc possible, so the split belongs there, and this only keeps the part the trace is genuinely
     * better at: the near-field highlight shapes a blur cannot invent.
     */
    public static final float TRACED_DISC_PIXELS = 3.0f;

    /**
     * The circle of confusion a viewer still reads as sharp, in metres: 0.03 mm, the long-standing convention
     * for a 35 mm frame.
     *
     * <p>It is not the disc the renderer draws -- that is {@code DEPTH_OF_FIELD}'s own cap, and it is far larger,
     * because it says how much blur to put on the screen rather than how much the eye forgives. This one exists
     * to answer "is this sharp", which is the question the hyperfocal distance is defined by, and one block is
     * one metre so it is usable as a distance directly.
     */
    public static final float PHOTOGRAPHIC_COC_METRES = 0.03e-3f;

    /**
     * The disc the traced aperture samples cover, as a fraction of the full circle of confusion.
     *
     * <p>Expressed as a fraction rather than as a lens radius because the two are the same measurement: a lens
     * radius {@code a} projects a point at infinity to {@code a * renderHeight / sensorHeight} pixels, so
     * scaling the lens scales the disc linearly and the cap is exact at every focus distance and resolution.
     *
     * @param fullRadiusPixels the disc a point at infinity reaches through the whole aperture, in render pixels
     * @param tracedRadiusPixels the largest disc the trace may cover, in render pixels
     */
    public static float tracedDiscFraction(float fullRadiusPixels, float tracedRadiusPixels) {
        if (!(fullRadiusPixels > 0.0f) || !(tracedRadiusPixels > 0.0f)) {
            return 1.0f;
        }
        return Math.clamp(tracedRadiusPixels / fullRadiusPixels, 0.0f, 1.0f);
    }

    /**
     * Time for the auto-focus plane to cover half the distance to the metered depth, in seconds. Focusing is a
     * lens movement, not a per-frame assignment: the metered depth changes by metres the moment a nearer surface
     * crosses the centre of the frame, and a focal plane that jumped with it would read as the image snapping
     * rather than as a lens racking. Half a second is long enough to smooth one centre sample without lagging
     * behind a deliberate turn of the view.
     */
    public static final float FOCUS_HALFLIFE_SECONDS = 0.5f;

    /**
     * Fraction of the remaining distance to the metered depth the focal plane covers in {@code frameSeconds},
     * for a given half-life.
     *
     * <p>Expressed as a half-life rather than as a per-frame blend because that is what makes the motion the
     * same at every frame rate: the fraction follows the frame's own duration, so 30 fps and 240 fps rack the
     * lens over the same wall-clock time. It is also the exact discrete form of an exponential approach, so
     * repeated steps compose to the same curve as one long step.
     */
    public static float focusFadeFactor(float frameSeconds, float halflifeSeconds) {
        return (float) Math.pow(2.0, -Math.max(frameSeconds, 0.0f) / Math.max(halflifeSeconds, 1.0e-4f));
    }

    /** Shortest and longest frame interval the auto programme's smoothing will believe, in seconds. */
    private static final float MIN_FRAME_SECONDS = 1.0f / 1000.0f;
    private static final float MAX_FRAME_SECONDS = 0.25f;

    /**
     * Aperture the auto programme has already settled on, so a rung boundary cannot make it oscillate. Cleared
     * by {@link #reset()}.
     */
    private static float autoAperture;

    /**
     * EV100 assumed before anything has been metered. It is the scene brightness at which the reference pair
     * exposes at the metering key, so the lens contributes exactly 1.0 until real metering arrives.
     */
    private static final float AUTO_FALLBACK_EV100 = 2.0f * CausticaConfig.Rt.Camera.REFERENCE_EV100;


    /** Acceptance used when converting the guide's reversed-Z depth into a distance. */
    private static final float DEPTH_EPSILON = 1.0e-7f;

    /**
     * How much closer to the target exposure another aperture rung has to land before the auto programme gives up
     * the f-number the player set, in stops. Below this the improvement is smaller than the quantization the
     * shutter's rungs already impose, so moving the iris would buy accuracy the image cannot show while costing
     * the depth of field the player chose. This is what makes the programme aperture priority rather than a
     * per-frame optimizer.
     */
    private static final float AUTO_EXPOSURE_TOLERANCE_STOPS = 0.33f;

    /** The frame's blur bound, as a fraction of the height: the default scaled by the player's setting. */
    private static float maxDefocusFraction() {
        return MAX_DEFOCUS_FRAME_FRACTION
                * (CausticaConfig.Rt.Camera.MAX_BLUR_PERCENT.value() / DEFAULT_MAX_BLUR_PERCENT);
    }

    private RtCamera() {
    }

    /**
     * This frame's solved optics. The aperture and shutter time are ladder values — the same rungs a
     * photographer can actually set — and {@code physicalExposureScale} is what those rungs do to the image.
     */
    public record Frame(
            float aperture,
            float shutterSeconds,
            float focusDistanceBlocks,
            float sensorHeightMetres,
            float focalLengthMetres,
            float defocusScaleBlocks,
            float physicalExposureScale,
            int renderHeight
    ) {
        /**
         * The cap the shader applies to the circle of confusion, as a multiple of one pixel's own ray cone. The
         * shader multiplies this by the cone it measures per pixel, so the result is the cap in BLOCKS.
         *
         * <p>A pixel's cone is {@code 2 / m11} in the shader's units, and the frame is {@code renderHeight}
         * pixels tall, so covering {@code MAX_DEFOCUS_FRAME_FRACTION} of the frame is
         * {@code fraction * renderHeight} pixel cones. Expressing it that way is the whole point: it cancels the
         * focal length and the sensor size, so the cap is the same fraction of the image at any field of view,
         * aspect ratio and render resolution — exactly what the effect needs, and exactly what the hand-written
         * slope/film conversions kept getting wrong.
         */
        public float maxDefocusConeMultiple() {
            return maxDefocusFraction() * renderHeight;
        }
    }

    // Auto programme history: the smoothed EV100 the lens is exposing for, the aperture it last settled on, the
    // exposure the lens is currently contributing, and the clock the smoothing was last advanced by.
    private static float autoLensEv = Float.NaN;
    private static long lastAutoNanos;
    private static float smoothedPhysicalScale = Float.NaN;
    /** Interval between latches, in seconds; shared by the programme's and the scale's smoothing. */
    private static float lastFrameSeconds = 1.0f / 60.0f;

    /**
     * Solves this frame's aperture, shutter time and focus distance.
     *
     * @param projection   the frame's projection matrix, whose [1][1] term carries the vertical FOV
     * @param lensTargetEv100 the EV100 the lens should expose for, relative to the reference pair: the metered
     *                     scene's own EV plus the reference pair's EV, because a lens scale of 1.0 is defined by
     *                     the reference pair rather than by the scene convention; {@code NaN} while nothing has
     *                     been metered yet, which leaves the lens at the reference pair
     * @param meteredFocus auto-focus distance in blocks sampled from the frame's depth guide, or a
     *                     non-positive value when it is not available yet
     * @param renderHeight the height the frame is traced at, in pixels; sizes the blurred-circle cap
     */
    public static Frame latch(Matrix4fc projection, float lensTargetEv100, float meteredFocus,
                              int renderHeight) {
        float focalLength = focalLengthMetres(projection);
        float targetEv100 = autoTargetEv100(lensTargetEv100);
        float aperture = resolveAperture(targetEv100);
        float shutter = resolveShutter(targetEv100, aperture);
        float focus = resolveFocusDistance(CausticaConfig.Rt.Camera.FOCUS_MODE.get(),
                CausticaConfig.Rt.Camera.FOCUS_DISTANCE.value(), meteredFocus);
        float defocus = defocusScaleBlocks(focalLength, aperture);
        return new Frame(aperture, shutter, focus, SENSOR_HEIGHT_METRES, focalLength, defocus,
                smoothPhysicalScale(physicalExposureScale(aperture, shutter)), Math.max(renderHeight, 1));
    }

    /**
     * The lens' contribution to the image, smoothed over the same time constant the lens itself moves on.
     *
     * <p>The aperture and shutter are quantized because they are physical parts, but the exposure they imply
     * does not have to arrive in steps: moving one aperture rung changes the image by up to a stop in a single
     * frame. Smoothing the delivered scale keeps the physical quantization while removing the step, which is
     * what the eye reads as a jump. Only a manual pair delivers this scale at all; under the auto programme the
     * lens is left out of the exposure (see RtExposure.lensExposureScale).
     */
    private static float smoothPhysicalScale(float target) {
        if (Float.isNaN(smoothedPhysicalScale) || !(smoothedPhysicalScale > 0.0f)) {
            smoothedPhysicalScale = target;
            return target;
        }
        float elapsed = lastFrameSeconds;
        float alpha = 1.0f - (float) Math.exp(-elapsed / AUTO_ADAPT_SECONDS);
        smoothedPhysicalScale += (target - smoothedPhysicalScale) * Math.clamp(alpha, 0.0f, 1.0f);
        return smoothedPhysicalScale;
    }

    /**
     * Focal length in metres for the frame's field of view, from the pinhole relation
     * {@code m11 = 1 / tan(fovY / 2)}.
     *
     * <p>Minecraft builds its level projection with {@code Matrix4f.setPerspective(fov, aspect, ...)}, whose
     * first argument is the <em>vertical</em> field of view: the [1][1] term is {@code 1 / tan(fov / 2)} and
     * [0][0] is that divided by the aspect ratio, which is why a wider window shows more horizontally at the
     * same setting. Focal length therefore comes from [1][1] and the sensor's height, and the horizontal FOV
     * that follows from the aspect ratio does not change how tight the lens is. View bobbing is a rigid
     * transform applied to the projection, so it leaves both terms untouched.
     */
    public static float focalLengthMetres(Matrix4fc projection) {
        float m11 = projection.m11();
        float tanHalfFov = m11 > 1.0e-6f ? 1.0f / m11 : 1.0f;
        return SENSOR_HEIGHT_METRES / (2.0f * tanHalfFov);
    }

    /**
     * Circle of confusion in blocks per block of distance from the eye, for the worst case of a focal plane
     * at infinity: the largest image-space blur the lens can produce at any depth. The shader scales it by
     * {@code |Zf / Z - 1|}, so this is the only lens quantity it needs.
     */
    public static float defocusScaleBlocks(float focalLengthMetres, float aperture) {
        return Math.max(focalLengthMetres, 0.0f) / Math.max(aperture, 1.0e-4f);
    }

    /**
     * The pair a frame settled on, as its own EV100: what the scale below is relative to.
     */
    public static float frameEv100(Frame frame) {
        return CausticaConfig.Rt.Camera.ev100(frame.aperture(), frame.shutterSeconds());
    }

    /**
     * Exposure multiplier of a lens setting, in linear light: what the display chain multiplies the stored scene
     * radiance by.
     *
     * <p>It is normalised to the reference pair, f/2.8 at 1/60 s being exactly 1.0. That normalisation is not
     * arbitrary: the exposure controller's target is built as {@code preExposure * residual}, where the residual
     * is the multiplier on ALREADY PRE-EXPOSED values, so it assumes a lens contributing 1.0 and the lens is what
     * supplies the rest. {@link #SCENE_REFERENCE_EV100} is the EV100 at which that reference pair exposes a scene
     * at the metering key, and it is what turns this camera-relative scale into a scene exposure.
     */
    public static float physicalExposureScale(float aperture, float shutterSeconds) {
        float ev100 = CausticaConfig.Rt.Camera.ev100(aperture, shutterSeconds);
        return (float) Math.pow(2.0, CausticaConfig.Rt.Camera.REFERENCE_EV100 - ev100);
    }

    /**
     * Converts the guide's non-linear, reversed-Z depth into the distance along the view axis, in blocks.
     *
     * <p>For a reversed-Z projection built as {@code m22 = -far / (far - near)} and
     * {@code m32 = -near * far / (far - near)}, the guide holds {@code depth = -m22 + m32 / d}: the depth
     * approaches 1 at the near plane and 0 at infinity, which is what makes the near field the precise end of
     * the range. Inverting that gives {@code d = m32 / (depth + m22)}.
     *
     * <p>The acceptance is relative to the projection's scale rather than an absolute epsilon: an absolute one
     * would reject the far field, where the depth approaches zero and the denominator shrinks to {@code m22}.
     */
    public static float viewDistanceBlocks(float depth, float projM22, float projM32) {
        float denominator = depth + projM22;
        float acceptance = DEPTH_EPSILON * Math.max(1.0f, Math.abs(projM22) + Math.abs(projM32));
        if (!(depth > 0.0f) || Math.abs(denominator) < acceptance) {
            return Float.POSITIVE_INFINITY;
        }
        float distance = projM32 / denominator;
        return distance > 0.0f ? distance : Float.POSITIVE_INFINITY;
    }

    /**
     * The distance the lens is focused at. The metered value is along the primary ray, which is what the
     * circle of confusion is defined against; the manual setting is used verbatim, with very large values
     * standing in for infinity.
     */
    public static float resolveFocusDistance(String focusMode, float manualBlocks, float meteredBlocks) {
        if (!"manual".equals(focusMode) && meteredBlocks > 0.0f && Float.isFinite(meteredBlocks)) {
            return Math.min(meteredBlocks, FOCUS_INFINITY_BLOCKS);
        }
        return Math.clamp(manualBlocks, 0.25f, FOCUS_INFINITY_BLOCKS);
    }

    /**
     * The EV100 the auto programme aims for, low-passed: the metered scene's own EV100 plus the compensation the
     * look package applies, i.e. the exposure the frame wants.
     *
     * <p>Driving from the metered scene rather than from the exposure controller's remaining correction is what
     * keeps the two loops from fighting: the controller's residual is only zero once it has converged, so a lens
     * following it racks the iris on every scene change, while the scene's EV100 is a property of the frame. The
     * target is still low-passed, because an aperture is a mechanical part rather than something that re-racks
     * on every flicker of the histogram, and because a scene sitting exactly on a rung boundary would otherwise
     * hunt all night.
     */
    public static float autoTargetEv100(float lensTargetEv100) {
        advanceClock();
        float instantaneous = Float.isFinite(lensTargetEv100) ? lensTargetEv100 : AUTO_FALLBACK_EV100;
        float elapsed = lastFrameSeconds;
        if (Float.isNaN(autoLensEv)) {
            autoLensEv = instantaneous;
        } else {
            float alpha = 1.0f - (float) Math.exp(-elapsed / AUTO_ADAPT_SECONDS);
            autoLensEv += (instantaneous - autoLensEv) * Math.clamp(alpha, 0.0f, 1.0f);
        }
        // Aperture and shutter are quantized to whole and third stops, so the programme is always within half
        // a stop of its target.
        return autoLensEv;
    }

    /** Drops the auto programme's history so a new world starts from the freshly metered scene. */
    public static void reset() {
        autoLensEv = Float.NaN;
        autoAperture = Float.NaN;
        smoothedPhysicalScale = Float.NaN;
        lastAutoNanos = 0L;
        lastFrameSeconds = 1.0f / 60.0f;
    }

    /** Advances the shared frame-interval estimate, once per latch. */
    private static void advanceClock() {
        long now = System.nanoTime();
        lastFrameSeconds = lastAutoNanos == 0L
                ? 1.0f / 60.0f
                : Math.clamp((now - lastAutoNanos) / 1_000_000_000.0f, MIN_FRAME_SECONDS, MAX_FRAME_SECONDS);
        lastAutoNanos = now;
    }

    /**
     * Aperture for this frame. In manual mode it is the setting verbatim; in auto mode it is the rung whose best
     * shutter rung lands closest to the target exposure, with the manually chosen f-number preferred unless
     * another rung beats it by {@link #AUTO_EXPOSURE_TOLERANCE_STOPS} — a camera's program line, where the
     * aperture holds and the shutter does the work until it runs out of rungs.
     */
    public static float resolveAperture(float targetEv100) {
        CausticaConfig.FloatSetting manual = CausticaConfig.Rt.Camera.MANUAL_APERTURE;
        if (!"auto".equals(CausticaConfig.Rt.Camera.APERTURE_MODE.get())) {
            return manual.value();
        }
        // The whole ladder is searched, in both directions: a scene far off the reference may need the iris
        // opened to gather light or the shutter simply may not reach. Each rung is scored by the exposure error
        // its best shutter leaves, and the preset only wins when nothing beats it by more than the margin --
        // which is what makes the programme aperture priority rather than a per-frame optimizer.
        float preset = manual.value();
        float best = preset;
        float bestError = exposureError(targetEv100, CausticaConfig.Rt.Camera.apertureStops(preset));
        for (int rung = CausticaConfig.Rt.Camera.APERTURE_MIN_STOP;
             rung <= CausticaConfig.Rt.Camera.APERTURE_MAX_STOP; rung++) {
            float stops = CausticaConfig.Rt.Camera.apertureStops(CausticaConfig.Rt.Camera.fStop(rung));
            float error = exposureError(targetEv100, stops);
            if (error + AUTO_EXPOSURE_TOLERANCE_STOPS < bestError) {
                best = CausticaConfig.Rt.Camera.fStop(rung);
                bestError = error;
            }
        }
        return best;
    }

    /** Shutter time in seconds for this frame, given the aperture already solved. */
    public static float resolveShutter(float targetEv100, float aperture) {
        if (!"auto".equals(CausticaConfig.Rt.Camera.SHUTTER_MODE.get())) {
            return CausticaConfig.Rt.Camera.MANUAL_SHUTTER_TIME.value();
        }
        return shutterSecondsAt(shutterStopsFor(targetEv100, aperture));
    }

    /**
     * Shutter-time ladder position for the aperture this frame solved, as the quantity {@code log2(1 / t)}:
     * 0 is a one-second exposure, 5.9 is 1/60 s, 11 is 1/2000 s. A brighter scene (a lower EV100) sits higher on
     * the dial. From {@code EV100 = 2 * log2(N) - log2(t)} this is {@code EV100 - 2 * log2(N)}.
     *
     * <p>Kept continuous rather than snapped so callers can tell whether the ladder could express the target at
     * all.
     */
    public static float shutterStopsFor(float targetEv100, float aperture) {
        return targetEv100 - CausticaConfig.Rt.Camera.apertureStops(aperture);
    }

    /**
     * The ladder position a shutter time sits at, in the same units as {@link #shutterStopsFor}: the count of
     * stops above one second. A time longer than a second lands on a negative position, which is the side of the
     * dial where the shutter has run out of speed and the aperture has to take over.
     */
    public static float shutterStopsOf(float shutterSeconds) {
        return (float) (-Math.log(Math.max(shutterSeconds, 1.0e-9)) / Math.log(2.0));
    }

    /**
     * Exposure value at ISO 100 of a pair, on the same {@code EV100 = 2 * log2(N) + log2(1/t)} form both ladders
     * are indexed on. Higher means less light reaching the frame.
     */
    private static float pairEv100(float apertureStops, float shutterStops) {
        return apertureStops + shutterStops;
    }

    /**
     * How far the best shutter rung for an aperture misses the target exposure, in stops. The shutter only moves a
     * whole rung at a time, so this is the error the exposure controller has to absorb.
     */
    private static float exposureError(float targetEv100, float apertureStops) {
        float rung = shutterStopsOf(shutterSecondsAt(targetEv100 - apertureStops));
        return Math.abs(pairEv100(apertureStops, rung) - targetEv100);
    }

    /** Nearest aperture rung to a continuous f-stop position; outside the ladder it clamps to an end rung. */
    public static float fStopAt(float stops) {
        int rung = Math.round(stops) - 1;
        rung = Math.clamp(rung, CausticaConfig.Rt.Camera.APERTURE_MIN_STOP,
                CausticaConfig.Rt.Camera.APERTURE_MAX_STOP);
        return CausticaConfig.Rt.Camera.fStop(rung);
    }

    /**
     * Nearest shutter rung to a continuous ladder position; outside the ladder it clamps to an end rung. Compared
     * in log space, because the rungs are a fixed ratio apart and an absolute difference would weight the long
     * exposures far more heavily than the short ones.
     */
    public static float shutterSecondsAt(float stops) {
        if (!Float.isFinite(stops)) {
            return stops < 0.0f
                    ? CausticaConfig.Rt.Camera.shutterSeconds(1)
                    : CausticaConfig.Rt.Camera.shutterSeconds(
                            CausticaConfig.Rt.Camera.SHUTTER_DENOMINATORS.get(0));
        }
        float best = CausticaConfig.Rt.Camera.shutterSeconds(1);
        float bestDistance = Float.MAX_VALUE;
        for (int denominator : CausticaConfig.Rt.Camera.SHUTTER_DENOMINATORS) {
            // The rung sits at log2(denominator) stops above one second, which is the same quantity `stops`
            // counts, so the two are directly comparable.
            float distance = Math.abs(stops - (float) (Math.log(denominator) / Math.log(2.0)));
            if (distance < bestDistance) {
                bestDistance = distance;
                best = CausticaConfig.Rt.Camera.shutterSeconds(denominator);
            }
        }
        return best;
    }
}
