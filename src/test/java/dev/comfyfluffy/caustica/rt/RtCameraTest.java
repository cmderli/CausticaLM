package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.rt.pipeline.RtExposure;
import org.joml.Matrix4f;
import org.joml.Vector4f;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The camera's lens is one physical model shared by three features — depth of field, the auto aperture and
 * shutter programme, and the exposure the display pass applies — so its arithmetic is pinned here rather than
 * left to be discovered by looking at a render.
 */
final class RtCameraTest {
    private static final float EPSILON = 1.0e-4f;

    /** f/2.8 at 1/60 s: the combination the physical exposure scale is measured from. */
    private static final float REFERENCE_EV100 = 8.877744f;

    /**
     * Asserts the programme landed on the shutter rung nearest {@code expectedDenominator}, a reciprocal second
     * as the ladder is named. The rungs are up to half a stop apart, so an exact comparison would be asserting
     * the ladder rather than the choice; the ladder itself is pinned by CausticaCameraConfigTest.
     */
    private static void assertShutterRung(float expectedDenominator, float actualSeconds) {
        float actualDenominator = 1.0f / actualSeconds;
        // One stop: the shutter only moves a whole rung, so the assertion is that it moved to the right rung
        // given where it started, not that it hit an exact time.
        assertEquals(Math.log(expectedDenominator) / Math.log(2.0), Math.log(actualDenominator) / Math.log(2.0),
                1.0, "expected about 1/" + Math.round(expectedDenominator) + " s but was 1/"
                        + Math.round(actualDenominator));
    }

    /** The nearest ladder rung to a shutter time, for checking that the programme stays on the ladder. */
    private static float nearestRungSeconds(float seconds) {
        float best = 1.0f;
        float bestDistance = Float.MAX_VALUE;
        for (int denominator : CausticaConfig.Rt.Camera.SHUTTER_DENOMINATORS) {
            float candidate = CausticaConfig.Rt.Camera.shutterSeconds(denominator);
            float distance = Math.abs((float) Math.log(candidate / seconds));
            if (distance < bestDistance) {
                bestDistance = distance;
                best = candidate;
            }
        }
        return best;
    }

    /** A projection built the way Minecraft builds one: the FOV argument is the vertical field of view. */
    private static Matrix4f projectionForFov(float degrees) {
        return new Matrix4f().setPerspective((float) Math.toRadians(degrees), 16.0f / 9.0f, 0.05f, 1000.0f, true);
    }

    /**
     * Runs {@code body} with both camera modes on auto and the given preset aperture, restoring them after. The
     * programme's smoothing history is dropped first: it is deliberately sticky across frames, so a test that
     * inherited it would be testing whatever ran before rather than the rule in front of it.
     */
    private static void withAutoLens(float aperture, Runnable body) {
        RtCamera.reset();
        CausticaConfig.StringSetting apertureMode = CausticaConfig.Rt.Camera.APERTURE_MODE;
        CausticaConfig.StringSetting shutterMode = CausticaConfig.Rt.Camera.SHUTTER_MODE;
        CausticaConfig.FloatSetting apertureSetting = CausticaConfig.Rt.Camera.MANUAL_APERTURE;
        String previousApertureMode = apertureMode.get();
        String previousShutterMode = shutterMode.get();
        float previousAperture = apertureSetting.value();
        try {
            apertureMode.set("auto");
            shutterMode.set("auto");
            apertureSetting.set(aperture);
            body.run();
        } finally {
            apertureMode.set(previousApertureMode);
            shutterMode.set(previousShutterMode);
            apertureSetting.set(previousAperture);
        }
    }

    @Test
    void focalLengthFollowsTheProjectionFieldOfView() {
        // Minecraft hands its FOV option straight to Matrix4f.setPerspective, whose FOV argument lands on the
        // projection's [1][1] term. That term, and the sensor height behind it, are what set the lens.
        Matrix4f projection = projectionForFov(70.0f);
        assertEquals(0.024f / (2.0f * (float) Math.tan(Math.toRadians(35.0))),
                RtCamera.focalLengthMetres(projection), EPSILON);

        // 70 degrees vertical on a 24 mm-tall frame is a 17 mm lens: a wide angle, which is what a Minecraft
        // field of view is. Sanity check that the sensor constant is in the right unit.
        assertEquals(0.01714f, RtCamera.focalLengthMetres(projection), 5.0e-4f);

        // A narrower FOV is a longer lens, which is what focuses tighter at the same aperture. This is the whole
        // reason the FOV option changes the depth of field.
        assertTrue(RtCamera.focalLengthMetres(projectionForFov(30.0f))
                > RtCamera.focalLengthMetres(projectionForFov(90.0f)));

        // Aspect ratio must not move the focal length: the same FOV option on a wider window is the same lens,
        // showing more of the world rather than a different depth of field.
        Matrix4f wide = new Matrix4f().setPerspective(
                (float) Math.toRadians(70.0f), 21.0f / 9.0f, 0.05f, 1000.0f, true);
        assertEquals(RtCamera.focalLengthMetres(projection), RtCamera.focalLengthMetres(wide), EPSILON);
    }

    @Test
    void defocusRadiusMatchesTheThinLensCircleOfConfusion() {
        float focalLength = 0.025f;
        float aperture = 2.8f;
        // The shader aims its off-centre rays at the focal plane from a point on the entrance pupil, whose
        // radius is f / (2N), so the on-film confusion disc of a point at Z is (f / N) * |Zf / Z - 1|. This
        // constant is that (f / N): the shader only has to scale it by the depth ratio.
        assertEquals(focalLength / aperture, RtCamera.defocusScaleBlocks(focalLength, aperture), EPSILON);

        // The disc grows in proportion to how far a point sits from the focal plane. A point twice as far behind
        // it as another is three times as blurred, because |Zf / Z - 1| is 1 at Z = Zf/2 and 1/3 at Z = 3Zf/4.
        float focusDistance = 8.0f;
        float scale = RtCamera.defocusScaleBlocks(focalLength, aperture);
        assertEquals(3.0f,
                (scale * Math.abs(focusDistance / 4.0f - 1.0f)) / (scale * Math.abs(focusDistance / 6.0f - 1.0f)),
                1.0e-3f);

        // A wider aperture blurs proportionally more; one stop is a factor of sqrt(2) in the f-number. The
        // ladder's f-numbers are exact powers of two rather than the rounded markings on a barrel, so the ratio
        // is compared loosely.
        assertEquals(Math.sqrt(2.0f),
                RtCamera.defocusScaleBlocks(focalLength, 2.0f)
                        / RtCamera.defocusScaleBlocks(focalLength, 2.828427f), 0.02f);
    }

    /**
     * The blur cap, in pixels — the only unit in which "is the depth of field visible" has an answer.
     *
     * <p>The shader hands the defocus radius to the pixel by dividing it by one pixel's own ray cone, so the
     * multiple it is given times that cone IS the radius in pixels. The cap therefore has to be
     * {@code fraction * renderHeight} pixel cones, and the check below divides it back out: at 1080p and a 5%
     * fraction that is 54 pixels, and every earlier version of this measured between 0.5 and 5 pixels, which is
     * why the effect kept disappearing.
     */
    @Test
    void defocusCapIsAFractionOfTheFrameInPixels() {
        for (float fov : new float[] {70.0f, 30.0f, 110.0f}) {
            for (int height : new int[] {1080, 2160}) {
                Matrix4f projection = projectionForFov(fov);
                // One pixel's ray cone at the frame centre, exactly as primaryRayConeSpread measures it.
                float cone = 2.0f / projection.m11();
                float capBlocks = RtCamera.latch(projection, 0.0f, 0.0f, height).maxDefocusConeMultiple() * cone;
                float capPixels = capBlocks / cone;
                assertEquals(RtCamera.MAX_DEFOCUS_FRAME_FRACTION * height, capPixels, 0.5f,
                        "the cap must be " + RtCamera.MAX_DEFOCUS_FRAME_FRACTION + " of the frame at "
                                + fov + " deg, " + height + "p");
                assertTrue(capPixels > 20.0f, "and that has to be a visible blur, was " + capPixels);
            }
        }
    }

    @Test
    void physicalExposureScaleFollowsTheLens() {
        // f/2.8 at 1/60 s is the reference the scale is relative to, so it must come out as exactly 1.0 --
        // otherwise every existing configuration would shift the moment these options were added.
        assertEquals(REFERENCE_EV100, CausticaConfig.Rt.Camera.ev100(2.8f, 1.0f / 60.0f), 0.01f);
        assertEquals(1.0f, RtCamera.physicalExposureScale(2.8f, 1.0f / 60.0f), EPSILON);

        // Two stops narrower squares the f-number twice: a quarter of the light. The ladder's f-numbers are the
        // exact sqrt(2) steps rather than the rounded markings on a barrel, so the comparison is against the
        // scale that pair produces (0.245 for the exact f/5.657) rather than a quarter exactly.
        assertEquals(0.25f, RtCamera.physicalExposureScale(5.6568542f, 1.0f / 60.0f), 0.01f);
        // Three ladder rungs wider than f/4 is f/1.414, and three stops is 2^3 times the light.
        assertEquals(8.0f * RtCamera.physicalExposureScale(4.0f, 1.0f / 60.0f),
                RtCamera.physicalExposureScale(1.4142135f, 1.0f / 60.0f), 0.1f);
        // One stop longer doubles the shutter time: twice the light. One stop shorter halves it.
        assertEquals(2.0f, RtCamera.physicalExposureScale(2.8f, 1.0f / 30.0f), EPSILON);
        assertEquals(0.5f, RtCamera.physicalExposureScale(2.8f, 1.0f / 120.0f), EPSILON);
    }

    /**
     * A neutral scene has to leave the lens exactly where it started, and stay there: the programme is fed the
     * metered scene every frame, so any drift in its settled output would show up as the iris slowly walking
     * across its range.
     */
    @Test
    void autoProgrammeSettlesInsteadOfDrifting() {
        RtCamera.reset();
        float first = RtCamera.autoTargetEv100(REFERENCE_EV100);
        for (int frame = 0; frame < 600; frame++) {
            assertEquals(first, RtCamera.autoTargetEv100(REFERENCE_EV100), EPSILON,
                    "a constant metered scene must not move the target");
        }
        // And the settled target is the reference combination, so the lens contributes exactly 1.0.
        assertShutterRung(60.0f, RtCamera.resolveShutter(first, 2.8f));
        assertEquals(1.0f, RtCamera.latch(projectionForFov(70.0f), REFERENCE_EV100, 0.0f, 1080).physicalExposureScale(), EPSILON);
    }

    /**
     * A very bright or very dark scene asks the lens for more than either ladder can give. The programme has to
     * answer with a rung at the end of its range rather than refusing or oscillating: the aperture is the only
     * control beyond the shutter's span, so clamping it would freeze the lens and hand the whole correction back
     * to the exposure controller.
     */
    @Test
    void autoProgrammeUsesTheFullApertureRange() {
        withAutoLens(2.8f, () -> {
            float darkestPair = CausticaConfig.Rt.Camera.ev100(
                    CausticaConfig.Rt.Camera.fStop(CausticaConfig.Rt.Camera.APERTURE_MAX_STOP),
                    CausticaConfig.Rt.Camera.shutterSeconds(1));
            float brightestPair = CausticaConfig.Rt.Camera.ev100(
                    CausticaConfig.Rt.Camera.fStop(CausticaConfig.Rt.Camera.APERTURE_MIN_STOP),
                    CausticaConfig.Rt.Camera.shutterSeconds(
                            CausticaConfig.Rt.Camera.SHUTTER_DENOMINATORS.get(0)));

            // Both ends of the range are reachable as the scene walks past what the shutter can cover.
            boolean reachedWideOpen = false;
            boolean reachedStoppedDown = false;
            for (float scene = -4.0f; scene <= 26.0f; scene += 0.25f) {
                float aperture = RtCamera.resolveAperture(scene);
                float stops = CausticaConfig.Rt.Camera.apertureStops(aperture);
                assertEquals(Math.round(stops), stops, 0.05f,
                        "aperture off the ladder at scene " + scene + ": f/" + aperture);
                reachedWideOpen |= Math.abs(aperture
                        - CausticaConfig.Rt.Camera.fStop(CausticaConfig.Rt.Camera.APERTURE_MIN_STOP)) < 1.0e-3f;
                reachedStoppedDown |= Math.abs(aperture
                        - CausticaConfig.Rt.Camera.fStop(CausticaConfig.Rt.Camera.APERTURE_MAX_STOP)) < 1.0e-3f;
            }
            assertTrue(reachedWideOpen, "the programme must open to the widest rung for a bright enough scene");
            assertTrue(reachedStoppedDown, "the programme must stop down to the narrowest rung for a dim enough scene");

            // The pairs it settles on track the scene rather than sitting at the preset: somewhere in the middle
            // the aperture has to leave f/2.8, or the lens would be frozen and the controller would carry every
            // scene change alone.
            boolean leftThePreset = false;
            for (float scene = 0.0f; scene <= 25.0f; scene += 0.25f) {
                leftThePreset |= Math.abs(RtCamera.resolveAperture(scene) - 2.8f) > 1.0e-3f;
            }
            assertTrue(leftThePreset, "the auto aperture must move with the scene");
        });
    }

    /**
     * The lens programme's contract. Under the auto programme the lens does not contribute to the display
     * exposure at all (see RtExposure.lensExposureScale), so this is purely mechanical: which rung does the iris
     * take for the scene's brightness? It has to move monotonically, and the pair it picks has to be the one that
     * puts the frame nearest the metered exposure.
     */
    @Test
    void lensPairTracksTheSceneInTheRightDirection() {
        float reference = CausticaConfig.Rt.Camera.REFERENCE_EV100;
        float offset = RtSceneUnits.EV100_OFFSET;
        // Neutral scene: the lens target is the reference pair's own EV, so the pair is the reference pair.
        RtCamera.reset();
        RtCamera.Frame neutral = RtCamera.latch(projectionForFov(70.0f), reference, 0.0f, 1080);
        assertEquals(2.8f, neutral.aperture(), EPSILON);
        assertShutterRung(60.0f, neutral.shutterSeconds());

        // Brighter scenes take less light and darker ones more, with no rung ever reversing that.
        float previousLight = Float.MAX_VALUE;
        for (float evScene : new float[] {-2.0f, 1.5f, 3.0f, 7.0f, 12.0f, 17.45f}) {
            float target = RtExposure.lensTargetEv100(evScene);
            RtCamera.reset();
            RtCamera.Frame frame = RtCamera.latch(projectionForFov(70.0f), target, 0.0f, 1080);
            float light = 1.0f / CausticaConfig.Rt.Camera.ev100(frame.aperture(), frame.shutterSeconds());
            String pair = "f/" + frame.aperture() + " 1/" + Math.round(1.0f / frame.shutterSeconds());
            assertTrue(light < previousLight,
                    "a brighter scene must get less exposure: evScene " + evScene + " via " + pair);
            previousLight = light;

            // Within the ladders' range the pair lands within a rung and a half of what the scene asks for; past
            // the ends it saturates on the darkest or brightest pair, which is still monotonic and still as close
            // as the hardware allows.
            assertEquals(target, RtCamera.frameEv100(frame), 2.5f,
                    "lens target " + target + " landed on " + pair);
        }
    }

    @Test
    void autoProgrammeMeetsTheReferenceCombinationAtTheReferenceExposure() {
        RtCamera.reset();
        // No correction needed from the lens: the controller has already reached its target.
        float targetEv100 = RtCamera.autoTargetEv100(REFERENCE_EV100);
        assertEquals(REFERENCE_EV100, targetEv100, EPSILON);
        // Which the programme solves as the reference combination itself: f/2.8 at 1/60 s, unchanged from the
        // defaults, so a scene the controller is happy with leaves the lens where it started.
        assertEquals(2.8f, RtCamera.resolveAperture(targetEv100), EPSILON);
        assertShutterRung(60.0f, RtCamera.resolveShutter(targetEv100, 2.8f));
        assertEquals(1.0f, RtCamera.latch(projectionForFov(70.0f), REFERENCE_EV100, 0.0f, 1080)
                .physicalExposureScale(), EPSILON);
    }

    @Test
    void autoProgrammeIsAperturePriorityUntilTheShutterRunsOutOfRungs() {
        withAutoLens(2.8f, () -> {
            // The programme holds the f-number the player set and lets the shutter take the whole correction, so
            // a neutral scene keeps f/2.8 with the shutter rung nearest the exact 1/60 s the reference exposure
            // implies. Each case below checks the exposure value of the pair the programme chose, because the
            // rungs are a third to a half stop apart and asserting one rung would be asserting the ladder.
            assertEquals(2.8f, RtCamera.resolveAperture(REFERENCE_EV100), EPSILON);
            assertEquals(REFERENCE_EV100, CausticaConfig.Rt.Camera.ev100(2.8f, 1.0f / 60.0f), 0.01f);
            assertShutterRung(60.0f, RtCamera.resolveShutter(REFERENCE_EV100, 2.8f));

            // A dimmer scene needs a longer exposure, but the shutter only moves a whole rung at a time: one
            // stop of light at f/2.8 asks for 1/30 s, and if the rung nearest it is 1/125 then the pair still
            // lands well inside a stop of the scene (the exposure controller absorbs the remainder).
            float dimmer = REFERENCE_EV100 + 1.0f;
            assertEquals(2.8f, RtCamera.resolveAperture(dimmer), EPSILON);
            float dimmerShutter = RtCamera.resolveShutter(dimmer, 2.8f);
            assertTrue(dimmerShutter >= 1.0f / 125.0f && dimmerShutter <= 1.0f / 30.0f,
                    "the shutter must move into the 1/30..1/125 band, not past it: " + dimmerShutter);
            assertEquals(dimmer, CausticaConfig.Rt.Camera.ev100(2.8f, dimmerShutter), 1.0f);

            // Past 1 second the shutter is out of rungs, so the iris moves to carry the rest: the pair has to
            // land within a rung of the dim scene's exposure value rather than the 0.94 stops short the preset
            // leaves. Darker still and it would be the opening direction instead of the closing one.
            float dimScene = REFERENCE_EV100 + 6.0f;
            float adjusted = RtCamera.resolveAperture(dimScene);
            assertTrue(adjusted != 2.8f, "the aperture must move once the shutter is at its longest: " + adjusted);
            assertEquals(dimScene,
                    CausticaConfig.Rt.Camera.ev100(adjusted, RtCamera.resolveShutter(dimScene, adjusted)),
                    0.25f);

            // The mirror image: a bright scene shortens the shutter towards 1/2000 s with the aperture still
            // untouched, until the iris finally has to open to stay exposed.
            // Three stops brighter: the target is EV100 5.88, which at f/2.8 asks for 1/8 s and lands there.
            float brightScene = REFERENCE_EV100 - 3.0f;
            assertEquals(2.8f, RtCamera.resolveAperture(brightScene), EPSILON);
            assertShutterRung(8.0f, RtCamera.resolveShutter(brightScene, 2.8f));

            // One more stop of light and the aperture is still left alone, because the error the shutter's
            // coarse rung leaves is smaller than the error opening the iris one rung would introduce. Switching
            // to a different rung is only worth it once the shutter has genuinely run out.
            float veryBrightScene = REFERENCE_EV100 - 1.0f;
            assertEquals(2.8f, RtCamera.resolveAperture(veryBrightScene), EPSILON);
            assertTrue(
                    Math.abs(veryBrightScene - CausticaConfig.Rt.Camera.ev100(2.8f,
                            RtCamera.resolveShutter(veryBrightScene, 2.8f))) <= 1.0f,
                    "the pair must stay within a stop of the scene");
            // A mid-ladder bright scene: the pair stays within a stop of the target, which is as close as a
            // quantized ladder gets.
            float brightMidLadder = REFERENCE_EV100 - 4.0f;
            float closedDown = RtCamera.resolveAperture(brightMidLadder);
            assertEquals(brightMidLadder,
                    CausticaConfig.Rt.Camera.ev100(closedDown, RtCamera.resolveShutter(brightMidLadder, closedDown)),
                    1.0f);

            // Nine stops brighter still is past what the ladder can expose: the darkest pair it can make is
            // f/32 at 1/2000 s at EV100 20.97, and the brightest is EV100 0.97. The iris opens to its stop and
            // the shutter goes to the rung nearest what that aperture needs, which for a target this far off the
            // floor is the longest exposure it has; the sensor takes the excess, as a camera without an ND filter
            // does.
            float beyondTheLadder = REFERENCE_EV100 - 9.0f;
            float widest = CausticaConfig.Rt.Camera.fStop(CausticaConfig.Rt.Camera.APERTURE_MIN_STOP);
            assertEquals(widest, RtCamera.resolveAperture(beyondTheLadder), EPSILON);
            assertTrue(CausticaConfig.Rt.Camera.ev100(widest,
                    RtCamera.resolveShutter(beyondTheLadder, widest)) < 1.5f,
                    "the pair must sit near the ladder's brightest end");
        });
    }

    @Test
    void autoProgrammeLandsOnLadderValuesWhateverItIsAsked() {
        withAutoLens(2.8f, () -> {
            // The programme is quantized to the same rungs a photographer can set and must never leave them,
            // including at the extremes where the aperture clamps and the controller absorbs the rest.
            for (float target = -20.0f; target <= 40.0f; target += 0.37f) {
                float aperture = RtCamera.resolveAperture(target);
                float shutter = RtCamera.resolveShutter(target, aperture);
                // The preset aperture is on a rung of its own, so the result has to be on a rung too: the
                // distance to the nearest one is zero (within float precision) for every value returned.
                float apertureStops = CausticaConfig.Rt.Camera.apertureStops(aperture);
                assertEquals(Math.round(apertureStops), apertureStops, 0.05f,
                        "aperture off the ladder at target " + target + ": f/" + aperture);
                // The shutter rungs are the camera scale rather than powers of two, so the check is that the
                // time is exactly one of them.
                assertEquals(nearestRungSeconds(shutter), shutter, 1.0e-6f,
                        "shutter off the ladder at target " + target + ": 1/" + Math.round(1.0f / shutter));
            }
        });
    }

    @Test
    void manualLensIgnoresTheProgramme() {
        CausticaConfig.StringSetting apertureMode = CausticaConfig.Rt.Camera.APERTURE_MODE;
        CausticaConfig.StringSetting shutterMode = CausticaConfig.Rt.Camera.SHUTTER_MODE;
        CausticaConfig.FloatSetting aperture = CausticaConfig.Rt.Camera.MANUAL_APERTURE;
        CausticaConfig.FloatSetting shutter = CausticaConfig.Rt.Camera.MANUAL_SHUTTER_TIME;
        String previousApertureMode = apertureMode.get();
        String previousShutterMode = shutterMode.get();
        float previousAperture = aperture.value();
        float previousShutter = shutter.value();
        try {
            apertureMode.set("manual");
            shutterMode.set("manual");
            aperture.set(11.0f);
            shutter.set(1.0f / 250.0f);

            assertEquals(11.0f, RtCamera.resolveAperture(REFERENCE_EV100 + 9.0f), EPSILON);
            assertEquals(1.0f / 250.0f, RtCamera.resolveShutter(REFERENCE_EV100 + 9.0f, 11.0f), EPSILON);
            assertEquals(11.0f, RtCamera.latch(projectionForFov(70.0f), REFERENCE_EV100, 0.0f, 1080)
                    .aperture(), EPSILON);
        } finally {
            apertureMode.set(previousApertureMode);
            shutterMode.set(previousShutterMode);
            aperture.set(previousAperture);
            shutter.set(previousShutter);
        }
    }

    @Test
    void depthGuideDepthIsConvertedBackToADistance() {
        // The guide reports the distance along the view axis, so that is what the conversion must return, and
        // the round trip through the camera's own matrix is the invariant rather than the formula. Zero-to-one
        // depth is the convention the engine builds for a Vulkan device.
        Matrix4f projection = projectionForFov(70.0f);
        float projM22 = projection.m22();
        float projM32 = projection.m32();

        for (float ndcY : new float[] {0.0f, 0.6f}) {
            for (float rayDistance : new float[] {0.25f, 1.0f, 8.0f, 512.0f}) {
                // The world point `rayDistance` along the ray through the pixel. The projection's w is that
                // distance, which is why the depth inverts back to it and not to the shorter distance along
                // the view axis.
                float ndcX = 0.0f;
                org.joml.Vector3f ray = new org.joml.Vector3f(
                        ndcX / projection.m00(), ndcY / projection.m11(), -1.0f).normalize().mul(rayDistance);
                Vector4f point = new Vector4f(ray.x, ray.y, ray.z, 1.0f);
                Vector4f clip = projection.transform(point);
                float depth = clip.z / clip.w;
                // A float depth guide quantizes the far field, so the tolerance is the depth precision rather
                // than a fixed number of blocks. The round trip is what matters: the depth the projection wrote
                // for that point inverts back to the distance the point actually is away.
                assertEquals(clip.w, RtCamera.viewDistanceBlocks(depth, projM22, projM32),
                        Math.max(0.02f, rayDistance * 1.0e-3f),
                        "ray distance " + rayDistance + " at ndc.y " + ndcY);
            }
        }

        // Sky leaves the depth guide at zero, which must not be read as a focal plane at the lens.
        assertEquals(Float.POSITIVE_INFINITY, RtCamera.viewDistanceBlocks(0.0f, projM22, projM32));
        assertEquals(Float.POSITIVE_INFINITY, RtCamera.viewDistanceBlocks(-projM22, projM22, projM32));
    }

    @Test
    void narrowFieldOfViewWidensTheGatheredPartOfTheBlur() {
        // The whole point of splitting the aperture between the trace and the gather is that a narrow field of
        // view makes the blur bigger, not smaller: the focal length is sensorHeight / (2 tan(fovY / 2)), so the
        // same f-number projects a longer lens' aperture onto the film as a wider disc. The traced part is capped
        // at a fixed number of pixels, so a narrower FOV has to push the difference into the gather.
        float wideFov = RtCamera.focalLengthMetres(projectionForFov(90.0f));
        float narrowFov = RtCamera.focalLengthMetres(projectionForFov(30.0f));
        assertTrue(narrowFov > wideFov * 3.0f,
                "a 30 degree vertical FOV should be a far longer lens than 90: " + narrowFov + " vs " + wideFov);

        // The disc a point at infinity reaches is the lens radius on the film, in pixels.
        float renderHeight = 1080.0f;
        float aperture = 2.0f; // f/2.0
        float pixelsPerBlock = renderHeight / RtCamera.SENSOR_HEIGHT_METRES;
        float wideDisc = RtCamera.defocusScaleBlocks(wideFov, aperture) * pixelsPerBlock;
        float narrowDisc = RtCamera.defocusScaleBlocks(narrowFov, aperture) * pixelsPerBlock;
        assertTrue(narrowDisc > wideDisc * 3.0f,
                "the narrower FOV must produce a wider disc: " + narrowDisc + " vs " + wideDisc);

        // And the split must hand the trace the same capped disc either way, leaving the rest to the gather.
        for (float disc : new float[] {wideDisc, narrowDisc}) {
            float traced = disc * RtCamera.tracedDiscFraction(disc, RtCamera.TRACED_DISC_PIXELS);
            assertEquals(RtCamera.TRACED_DISC_PIXELS, traced, 1.0e-3f,
                    "the traced disc must hit the cap at disc " + disc);
            assertTrue(disc - traced > 0.0f, "the gather must be left something to do at disc " + disc);
        }
    }

    @Test
    void tracedDiscIsCappedSoTheGatherSuppliesTheRestOfTheAperture() {
        // A disc already smaller than the cap is traced whole: the gather then has nothing to add, which is what
        // keeps a modest aperture a purely ray-traced one.
        assertEquals(1.0f, RtCamera.tracedDiscFraction(RtCamera.TRACED_DISC_PIXELS * 0.5f,
                RtCamera.TRACED_DISC_PIXELS), 1.0e-6f);
        assertEquals(1.0f, RtCamera.tracedDiscFraction(RtCamera.TRACED_DISC_PIXELS,
                RtCamera.TRACED_DISC_PIXELS), 1.0e-6f);

        // A wide disc is traced only up to the cap, and the fraction is what the primary pass scales its lens
        // radius by. The disc scales linearly with both the lens radius and the render height, so the SAME
        // fraction always means the same number of traced pixels: doubling the aperture or the resolution
        // doubles the disc and leaves the traced part where it was, which is what stops a wide aperture or a
        // higher resolution from pushing the trace back into the speckle this cap exists to avoid.
        float wide = RtCamera.TRACED_DISC_PIXELS * 10.0f;
        assertEquals(0.1f, RtCamera.tracedDiscFraction(wide, RtCamera.TRACED_DISC_PIXELS), 1.0e-6f);
        // The fraction is per-frame, so what has to hold is that applying THIS frame's fraction to THIS frame's
        // disc lands on the cap -- at any aperture and any resolution.
        for (float disc : new float[] {wide, wide * 2.0f, wide * 0.5f, 4096.0f}) {
            float traced = disc * RtCamera.tracedDiscFraction(disc, RtCamera.TRACED_DISC_PIXELS);
            assertEquals(RtCamera.TRACED_DISC_PIXELS, traced, 1.0e-3f, "disc " + disc);
        }

        // Degenerate inputs must not produce a NaN fraction, which would reach the shader as a lens radius and
        // blank the frame: no disc means nothing to gather, so the trace keeps the whole (zero) aperture.
        assertEquals(1.0f, RtCamera.tracedDiscFraction(0.0f, RtCamera.TRACED_DISC_PIXELS), 1.0e-6f);
        assertEquals(1.0f, RtCamera.tracedDiscFraction(-1.0f, 1.0f), 1.0e-6f);
        assertEquals(1.0f, RtCamera.tracedDiscFraction(10.0f, 0.0f), 1.0e-6f);
        assertEquals(1.0f, RtCamera.tracedDiscFraction(Float.NaN, 1.0f), 1.0e-6f);
    }

    @Test
    void focusFadeFactorIsAFrameRateIndependentHalfLife() {
        // Half the remaining distance at exactly one half-life, whatever the frame rate: that is the property a
        // per-frame blend factor cannot have, and it is why the focal plane racks over the same wall-clock time
        // at 30 fps as at 240 fps.
        assertEquals(0.5f, RtCamera.focusFadeFactor(0.5f, 0.5f), 1.0e-4f);
        assertEquals(0.25f, RtCamera.focusFadeFactor(1.0f, 0.5f), 1.0e-4f);

        // Two frames of half the duration cover the same distance as one frame of the full duration, so the
        // smoothing is the discrete form of one exponential rather than a curve that depends on the step size.
        float oneStep = 1.0f - RtCamera.focusFadeFactor(1.0f / 60.0f, 0.5f);
        float twoSteps = 1.0f - RtCamera.focusFadeFactor(1.0f / 120.0f, 0.5f);
        twoSteps = twoSteps + (1.0f - twoSteps) * twoSteps;
        assertEquals(oneStep, twoSteps, 1.0e-5f);

        // A longer half-life moves less per frame; a zero or negative frame interval does not move at all.
        assertTrue(RtCamera.focusFadeFactor(1.0f / 60.0f, 2.0f)
                > RtCamera.focusFadeFactor(1.0f / 60.0f, 0.5f));
        assertEquals(1.0f, RtCamera.focusFadeFactor(0.0f, 0.5f), 1.0e-6f);
        assertEquals(1.0f, RtCamera.focusFadeFactor(-1.0f, 0.5f), 1.0e-6f);
        // The half-life is floored rather than allowed to divide by zero, so a stalled clock cannot snap it.
        assertEquals(RtCamera.focusFadeFactor(1.0f, 0.0f), RtCamera.focusFadeFactor(1.0f, 1.0e-4f), 1.0e-6f);
    }

    @Test
    void focusDistanceFollowsTheModeAndFallsBackWhenNothingIsMetered() {
        // Auto uses the metered centre distance...
        assertEquals(6.0f, RtCamera.resolveFocusDistance("auto", 20.0f, 6.0f), EPSILON);
        // ...and holds the manual value when the frame metered no surface (all sky).
        assertEquals(20.0f, RtCamera.resolveFocusDistance("auto", 20.0f, 0.0f), EPSILON);
        // Manual ignores the meter entirely.
        assertEquals(20.0f, RtCamera.resolveFocusDistance("manual", 20.0f, 6.0f), EPSILON);
    }
}
