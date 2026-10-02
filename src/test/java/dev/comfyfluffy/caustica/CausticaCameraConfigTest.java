package dev.comfyfluffy.caustica;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The camera's two ladders are what a player dials, and the exposure scale is derived from them, so their
 * spacing and endpoints are pinned rather than left to the slider positions in the options screen.
 */
final class CausticaCameraConfigTest {
    private static final float EPSILON = 1.0e-4f;

    @Test
    void apertureLadderIsWholeStopsFromF14ToF32() {
        // The rungs are the standard f-number series: 2^(stop/2) is sqrt(2) steps, so 5.6 is 5.657 and 22 is
        // 22.6 on the exact scale -- the numbers a lens barrel is marked with.
        List<Float> expected = List.of(1.4f, 2.0f, 2.8f, 4.0f, 5.7f, 8.0f, 11.3f, 16.0f, 22.6f, 32.0f);
        for (int i = 0; i < expected.size(); i++) {
            int stop = CausticaConfig.Rt.Camera.APERTURE_MIN_STOP + i;
            assertEquals(expected.get(i), CausticaConfig.Rt.Camera.fStop(stop), 0.05f,
                    "f-number at ladder stop " + stop);
        }

        // One rung is one stop, i.e. one EV of exposure: the scale the auto programme walks in.
        for (int stop = CausticaConfig.Rt.Camera.APERTURE_MIN_STOP;
             stop < CausticaConfig.Rt.Camera.APERTURE_MAX_STOP; stop++) {
            float stepEv = CausticaConfig.Rt.Camera.apertureStops(CausticaConfig.Rt.Camera.fStop(stop + 1))
                    - CausticaConfig.Rt.Camera.apertureStops(CausticaConfig.Rt.Camera.fStop(stop));
            assertEquals(1.0f, stepEv, EPSILON, "ladder step at stop " + stop);
        }
    }

    @Test
    void shutterLadderIsDescendingAndMeetsTheApertureRange() {
        List<Integer> ladder = CausticaConfig.Rt.Camera.SHUTTER_DENOMINATORS;
        for (int i = 1; i < ladder.size(); i++) {
            assertTrue(ladder.get(i) < ladder.get(i - 1),
                    "the ladder must run from the fastest shutter to the slowest: " + ladder);
        }
        assertEquals(1.0f / 60.0f, CausticaConfig.Rt.Camera.shutterSeconds(60), EPSILON);
        assertEquals(1.0f / 2000.0f, CausticaConfig.Rt.Camera.shutterSeconds(2000), EPSILON);

        // Together the two ladders have to reach past both ends of a physically plausible exposure, so the
        // auto programme is bounded by its rungs rather than by an arbitrary limit: 1/2000 s at f/32 is a
        // snowfield at noon, 1 s at f/1.4 is a moonlit one.
        assertTrue(CausticaConfig.Rt.Camera.ev100(32.0f, 1.0f / 2000.0f) > 20.0f);
        assertTrue(CausticaConfig.Rt.Camera.ev100(1.4f, 1.0f) < 1.0f);
    }

    @Test
    void manualSettingsAreClampedToTheLadders() {
        CausticaConfig.FloatSetting aperture = CausticaConfig.Rt.Camera.MANUAL_APERTURE;
        CausticaConfig.FloatSetting shutter = CausticaConfig.Rt.Camera.MANUAL_SHUTTER_TIME;
        CausticaConfig.FloatSetting focus = CausticaConfig.Rt.Camera.FOCUS_DISTANCE;
        float previousAperture = aperture.value();
        float previousShutter = shutter.value();
        float previousFocus = focus.value();
        try {
            aperture.set(0.5f);
            assertEquals(CausticaConfig.Rt.Camera.fStop(CausticaConfig.Rt.Camera.APERTURE_MIN_STOP),
                    aperture.value(), EPSILON);
            aperture.set(500.0f);
            assertEquals(CausticaConfig.Rt.Camera.fStop(CausticaConfig.Rt.Camera.APERTURE_MAX_STOP),
                    aperture.value(), EPSILON);

            // Both ends of the shutter ladder are hard bounds, so the manual value can never express a shutter
            // time the virtual camera could not actually hold open.
            shutter.set(0.0f);
            assertEquals(CausticaConfig.Rt.Camera.shutterSeconds(2000), shutter.value(), EPSILON);
            shutter.set(10.0f);
            assertEquals(CausticaConfig.Rt.Camera.shutterSeconds(1), shutter.value(), EPSILON);

            // A focus distance may not collapse onto the lens; 0 would otherwise be reachable from the TOML
            // file and would make the whole frame one infinite circle of confusion.
            focus.set(0.0f);
            assertTrue(focus.value() >= 0.25f, "focus distance must stay off the lens: " + focus.value());
        } finally {
            aperture.set(previousAperture);
            shutter.set(previousShutter);
            focus.set(previousFocus);
        }
    }

    @Test
    void modesRejectUnknownValues() {
        CausticaConfig.StringSetting apertureMode = CausticaConfig.Rt.Camera.APERTURE_MODE;
        CausticaConfig.StringSetting shutterMode = CausticaConfig.Rt.Camera.SHUTTER_MODE;
        CausticaConfig.StringSetting focusMode = CausticaConfig.Rt.Camera.FOCUS_MODE;
        String previousApertureMode = apertureMode.get();
        String previousShutterMode = shutterMode.get();
        String previousFocusMode = focusMode.get();
        try {
            apertureMode.set("AUTO");
            assertEquals("auto", apertureMode.get());
            apertureMode.set("nonsense");
            assertEquals("auto", apertureMode.get());
            shutterMode.set("MANUAL");
            assertEquals("manual", shutterMode.get());
            focusMode.set("nonsense");
            assertEquals("auto", focusMode.get());
        } finally {
            apertureMode.set(previousApertureMode);
            shutterMode.set(previousShutterMode);
            focusMode.set(previousFocusMode);
        }
    }
}
