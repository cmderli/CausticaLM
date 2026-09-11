package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import dev.comfyfluffy.caustica.rt.RtHdrSwapchainFormat.Decision;
import dev.comfyfluffy.caustica.rt.RtHdrSwapchainFormat.Pair;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The window swapchain's (format,colorSpace) policy. The interesting case is the sticky PQ choice: a
 * presentation stack that advertises PQ only while the surface is not already in PQ must not be able to flip
 * the decision on every recreation, because the flip itself is what makes the fresh swapchain suboptimal and
 * recreates it again.
 */
final class RtHdrSwapchainFormatTest {
    private static final int PQ = RtHdrSwapchainFormat.COLOR_SPACE_HDR10_ST2084;
    private static final int SRGB = RtHdrSwapchainFormat.COLOR_SPACE_SRGB_NONLINEAR;
    private static final int FMT_A2R10G10B10_UNORM_PACK32 = 64;
    private static final int FMT_A2B10G10R10_UNORM_PACK32 = 58;
    private static final int FMT_R16G16B16A16_SFLOAT = 97;
    private static final int FMT_R8G8B8A8_UNORM = 37;
    private static final int FMT_B8G8R8A8_UNORM = 44;
    private static final int FMT_B8G8R8A8_SRGB = 50;

    private static final List<Pair> PQ_AND_SDR = List.of(
            new Pair(FMT_A2R10G10B10_UNORM_PACK32, PQ),
            new Pair(FMT_R8G8B8A8_UNORM, SRGB));
    private static final List<Pair> SDR_ONLY = List.of(
            new Pair(FMT_B8G8R8A8_UNORM, SRGB));

    @Test
    void choosesPqWhileHdrIsEnabledAndNativeSdrOtherwise() {
        RtHdrSwapchainFormat format = new RtHdrSwapchainFormat();
        format.observe(PQ_AND_SDR);

        Decision pq = format.decide(true);
        assertTrue(pq.pq());
        assertEquals(FMT_A2R10G10B10_UNORM_PACK32, pq.format());
        assertEquals(PQ, pq.colorSpace());

        Decision sdr = format.decide(false);
        assertFalse(sdr.pq());
        assertEquals(FMT_R8G8B8A8_UNORM, sdr.format());
        assertEquals(SRGB, sdr.colorSpace());
        assertTrue(format.pqAdvertised(), "availability is a property of the surface, not of the toggle");
    }

    /**
     * The regression this policy exists for: enumerations that disagree about PQ must not make the swapchain
     * oscillate. Once PQ is seen it stays chosen, and the HDR state stays on.
     */
    @Test
    void keepsPqAcrossEnumerationsThatStopAdvertisingIt() {
        RtHdrSwapchainFormat format = new RtHdrSwapchainFormat();

        // First configure: the compositor is in SDR mode, so it advertises both pairs.
        format.observe(PQ_AND_SDR);
        assertTrue(format.decide(true).pq());

        // Later configures: the surface is already PQ, so the driver reports SDR only. Every one of them must
        // still choose PQ rather than flipping back to SDR.
        List<Boolean> decisions = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            format.observe(SDR_ONLY);
            decisions.add(format.decide(true).pq());
        }
        assertFalse(decisions.contains(false), "the PQ decision must not flip once PQ has been seen: " + decisions);
        assertTrue(format.pqAdvertised());
    }

    /** A rejected creation is the one thing that turns PQ off, and it switches to SDR from then on. */
    @Test
    void fallsBackToNativeSdrWhenThePresentationStackRejectsPq() {
        RtHdrSwapchainFormat format = new RtHdrSwapchainFormat();
        format.observe(PQ_AND_SDR);
        assertTrue(format.decide(true).pq());

        format.rejectPq();
        Decision afterRejection = format.decide(true);
        assertFalse(afterRejection.pq());
        assertEquals(FMT_R8G8B8A8_UNORM, afterRejection.format());
        assertFalse(format.pqAdvertised(), "a rejected PQ pair is no longer offered to the player");
    }

    @Test
    void ranksTenBitAboveFloatAndSkipsSrgbForPq() {
        Pair tenBit = RtHdrSwapchainFormat.selectPq(List.of(
                new Pair(FMT_R16G16B16A16_SFLOAT, PQ),
                new Pair(FMT_A2R10G10B10_UNORM_PACK32, PQ)));
        assertEquals(FMT_A2R10G10B10_UNORM_PACK32, tenBit.format());

        Pair alternateTenBit = RtHdrSwapchainFormat.selectPq(List.of(
                new Pair(FMT_A2B10G10R10_UNORM_PACK32, PQ)));
        assertEquals(FMT_A2B10G10R10_UNORM_PACK32, alternateTenBit.format());

        Pair floatWhenNoTenBit = RtHdrSwapchainFormat.selectPq(List.of(
                new Pair(FMT_R16G16B16A16_SFLOAT, PQ)));
        assertEquals(FMT_R16G16B16A16_SFLOAT, floatWhenNoTenBit.format());

        // An _SRGB format would make the presentation engine sRGB-encode PQ code values on scanout.
        assertNull(RtHdrSwapchainFormat.selectPq(List.of(new Pair(FMT_B8G8R8A8_SRGB, PQ))));
        assertNull(RtHdrSwapchainFormat.selectPq(SDR_ONLY));
    }

    @Test
    void keepsVanillaFormatWhenTheSurfaceAdvertisesNothingUsable() {
        RtHdrSwapchainFormat format = new RtHdrSwapchainFormat();
        format.observe(List.of());

        Decision decision = format.decide(true);
        assertEquals(0, decision.format(), "no usable pair means the caller keeps Minecraft's own choice");
        assertFalse(decision.pq());
        assertFalse(format.pqAdvertised());
    }
}
