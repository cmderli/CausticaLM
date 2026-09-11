package dev.comfyfluffy.caustica.rt;

import java.util.List;

/**
 * Chooses the (format, colorSpace) pair the window swapchain presents with, and keeps that choice stable while
 * HDR stays enabled.
 *
 * <p>The pair list a Vulkan surface advertises describes the display stack, not the swapchain being replaced.
 * Deriving the choice from it on every configure lets a driver or compositor that reports the PQ pair only in
 * one of the two states flip the decision on each recreation: the flip makes the presentation engine mark the
 * fresh swapchain suboptimal, that recreates it again, and the swapchain oscillates between HDR and SDR. The
 * discovered pair is therefore cached on first sight and reused; only a rejected creation clears it (see
 * {@link #rejectPq()}), which is the one signal that the presentation stack genuinely cannot present it.
 *
 * <p>Pure host state: it holds no Vulkan objects, so the policy is unit-testable.
 */
public final class RtHdrSwapchainFormat {
    /** {@code VK_COLOR_SPACE_HDR10_ST2084_EXT}. */
    public static final int COLOR_SPACE_HDR10_ST2084 = 1000104008;
    /** {@code VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}. */
    public static final int COLOR_SPACE_SRGB_NONLINEAR = 0;
    /** {@code VK_FORMAT_A2R10G10B10_UNORM_PACK32} / {@code VK_FORMAT_A2B10G10R10_UNORM_PACK32}. */
    private static final int FORMAT_A2R10G10B10_UNORM_PACK32 = 64;
    private static final int FORMAT_A2B10G10R10_UNORM_PACK32 = 58;
    /** {@code VK_FORMAT_R16G16B16A16_SFLOAT}. */
    private static final int FORMAT_R16G16B16A16_SFLOAT = 97;
    /** {@code VK_FORMAT_R8G8B8A8_SRGB} / {@code VK_FORMAT_B8G8R8A8_SRGB}. */
    private static final int FORMAT_R8G8B8A8_SRGB = 43;
    private static final int FORMAT_B8G8R8A8_SRGB = 50;
    /** {@code VK_FORMAT_R8G8B8A8_UNORM} / {@code VK_FORMAT_B8G8R8A8_UNORM}: vanilla's SDR pairs. */
    private static final int FORMAT_R8G8B8A8_UNORM = 37;
    private static final int FORMAT_B8G8R8A8_UNORM = 44;

    /** One advertised surface (format, colorSpace) pair. */
    public record Pair(int format, int colorSpace) {
    }

    /**
     * The chosen presentation pair. {@code format == 0} means "nothing usable was advertised": the caller
     * keeps whatever Minecraft selected.
     */
    public record Decision(int format, int colorSpace, boolean pq) {
        static final Decision KEEP_VANILLA = new Decision(0, COLOR_SPACE_SRGB_NONLINEAR, false);
    }

    // 0 = not discovered yet, -1 = this surface advertises none, otherwise the cached format.
    private int pqFormat;
    private int pqColorSpace;
    private int sdrFormat;
    private int sdrColorSpace;

    /**
     * Caches the best PQ pair and the native SDR pair from an enumeration. A PQ pair that was already found
     * is kept: later enumerations are not allowed to un-discover it, which is what stops the oscillation.
     */
    public void observe(List<Pair> advertised) {
        if (pqFormat == 0) {
            Pair best = selectPq(advertised);
            pqFormat = best != null ? best.format() : -1;
            pqColorSpace = best != null ? best.colorSpace() : 0;
        }
        if (sdrFormat == 0) {
            Pair sdr = selectSdr(advertised);
            sdrFormat = sdr != null ? sdr.format() : 0;
            sdrColorSpace = sdr != null ? sdr.colorSpace() : 0;
        }
    }

    /** 0 while nothing has been enumerated, -1 when the surface advertised no PQ pair, else its format. */
    public int pqFormat() {
        return pqFormat;
    }

    /** 0 until a native SDR pair has been seen, else its format. */
    public int sdrFormat() {
        return sdrFormat;
    }

    /** Whether this surface has ever advertised a PQ pair (gates the HDR options in the video screen). */
    public boolean pqAdvertised() {
        return pqFormat > 0;
    }

    /**
     * The pair to create the swapchain with: the cached PQ pair while HDR is enabled and one was seen, the
     * cached native SDR pair otherwise.
     */
    public Decision decide(boolean hdrEnabled) {
        if (hdrEnabled && pqFormat > 0) {
            return new Decision(pqFormat, pqColorSpace, true);
        }
        if (sdrFormat != 0) {
            return new Decision(sdrFormat, sdrColorSpace, false);
        }
        return Decision.KEEP_VANILLA;
    }

    /**
     * Records that the presentation stack refused a PQ swapchain. PQ is then unavailable for this surface, so
     * the next decision is native SDR instead of retrying the same rejected pair.
     */
    public void rejectPq() {
        pqFormat = -1;
        pqColorSpace = 0;
    }

    /**
     * The surface's best PQ pair. A 10-bit UNORM pack32 comes first (PQ is a 10-bit curve and every HDR10
     * swapchain carries one), then 16-bit float, then anything else. {@code *_SRGB} formats are skipped
     * outright: the presentation engine would treat the PQ code values this renderer writes as linear and
     * sRGB-encode them on scanout, which is the wrong transfer function rather than a precision loss.
     */
    public static Pair selectPq(List<Pair> advertised) {
        Pair best = null;
        int bestRank = 0;
        for (Pair pair : advertised) {
            if (pair.colorSpace() != COLOR_SPACE_HDR10_ST2084) {
                continue;
            }
            int rank = switch (pair.format()) {
                case FORMAT_A2R10G10B10_UNORM_PACK32, FORMAT_A2B10G10R10_UNORM_PACK32 -> 3;
                case FORMAT_R16G16B16A16_SFLOAT -> 2;
                case FORMAT_R8G8B8A8_SRGB, FORMAT_B8G8R8A8_SRGB -> 0;
                default -> 1;
            };
            if (rank > bestRank) {
                bestRank = rank;
                best = pair;
            }
        }
        return best;
    }

    /** Vanilla's native SDR pair: linear sRGB color space with an 8-bit UNORM format. */
    public static Pair selectSdr(List<Pair> advertised) {
        for (Pair pair : advertised) {
            if (pair.colorSpace() == COLOR_SPACE_SRGB_NONLINEAR
                    && (pair.format() == FORMAT_R8G8B8A8_UNORM || pair.format() == FORMAT_B8G8R8A8_UNORM)) {
                return pair;
            }
        }
        return null;
    }
}
