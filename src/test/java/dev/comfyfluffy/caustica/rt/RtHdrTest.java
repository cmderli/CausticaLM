package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the HDR output contract: the SMPTE ST 2086 metadata the swapchain carries, the ACES HDR output LUTs
 * the display pass samples, and the PQ/BT.2020 constants the present shaders implement. A mismatch in any of
 * them shows up as wrong colors on an HDR display without any runtime error, so each one is asserted here.
 */
final class RtHdrTest {
    private static final float EPSILON = 0.000001f;
    // SMPTE ST 2084 (PQ) constants; the same numbers must appear in the shaders and in this decoder.
    private static final double PQ_M1 = 0.1593017578125;
    private static final double PQ_M2 = 78.84375;
    private static final double PQ_C1 = 0.8359375;
    private static final double PQ_C2 = 18.8515625;
    private static final double PQ_C3 = 18.6875;
    private static final List<Integer> BAKED_PEAKS = List.of(500, 1000, 2000, 4000);

    @Test
    void buildsRec2020D65MetadataAtTheSelectedAcesMasteringPeak() {
        RtHdr.MasteringMetadata metadata = RtHdr.masteringMetadata(1000);

        assertChromaticity(metadata.red(), 0.708f, 0.292f);
        assertChromaticity(metadata.green(), 0.170f, 0.797f);
        assertChromaticity(metadata.blue(), 0.131f, 0.046f);
        assertChromaticity(metadata.white(), 0.3127f, 0.3290f);
        assertEquals(1000.0f, metadata.maxLuminance(), EPSILON);
        assertEquals(0.0001f, metadata.minLuminance(), EPSILON);
        assertEquals(1000.0f, metadata.maxContentLightLevel(), EPSILON);
        assertEquals(0.0f, metadata.maxFrameAverageLightLevel(), EPSILON);
    }

    @Test
    void rejectsAnInvalidMasteringPeak() {
        assertThrows(IllegalArgumentException.class, () -> RtHdr.masteringMetadata(0));
    }

    /**
     * The mastering metadata must describe the content the display pass actually produces: each baked ACES
     * HDR LUT is capped at its own mastering peak, so MaxCLL/mastering peak are truthful for it, and its
     * maxCLL claim changes with the selected peak exactly as {@link RtHdr#masteringMetadata} does.
     */
    @Test
    void everyBakedHdrLutIsCappedAtItsMasteringPeak() throws IOException {
        for (int peak : BAKED_PEAKS) {
            Lut lut = readLut("/caustica/color/luts/hdr_aces2_rec2020_" + peak + "nit.bin");
            double brightest = 0.0;
            for (int z = 0; z < lut.size; z += 8) {
                for (int y = 0; y < lut.size; y += 8) {
                    for (int x = 0; x < lut.size; x += 8) {
                        brightest = Math.max(brightest, pqDecode(lut.texel(x, y, z)[0]));
                    }
                }
            }
            RtHdr.MasteringMetadata metadata = RtHdr.masteringMetadata(peak);
            assertEquals(peak, metadata.maxLuminance(), EPSILON);
            assertEquals(peak, metadata.maxContentLightLevel(), EPSILON);
            assertTrue(brightest <= peak * 1.01,
                    peak + " nit LUT decodes to " + brightest + " nits, above its metadata peak");
            assertTrue(brightest >= peak * 0.9,
                    peak + " nit LUT decodes to " + brightest + " nits, well below its metadata peak");
        }
    }

    /** The ACES HDR output transforms place diffuse white (scene-linear 1.0) in a sane, peak-scaled band. */
    @Test
    void diffuseWhiteSitsInTheAcesHdrRangeAndGrowsWithTheMasteringPeak() throws IOException {
        double previous = 0.0;
        for (int peak : BAKED_PEAKS) {
            Lut lut = readLut("/caustica/color/luts/hdr_aces2_rec2020_" + peak + "nit.bin");
            int index = lut.shaperIndex(1.0);
            double white = pqDecode(lut.texel(index, index, index)[0]);
            assertTrue(white > 60.0 && white < 160.0,
                    peak + " nit LUT puts diffuse white at " + white + " nits");
            assertTrue(white > previous, "diffuse white must grow with the mastering peak");
            previous = white;
        }
    }

    /**
     * The UI compositing and SDR-to-PQ passes convert sRGB/BT.709 graphics to the HDR10 container. Both use
     * the standard D65 BT.709 -> BT.2020 matrix and the ST 2084 encode; a typo in either is a visible color
     * error (wrong saturation in the UI) with no other symptom.
     */
    @Test
    void presentShadersUseTheStandardBt709ToBt2020MatrixAndSt2084Encode() throws IOException {
        for (String shader : List.of("shaders/pipelines/hdr_composite/main.comp.slang",
                "shaders/pipelines/sdr_present/main.comp.slang")) {
            String source = readRepoFile(shader);
            assertBt709ToBt2020(matrixConstants(source));
            assertShaderConstant(source, "PQ_M1", PQ_M1);
            assertShaderConstant(source, "PQ_M2", PQ_M2);
            assertShaderConstant(source, "PQ_C1", PQ_C1);
            assertShaderConstant(source, "PQ_C2", PQ_C2);
            assertShaderConstant(source, "PQ_C3", PQ_C3);
            assertTrue(source.contains("pc.uiNits"), shader + ": paper white must come from the UI_NITS setting");
        }
    }

    /** The world display pass hands the swapchain PQ code values, so it must not re-encode them. */
    @Test
    void displayShaderPassesTheHdrLutThroughWithoutReencoding() throws IOException {
        String source = readRepoFile("shaders/pipelines/display/main.comp.slang");
        assertShaderConstant(source, "PQ_M1", PQ_M1);
        assertShaderConstant(source, "PQ_M2", PQ_M2);
        assertTrue(source.contains("hdrToneLut.SampleLevel"), "the HDR branch must sample the baked PQ LUT");
    }

    private static void assertBt709ToBt2020(double[] matrix) {
        assertNotNull(matrix, "BT709_TO_BT2020 matrix not found");
        assertEquals(9, matrix.length, "BT709_TO_BT2020 must have nine entries");
        // Bradford-adapted D65 BT.709 -> BT.2020 primaries conversion, row by row.
        double[] expected = {
                0.6274039, 0.3292830, 0.0433131,
                0.0690973, 0.9195406, 0.0113612,
                0.0163916, 0.0880132, 0.8955953,
        };
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], matrix[i], 5.0e-5,
                    "BT709_TO_BT2020[" + (i / 3) + "][" + (i % 3) + "]");
        }
        // Each output channel is a weighted sum of all three inputs, so unit row sums are what make D65
        // white (1,1,1) stay white through the primaries change.
        for (int row = 0; row < 3; row++) {
            double sum = matrix[row * 3] + matrix[row * 3 + 1] + matrix[row * 3 + 2];
            assertEquals(1.0, sum, 1.0e-5, "BT709_TO_BT2020 row " + row + " must map white to white");
        }
    }

    private static void assertShaderConstant(String source, String name, double expected) {
        Matcher matcher = Pattern.compile(
                "static\\s+const\\s+float\\s+" + name + "\\s*=\\s*([0-9.]+)").matcher(source);
        assertTrue(matcher.find(), name + " not found in the shader");
        assertEquals(expected, Double.parseDouble(matcher.group(1)), 1.0e-9, name + " in the shader");
    }

    private static double[] matrixConstants(String source) {
        Matcher matcher = Pattern.compile(
                "BT709_TO_BT2020\\s*=\\s*float3x3\\(([^)]*)\\)", Pattern.DOTALL).matcher(source);
        if (!matcher.find()) {
            return null;
        }
        String[] fields = matcher.group(1).trim().split("[,\\s]+");
        double[] values = new double[fields.length];
        for (int i = 0; i < fields.length; i++) {
            values[i] = Double.parseDouble(fields[i]);
        }
        return values;
    }

    private static String readRepoFile(String relativePath) throws IOException {
        Path path = Path.of(relativePath);
        Assumptions.assumeTrue(Files.isRegularFile(path), "run from the repository root: " + relativePath);
        return Files.readString(path);
    }

    private static void assertChromaticity(RtHdr.Chromaticity actual, float x, float y) {
        assertEquals(x, actual.x(), EPSILON);
        assertEquals(y, actual.y(), EPSILON);
    }

    /** Minimal reader for the "CLUT" container tools/bake_display_lut.py writes. */
    private record Lut(int size, float loStops, float hiStops, ByteBuffer texels) {
        static Lut read(InputStream stream) throws IOException {
            byte[] bytes = stream.readAllBytes();
            ByteBuffer data = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            assertEquals(0x54554C43, data.getInt(0), "CLUT magic");
            assertEquals(1, data.getInt(4), "CLUT version");
            int size = data.getInt(8);
            float lo = data.getFloat(12);
            float hi = data.getFloat(16);
            assertEquals(20 + (long) size * size * size * 8, bytes.length, "CLUT payload size");
            return new Lut(size, lo, hi, data);
        }

        float[] texel(int x, int y, int z) {
            int offset = 20 + ((z * size + y) * size + x) * 8;
            return new float[] {half(offset), half(offset + 2), half(offset + 4), half(offset + 6)};
        }

        /** Index on the shaper axis holding the given scene-linear value, matching display_common's shaper. */
        int shaperIndex(double linear) {
            double stops = Math.log(linear) / Math.log(2.0);
            double t = (stops - loStops) / (hiStops - loStops);
            return (int) Math.round(Math.min(Math.max(t, 0.0), 1.0) * (size - 1));
        }

        private float half(int offset) {
            return texels.getShort(offset) == 0 ? 0.0f : Float16.toFloat(texels.getShort(offset));
        }
    }

    private static Lut readLut(String resource) throws IOException {
        try (InputStream stream = RtHdrTest.class.getResourceAsStream(resource)) {
            assertNotNull(stream, "missing LUT resource " + resource);
            return Lut.read(stream);
        }
    }

    private static double pqDecode(double code) {
        double p = Math.pow(Math.max(code, 0.0), 1.0 / PQ_M2);
        double numerator = Math.max(p - PQ_C1, 0.0);
        double denominator = Math.max(PQ_C2 - PQ_C3 * p, 1.0e-12);
        return 10000.0 * Math.pow(numerator / denominator, 1.0 / PQ_M1);
    }

    /** IEEE 754 binary16 -> float, enough for the LUT's rgba16f payload. */
    private static final class Float16 {
        static float toFloat(short bits) {
            int sign = (bits >>> 15) & 0x1;
            int exponent = (bits >>> 10) & 0x1F;
            int mantissa = bits & 0x3FF;
            if (exponent == 0) {
                return (float) ((sign == 0 ? 1 : -1) * Math.scalb(mantissa, -24));
            }
            if (exponent == 0x1F) {
                return mantissa == 0 ? (sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY)
                        : Float.NaN;
            }
            return (float) ((sign == 0 ? 1 : -1) * Math.scalb(mantissa + 1024, exponent - 25));
        }
    }
}
