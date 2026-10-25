package dev.comfyfluffy.caustica.rt.clouds;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class RtCloudNoiseTest {
    @Test
    void fieldsTileAtTheirPeriod() {
        double[][] points = {
            {0.13, 0.27, 0.41},
            {0.5, 0.5, 0.5},
            {0.91, 0.07, 0.63},
        };
        for (double[] point : points) {
            int period = 8;
            assertEquals(RtCloudNoise.perlin(point[0], point[1], point[2], period, 7),
                    RtCloudNoise.perlin(point[0] + period, point[1], point[2], period, 7), 1.0e-9);
            assertEquals(RtCloudNoise.perlin(point[0], point[1], point[2], period, 7),
                    RtCloudNoise.perlin(point[0], point[1] + period, point[2], period, 7), 1.0e-9);
            assertEquals(RtCloudNoise.worley(point[0], point[1], point[2], period, 11),
                    RtCloudNoise.worley(point[0] + period, point[1], point[2], period, 11), 1.0e-9);
            assertEquals(RtCloudNoise.worley(point[0], point[1], point[2], period, 11),
                    RtCloudNoise.worley(point[0], point[1], point[2] + period, period, 11), 1.0e-9);
            assertEquals(RtCloudNoise.valueNoise2DPeriodic(point[0], point[1], period, 7),
                    RtCloudNoise.valueNoise2DPeriodic(point[0] + period, point[1], period, 7),
                    1.0e-9);
            assertEquals(RtCloudNoise.worleyNoise2DPeriodic(point[0], point[1], period, 11),
                    RtCloudNoise.worleyNoise2DPeriodic(point[0], point[1] + period, period, 11),
                    1.0e-9);
        }
    }

    @Test
    void shapeChannelsArePopulated() {
        int[] words = RtCloudNoise.buildShape();
        assertEquals(RtCloudNoise.SHAPE_WORDS, words.length);
        assertPopulated(words, 4, "shape");
    }

    @Test
    void detailChannelsArePopulated() {
        int[] words = RtCloudNoise.buildDetail();
        assertEquals(RtCloudNoise.DETAIL_WORDS, words.length);
        // Billows, wisps, and the two low-frequency distortion channels all carry structure.
        assertPopulated(words, 4, "detail");
    }

    @Test
    void placementChannelsArePopulated() {
        int[] words = RtCloudNoise.buildPlacement();
        assertEquals(RtCloudNoise.PLACEMENT_WORDS, words.length);
        // Cluster, top jitter and base lift are read; alpha is a marker.
        assertPopulated(words, 3, "placement");
    }

    @Test
    void weatherIsPopulated() {
        int[] words = RtCloudNoise.buildWeather();
        assertEquals(RtCloudNoise.WEATHER_WORDS, words.length);
        assertPopulated(words, 1, "weather");
    }

    @Test
    void nvdfBakesSignedDistanceField() {
        int[] placement = RtCloudNoise.buildPlacement();
        long started = System.nanoTime();
        float[] sdf = RtCloudNoise.buildNvdf(placement, 320.0f);
        long millis = (System.nanoTime() - started) / 1_000_000L;
        System.out.println("Cloud body SDF bake (128x32x128): " + millis + " ms");

        assertEquals(RtCloudNoise.NVDF_WORDS, sdf.length);
        assertEquals(RtCloudNoise.TOTAL_WORDS,
                RtCloudNoise.NVDF_BASE_WORDS + RtCloudNoise.NVDF_WORDS);

        int inside = 0;
        int outside = 0;
        int empty = 0;
        int liveAtDefaultCoverage = 0;
        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;
        for (float value : sdf) {
            assertTrue(Float.isFinite(value), "SDF contains a non-finite value: " + value);
            min = Math.min(min, value);
            max = Math.max(max, value);
            if (value < 0.0f) {
                inside++;
            } else if (value >= 999.0f) {
                empty++;
            } else {
                outside++;
            }
            if (value < -(0.3f - RtCloudNoise.NVDF_NOMINAL_COVERAGE)
                    * RtCloudNoise.NVDF_COVERAGE_OFFSET_BLOCKS) {
                liveAtDefaultCoverage++;
            }
        }
        System.out.println("Cloud body SDF: inside=" + inside + " outside=" + outside
                + " empty=" + empty + " live@0.3=" + liveAtDefaultCoverage
                + " range=[" + min + ", " + max + "]");
        assertTrue(inside > 0, "SDF has no occupied voxels");
        assertTrue(outside > 0, "SDF has no empty voxels near the body");
        assertTrue(min < 0.0f && max > 0.0f, "SDF is one-signed: [" + min + ", " + max + "]");
        assertTrue(max <= 1000.0f, "SDF exceeds the empty-air sentinel: " + max);
        // The body model is baked at coverage 0.5; a few percent either way is plausible, but an
        // all-empty or all-solid field means the placement tap or the height cuts are wrong.
        assertTrue(inside > sdf.length / 100, "SDF occupation is implausibly low: " + inside);
        assertTrue(inside < sdf.length * 9 / 10, "SDF occupation is implausibly high: " + inside);
        assertTrue(empty < sdf.length, "SDF is entirely empty air");
        // At the default live coverage (0.3) the level-set shift must still leave real bodies, or the
        // feature would render as clear sky at its default settings.
        assertTrue(liveAtDefaultCoverage > sdf.length / 10,
                "SDF has too little cloud at live coverage 0.3: " + liveAtDefaultCoverage);
    }

    private static void assertPopulated(int[] words, int channels, String name) {
        for (int channel = 0; channel < channels; channel++) {
            long sum = 0L;
            long sumSquares = 0L;
            int min = 255;
            int max = 0;
            for (int word : words) {
                int value = (word >>> (channel * 8)) & 0xff;
                sum += value;
                sumSquares += (long) value * value;
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
            double mean = (double) sum / words.length;
            double variance = (double) sumSquares / words.length - mean * mean;
            assertTrue(variance > 64.0, name + " channel " + channel + " is nearly constant: var=" + variance);
            assertTrue(min < 200 && max > 55,
                    name + " channel " + channel + " does not span its range: [" + min + ", " + max + "]");
            assertTrue(mean > 10.0 && mean < 245.0,
                    name + " channel " + channel + " has an extreme mean: " + mean);
        }
    }
}
