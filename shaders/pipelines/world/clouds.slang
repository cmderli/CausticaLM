// Volumetric cloud layer, traced along every ray that escapes the scene.
//
// Cloud placement, cumulus detail, value erosion and the multiple-scattering key light are adapted
// from NUMOS (https://github.com/Kim2091/numos-sky), Copyright 2026 Kim2091 (Sparkles),
// SPDX-License-Identifier: Apache-2.0. Modified for CausticaLM's block-scale integration; see
// THIRD_PARTY_NOTICES.md and licenses/NUMOS-LICENSE.txt.
//
// The layer is a bounded slab of one cumulus deck. A ray that misses geometry raymarches the slab and
// returns the light it scatters plus the transmittance of the cloud in front of the sky, so clouds are
// part of the path integral rather than a screen-space overlay: they appear in reflections, refractions,
// and every diffuse sky ray that lights the scene.
//
// The model uses a Nubis-style placement map and cumulus turbulence volume, and a CPU-baked signed
// distance field of the cloud body for the density profile. CausticaLM samples it in its existing
// sky-ray integration and bakes a tile-space shadow map for terrain NEE. A separate render-res
// temporal pass filters the resulting sky radiance; the dedicated cloud-only temporal path is not
// part of this module.
//
// Density reads the detail volume and the cloud-body SDF from the BDA buffer published in
// pc.cloudNoiseAddr by RtCloudNoise:
//
//   shape     96^3 RGBA8   r = Perlin-Worley, gba = Worley FBM; reserved in the layout
//   detail   128^3 RGBA8   Nubis turbulence: r billows, g wisps, ba distortion pair
//   weather  256^2 RGBA8   the large-scale moisture/coverage field (r)
//   placement 512^2 RGBA8  per-column cloud model consumed by the SDF bake
//   sdf      128x32x128 float  signed distance to the cloud body in blocks, negative inside
//
// All fields tile in world space. The CPU publishes a single world-pinned anchor reduced mod 4096
// in WorldPush.cloudParams1, together with the wind drift; 4096 is a multiple of the detail, weather
// and placement periods, so the anchor keeps every horizontal field fixed while the terrain rebase
// moves the traced coordinate origin. Vertically the fields are sampled relative to the cloud base,
// at a world-space scale independent of the configured thickness.
//
// The layer is illuminated by the dominant celestial body through a short light march and by the sky
// dome as ambient. The same density field is exposed as cloudShadowTransmittance() for the sun/moon NEE
// at shaded surfaces, so the shadows a cloud casts and the cloud a ray sees come from one density
// function and cannot disagree.
module clouds;

import world_common;
import sky;
import bindings;

// Keep in lock-step with RtCloudNoise.java: the base offsets are word indices into the same buffer.
public static const int CLOUD_SHAPE_DIM = 96;
public static const int CLOUD_DETAIL_DIM = 128;
public static const int CLOUD_WEATHER_DIM = 256;
public static const int CLOUD_PLACEMENT_DIM = 512;
public static const uint CLOUD_DETAIL_BASE =
        uint(CLOUD_SHAPE_DIM * CLOUD_SHAPE_DIM * CLOUD_SHAPE_DIM);
public static const uint CLOUD_WEATHER_BASE =
        CLOUD_DETAIL_BASE + uint(CLOUD_DETAIL_DIM * CLOUD_DETAIL_DIM * CLOUD_DETAIL_DIM);
public static const uint CLOUD_PLACEMENT_BASE =
        CLOUD_WEATHER_BASE + uint(CLOUD_WEATHER_DIM * CLOUD_WEATHER_DIM);
// Cloud-body SDF, one float word per voxel after the packed noise fields. X and Z span the placement
// tile; Y spans the configured cloud slab.
public static const int CLOUD_NVDF_X = 128;
public static const int CLOUD_NVDF_Y = 32;
public static const int CLOUD_NVDF_Z = 128;
public static const uint CLOUD_NVDF_BASE =
        CLOUD_PLACEMENT_BASE + uint(CLOUD_PLACEMENT_DIM * CLOUD_PLACEMENT_DIM);
// Bake-time constants: the body was baked at this coverage, and the live coverage shifts the
// iso-level by this many blocks per unit of coverage away from it.
public static const float CLOUD_NVDF_NOMINAL_COVERAGE = 0.5;
public static const float CLOUD_NVDF_COVERAGE_OFFSET = 160.0;
public static const float CLOUD_NVDF_PROFILE_DEPTH = 40.0;
// The shadow-map bake covers the same tile as the placement/SDF fields.
public static const int CLOUD_SHADOW_DIM = 256;

// World-space period of each field. The CPU anchor is reduced mod 4096, which is a multiple of every
// period below, so one anchor keeps all three world-locked across a terrain rebase.
public static const float CLOUD_DETAIL_TILE = 256.0;
public static const float CLOUD_WEATHER_TILE = 4096.0;
// The SDF spans one placement tile horizontally, so its horizontal period is the same 2048 blocks.
public static const float CLOUD_PLACEMENT_TILE = 2048.0;
// Vertical world-space period of the detail volume. Making it comparable to the horizontal period
// keeps the 3D noise from stretching into horizontal sheets.
public static const float CLOUD_DETAIL_Y_TILE = 256.0;

// Extinction per block at density 1. At the default ~320-block thickness a full-density core is still
// optically thick, but the default density multiplier keeps edges translucent and lets direct light
// reach well into the deck, which is what makes the volume readable instead of a lit shell.
public static const float CLOUD_EXTINCTION = 0.04;
public static const int CLOUD_LIGHT_STEPS = 6;
public static const int CLOUD_SHADOW_STEPS = 6;
// Wrenninge multiple-scattering octaves, following NUMOS' cloud defaults: each order is softer
// (phase anisotropy c^i), contributes less (b^i) and is absorbed more slowly (a^i).
public static const int CLOUD_SUN_OCTAVES = 4;
public static const float CLOUD_SUN_OCTAVE_A = 0.45;
public static const float CLOUD_SUN_OCTAVE_B = 0.55;
public static const float CLOUD_SUN_OCTAVE_C = 0.5;
public static const float CLOUD_SUN_PHASE_G = 0.8;
// Clouds are ignored past this distance. A near-horizontal ray can otherwise cross the slab for
// thousands of blocks, and the layer would turn into a bright band at the horizon; the fade below
// dissolves it into the sky before the cap is reached.
public static const float CLOUD_MAX_DISTANCE = 9000.0;
public static const float CLOUD_FADE_START = 5200.0;
public static const float CLOUD_AMBIENT_WEIGHT = 0.72;

public struct CloudState {
    public bool enabled;
    public int steps;
    public float3 lightDir;          // dominant celestial body, world space
    public float3 lightIlluminance;  // lux at the deck, already through the atmosphere, linear BT.709
    public float3 ambientRadiance;   // cd/m² from the zenith sky, linear BT.709
    public float3 horizonSunward;    // cd/m² just above the horizon on the sun's side
    public float3 horizonAntiSunward; // cd/m² just above the horizon on the far side
    public float baseY;              // trace-space blocks (sea level + altitude - rebase origin)
    public float thickness;          // blocks
    public float coverage;           // 0..1
    public float densityScale;
    public float seaLevelY;          // trace-space blocks; with baseY it recovers the deck's altitude
    public float2 anchor;            // world-pinned noise origin mod 4096, including wind drift
};

// Directional scattering. Two Henyey-Greenstein lobes: a strong forward lobe for the silver lining and
// a weak back lobe that keeps the lit side of the cloud from going flat. Both lobes integrate to one over
// the sphere, so the weights below preserve the phase function's normalisation.
public float cloudHenyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * SKY_PI * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1.0e-4), 1.5));
}

// ---- Noise sampling ------------------------------------------------------------------------------
//
// The noise volumes are RGBA8 words and the SDF is one float word per voxel; all are read through
// pc.cloudNoiseAddr as data rather than bound as samplers. Sampling interpolates manually, which for
// the detail volume is cheaper than the image plumbing would be.

public uint cloudNoiseIndex(uint3 cell, uint dim, uint base) {
    return base + (cell.z * dim + cell.y) * dim + cell.x;
}

public float4 cloudUnpackRgba8(uint word) {
    return float4(float(word & 0xffu), float((word >> 8u) & 0xffu),
                  float((word >> 16u) & 0xffu), float(word >> 24u)) * (1.0 / 255.0);
}

// Trilinear interpolation of the packed words. Positions are in tile units; `dim` is the
// field's resolution and the lattice wraps at it, which is what makes the fields tileable.
public float4 cloudSampleNoise(float3 tilePosition, uint base, uint dim) {
    float3 uv = frac(tilePosition) * float(dim);
    // frac() is in [0,1), but the multiply can round up to exactly `dim`; clamping the cell keeps the
    // corner indices in range, and the wrap below then only has to deal with the single top edge.
    int3 floorCell = min(int3(floor(uv)), int3(int(dim) - 1));
    float3 f = uv - float3(floorCell);
    uint3 cell = uint3(floorCell);
    uint3 next = cell + uint3(1u);
    next.x = next.x >= dim ? 0u : next.x;
    next.y = next.y >= dim ? 0u : next.y;
    next.z = next.z >= dim ? 0u : next.z;

    ConstPtr<uint> words = ConstPtr<uint>(pc.cloudNoiseAddr);
    float4 n000 = cloudUnpackRgba8(words[cloudNoiseIndex(uint3(cell.x, cell.y, cell.z), dim, base)]);
    float4 n100 = cloudUnpackRgba8(words[cloudNoiseIndex(uint3(next.x, cell.y, cell.z), dim, base)]);
    float4 n010 = cloudUnpackRgba8(words[cloudNoiseIndex(uint3(cell.x, next.y, cell.z), dim, base)]);
    float4 n110 = cloudUnpackRgba8(words[cloudNoiseIndex(uint3(next.x, next.y, cell.z), dim, base)]);
    float4 n001 = cloudUnpackRgba8(words[cloudNoiseIndex(uint3(cell.x, cell.y, next.z), dim, base)]);
    float4 n101 = cloudUnpackRgba8(words[cloudNoiseIndex(uint3(next.x, cell.y, next.z), dim, base)]);
    float4 n011 = cloudUnpackRgba8(words[cloudNoiseIndex(uint3(cell.x, next.y, next.z), dim, base)]);
    float4 n111 = cloudUnpackRgba8(words[cloudNoiseIndex(uint3(next.x, next.y, next.z), dim, base)]);

    float4 n00 = lerp(n000, n100, f.x);
    float4 n10 = lerp(n010, n110, f.x);
    float4 n01 = lerp(n001, n101, f.x);
    float4 n11 = lerp(n011, n111, f.x);
    return lerp(lerp(n00, n10, f.y), lerp(n01, n11, f.y), f.z);
}

// The weather field stores its scalar in red; the other channels are written as copies so the words stay
// legible in a dump without changing the sampler.
public float cloudSampleWeather(float2 tilePosition) {
    uint dim = uint(CLOUD_WEATHER_DIM);
    float2 uv = frac(tilePosition) * float(dim);
    int2 floorCell = min(int2(floor(uv)), int2(int(dim) - 1));
    float2 f = uv - float2(floorCell);
    uint2 cell = uint2(floorCell);
    uint2 next = cell + uint2(1u);
    next.x = next.x >= dim ? 0u : next.x;
    next.y = next.y >= dim ? 0u : next.y;

    ConstPtr<uint> words = ConstPtr<uint>(pc.cloudNoiseAddr);
    float4 w00 = cloudUnpackRgba8(words[CLOUD_WEATHER_BASE + cell.y * dim + cell.x]);
    float4 w10 = cloudUnpackRgba8(words[CLOUD_WEATHER_BASE + cell.y * dim + next.x]);
    float4 w01 = cloudUnpackRgba8(words[CLOUD_WEATHER_BASE + next.y * dim + cell.x]);
    float4 w11 = cloudUnpackRgba8(words[CLOUD_WEATHER_BASE + next.y * dim + next.x]);
    return lerp(lerp(w00.r, w10.r, f.x), lerp(w01.r, w11.r, f.x), f.y);
}

// Cloud-body SDF from the CPU bake, in blocks and positive outside. The buffer is data, not a
// sampler, so the lookup interpolates manually with texel-centre mapping: texel i sits at
// (i + 0.5) / dim, so a tile position maps to dim * position - 0.5. X and Z wrap over the tile, Y
// clamps because the slab has no vertical period.
public int cloudNvdfWrap(int value, int size) {
    int wrapped = value % size;
    return wrapped < 0 ? wrapped + size : wrapped;
}

public uint cloudNvdfIndex(int3 cell) {
    return CLOUD_NVDF_BASE
            + (uint(cell.z) * uint(CLOUD_NVDF_Y) + uint(cell.y)) * uint(CLOUD_NVDF_X)
            + uint(cell.x);
}

public float cloudSampleNvdf(float2 tilePosition, float height) {
    float3 uv = float3(frac(tilePosition.x) * float(CLOUD_NVDF_X) - 0.5,
                       height * float(CLOUD_NVDF_Y) - 0.5,
                       frac(tilePosition.y) * float(CLOUD_NVDF_Z) - 0.5);
    int3 cell0 = int3(floor(uv));
    int3 cell1 = cell0 + int3(1);
    float3 f = uv - float3(cell0);
    int3 baseCell = int3(cloudNvdfWrap(cell0.x, CLOUD_NVDF_X),
                         clamp(cell0.y, 0, CLOUD_NVDF_Y - 1),
                         cloudNvdfWrap(cell0.z, CLOUD_NVDF_Z));
    int3 nextCell = int3(cloudNvdfWrap(cell1.x, CLOUD_NVDF_X),
                         clamp(cell1.y, 0, CLOUD_NVDF_Y - 1),
                         cloudNvdfWrap(cell1.z, CLOUD_NVDF_Z));

    ConstPtr<uint> words = ConstPtr<uint>(pc.cloudNoiseAddr);
    float n000 = asfloat(words[cloudNvdfIndex(int3(baseCell.x, baseCell.y, baseCell.z))]);
    float n100 = asfloat(words[cloudNvdfIndex(int3(nextCell.x, baseCell.y, baseCell.z))]);
    float n010 = asfloat(words[cloudNvdfIndex(int3(baseCell.x, nextCell.y, baseCell.z))]);
    float n110 = asfloat(words[cloudNvdfIndex(int3(nextCell.x, nextCell.y, baseCell.z))]);
    float n001 = asfloat(words[cloudNvdfIndex(int3(baseCell.x, baseCell.y, nextCell.z))]);
    float n101 = asfloat(words[cloudNvdfIndex(int3(nextCell.x, baseCell.y, nextCell.z))]);
    float n011 = asfloat(words[cloudNvdfIndex(int3(baseCell.x, nextCell.y, nextCell.z))]);
    float n111 = asfloat(words[cloudNvdfIndex(int3(nextCell.x, nextCell.y, nextCell.z))]);

    float n00 = lerp(n000, n100, f.x);
    float n10 = lerp(n010, n110, f.x);
    float n01 = lerp(n001, n101, f.x);
    float n11 = lerp(n011, n111, f.x);
    return lerp(lerp(n00, n10, f.y), lerp(n01, n11, f.y), f.z);
}

// ---- Density -------------------------------------------------------------------------------------

// One density evaluation. `withDetail` is false for the light and shadow marches: those only need the
// body SDF, and the high-frequency erosion would not change whether the sun reaches the sample.
public float cloudDensityAt(float3 position, CloudState cloud, float weather, bool withDetail) {
    float height = (position.y - cloud.baseY) / cloud.thickness;
    if (height <= 0.0 || height >= 1.0) {
        return 0.0;
    }

    float2 worldXZ = float2(position.x + cloud.anchor.x, position.z + cloud.anchor.y);
    // Body: the baked SDF minus the coverage level-set shift. The bake is at nominal coverage 0.5,
    // so a higher live coverage moves the iso-surface outward and grows every body. Weather scales
    // that local coverage so the deck opens and closes across the sky without a rebake; scaling, not
    // offsetting, keeps the slider's clear and overcast endpoints meaningful.
    float sdf = cloudSampleNvdf(worldXZ * (1.0 / CLOUD_PLACEMENT_TILE), height);
    float liveCoverage = saturate(cloud.coverage * (0.7 + 0.6 * weather));
    float sdfLive = sdf - (liveCoverage - CLOUD_NVDF_NOMINAL_COVERAGE) * CLOUD_NVDF_COVERAGE_OFFSET;
    if (sdfLive >= 0.0) {
        return 0.0;
    }
    // Nubis cubed: the profile ramps from the body surface inward, then the detail erodes it.
    float profile = saturate(-sdfLive / CLOUD_NVDF_PROFILE_DEPTH);

    // Vertical coordinates are measured from the cloud base in blocks, so the world scale of the
    // vertical features does not depend on the configured thickness; a thicker deck simply reveals
    // more of the field. Base and top are world-fixed, so this needs no rebase anchor.
    float relativeAltitude = position.y - cloud.baseY;

    float detailValue = 0.5;
    if (withDetail) {
        float3 detailPosition = float3(
                worldXZ.x * (1.0 / CLOUD_DETAIL_TILE),
                relativeAltitude * (1.0 / CLOUD_DETAIL_Y_TILE),
                worldXZ.y * (1.0 / CLOUD_DETAIL_TILE));
        float4 detail = cloudSampleNoise(detailPosition, CLOUD_DETAIL_BASE,
                                         uint(CLOUD_DETAIL_DIM));
        // Nubis: wisps at the base, billows in the body and near the top. The ba channels warp the
        // sample slightly so the pattern does not read as a repeated volume.
        float billowMix = saturate(smoothstep(0.05, 0.45, height) * 0.75 + 0.15);
        detailValue = lerp(detail.g, detail.r, billowMix);
        float2 warp = (detail.ba - 0.5) * 2.0;
        detailValue = saturate(detailValue + warp.x * 0.06);
    }

    // NUMOS/Nubis value erosion: detail eats into the body, less deeply where the body is already
    // dense. The height density makes the base thinner than the body, so a thick deck reads as a
    // tower rather than a uniformly dense slab.
    float erosion = (1.0 - detailValue) * 0.58;
    float eroded = saturate((profile - erosion) / max(1.0 - erosion, 1.0e-3));
    float heightDensity = clamp(0.25 + 0.75 * height * cloud.thickness / 100.0, 0.25, 1.0);
    return eroded * heightDensity * cloud.densityScale;
}

// Optical depth (density x distance, before CLOUD_EXTINCTION) from a sample to the top of the layer
// along the dominant light. The march stops at a few thicknesses rather than at the true top
// distance: at a grazing sun the geometric distance grows without bound while the cloud is already
// optically thick, and the approximation is what keeps the light path cheap.
public float cloudSunOpticalDepth(float3 position, CloudState cloud, float weather) {
    if (cloud.lightDir.y <= 0.02) {
        return 1.0e3;
    }
    float top = cloud.baseY + cloud.thickness;
    float distanceToTop = max(top - position.y, 0.0) / max(cloud.lightDir.y, 0.02);
    distanceToTop = min(distanceToTop, cloud.thickness * 2.0);
    if (distanceToTop <= 0.0) {
        return 0.0;
    }
    float stepLength = distanceToTop / float(CLOUD_LIGHT_STEPS);
    float opticalDepth = 0.0;
    for (int i = 0; i < CLOUD_LIGHT_STEPS; i++) {
        float3 samplePosition = position + cloud.lightDir * (float(i) + 0.5) * stepLength;
        opticalDepth += cloudDensityAt(samplePosition, cloud, weather, false);
    }
    return opticalDepth * stepLength;
}

// Wrenninge multiple-scattering octave sum for the key light, as in NUMOS: octave i has its phase
// anisotropy scaled by c^i, its contribution by b^i, and its extinction by a^i. The first octave is
// shadowed on the sharp optical depth; the softer higher orders use a blurred optical depth.
public float3 cloudSunOctaves(float cosTheta, float opticalDepth) {
    float3 direct = float3(0.0, 0.0, 0.0);
    float b = 1.0;
    float c = 1.0;
    float a = 1.0;
    for (int i = 0; i < CLOUD_SUN_OCTAVES; i++) {
        float forward = cloudHenyeyGreenstein(cosTheta, c * CLOUD_SUN_PHASE_G);
        float back = cloudHenyeyGreenstein(cosTheta, -c * 0.3);
        float phase = b * lerp(forward, back, 0.3);
        float tau = i == 0 ? opticalDepth : opticalDepth * 0.65;
        direct += phase * exp(-a * tau) * float3(1.0, 1.0, 1.0);
        b *= CLOUD_SUN_OCTAVE_B;
        c *= CLOUD_SUN_OCTAVE_C;
        a *= CLOUD_SUN_OCTAVE_A;
    }
    return direct;
}

// ---- Per-frame state -----------------------------------------------------------------------------

// Zenith sky radiance, the ambient that fills the unlit side of the cloud. Both body slices are summed
// for the same reason the sky dome sums them: each slice is one body's contribution to the same
// atmosphere, not an alternative lighting state.
public float3 cloudAmbientRadiance(SkyState sky) {
    float3 up = float3(0.0, 1.0, 0.0);
    float2 sunUv = skyViewLutUv(sky.viewerRadiusKm, up.y, lightViewCosine(up, sky.sunDir),
            SKY_VIEW_BODY_SUN);
    float2 moonUv = skyViewLutUv(sky.viewerRadiusKm, up.y, lightViewCosine(up, sky.moonDir),
            SKY_VIEW_BODY_MOON);
    float3 radiance = skyViewLut.SampleLevel(sunUv, 0.0).rgb
                    + skyViewLut.SampleLevel(moonUv, 0.0).rgb;
    return radiance + sky.airglowLuminance;
}

public CloudState cloudState(WorldPush push, SkyState sky) {
    CloudState cloud;
    cloud.steps = int(clamp(push.cloudParams1.z, 0.0, 96.0));
    cloud.enabled = cloud.steps >= 4 && pc.cloudNoiseAddr != uint64_t(0);
    cloud.baseY = push.cloudParams0.x;
    cloud.thickness = max(push.cloudParams0.y, 1.0);
    cloud.coverage = clamp(push.cloudParams0.z, 0.0, 1.0);
    cloud.densityScale = max(push.cloudParams0.w, 0.0);
    cloud.seaLevelY = push.cloudParams1.w;
    cloud.anchor = push.cloudParams1.xy;
    if (!cloud.enabled) {
        cloud.lightDir = float3(0.0, 1.0, 0.0);
        cloud.lightIlluminance = float3(0.0, 0.0, 0.0);
        cloud.ambientRadiance = float3(0.0, 0.0, 0.0);
        cloud.horizonSunward = float3(0.0, 0.0, 0.0);
        cloud.horizonAntiSunward = float3(0.0, 0.0, 0.0);
        return cloud;
    }

    CelestialLight light = dominantCelestialLight(sky, transmittanceLut);
    cloud.lightDir = light.dir;
    // The dominant light is attenuated to the viewer by sky.dominantCelestialLight(). The deck sits
    // above the viewer, so the light reaching it is re-derived from the same transmittance table and
    // the two are divided out. Clamped because the ratio is meaningless where the viewer transmittance
    // has already gone to zero at the horizon, while the deck above it can still be in sunlight.
    float cloudAltitudeKm = max((cloud.baseY + cloud.thickness * 0.5 - cloud.seaLevelY) / 100.0,
            MIN_VIEWER_ALTITUDE_KM);
    float3 altitudeTransmittance = sampleTransmittance(transmittanceLut,
            ATMOS_BOTTOM_KM + cloudAltitudeKm, light.dir.y);
    float3 viewerTransmittance = sampleTransmittance(transmittanceLut,
            sky.viewerRadiusKm, light.dir.y);
    float3 altitudeGain = clamp(altitudeTransmittance / max(viewerTransmittance, float3(1.0e-3)),
            float3(0.5), float3(4.0));
    cloud.lightIlluminance = light.illuminance * altitudeGain;
    cloud.ambientRadiance = cloudAmbientRadiance(sky) * CLOUD_AMBIENT_WEIGHT;

    // Horizon bands: NUMOS' ambient integrates the sky ring so the sunward and anti-sunward sides
    // of a cloud take different light. Two directions recover that at sunset without a per-ray ring.
    float3 sunHorizon = normalize(float3(light.dir.x, 0.08, light.dir.z));
    float3 antiHorizon = float3(-sunHorizon.x, 0.08, -sunHorizon.z);
    float2 sunUv = skyViewLutUv(sky.viewerRadiusKm, sunHorizon.y,
            lightViewCosine(sunHorizon, sky.sunDir), SKY_VIEW_BODY_SUN);
    float2 moonUv = skyViewLutUv(sky.viewerRadiusKm, sunHorizon.y,
            lightViewCosine(sunHorizon, sky.moonDir), SKY_VIEW_BODY_MOON);
    cloud.horizonSunward = (skyViewLut.SampleLevel(sunUv, 0.0).rgb
            + skyViewLut.SampleLevel(moonUv, 0.0).rgb) * CLOUD_AMBIENT_WEIGHT;
    sunUv = skyViewLutUv(sky.viewerRadiusKm, antiHorizon.y,
            lightViewCosine(antiHorizon, sky.sunDir), SKY_VIEW_BODY_SUN);
    moonUv = skyViewLutUv(sky.viewerRadiusKm, antiHorizon.y,
            lightViewCosine(antiHorizon, sky.moonDir), SKY_VIEW_BODY_MOON);
    cloud.horizonAntiSunward = (skyViewLut.SampleLevel(sunUv, 0.0).rgb
            + skyViewLut.SampleLevel(moonUv, 0.0).rgb) * CLOUD_AMBIENT_WEIGHT;
    return cloud;
}

// Per-pixel, per-frame offset of the march. The layer is entered at a different point along each step
// every frame; a denoiser integrates that dither into a smooth result, while a fixed offset would leave
// its sample lattice visible as banding.
public float cloudJitter(float3 direction, uint frameIndex) {
    float noise = dot(direction, float3(12.9898, 78.233, 37.719))
            + float(frameIndex) * 0.61803398875;
    return frac(sin(noise) * 43758.5453);
}

// ---- Ray integration -----------------------------------------------------------------------------

public struct CloudResult {
    public float3 scattering;  // radiance scattered toward the ray origin, linear BT.709, cd/m²
    public float transmittance; // fraction of the background sky reaching the origin
};

public CloudResult integrateClouds(float3 rayOrigin, float3 rayDirection, CloudState cloud,
                                   float jitter) {
    CloudResult result;
    result.scattering = float3(0.0, 0.0, 0.0);
    result.transmittance = 1.0;
    if (!cloud.enabled || cloud.coverage <= 0.0 || cloud.densityScale <= 0.0) {
        return result;
    }

    // Slab intersection. The camera can be under, inside, or above the layer, and all three are the same
    // interval test once the interval is clamped to the ray's forward half.
    float top = cloud.baseY + cloud.thickness;
    float tEnter;
    float tExit;
    if (abs(rayDirection.y) < 1.0e-4) {
        if (rayOrigin.y <= cloud.baseY || rayOrigin.y >= top) {
            return result;
        }
        tEnter = 0.0;
        tExit = CLOUD_MAX_DISTANCE;
    } else {
        float lower = (cloud.baseY - rayOrigin.y) / rayDirection.y;
        float upper = (top - rayOrigin.y) / rayDirection.y;
        tEnter = max(min(lower, upper), 0.0);
        tExit = min(max(lower, upper), CLOUD_MAX_DISTANCE);
    }
    if (tExit <= tEnter) {
        return result;
    }

    float3 entry = rayOrigin + rayDirection * tEnter;
    float3 exitPoint = rayOrigin + rayDirection * tExit;
    float entryWeather = cloudSampleWeather(
            float2(entry.x + cloud.anchor.x, entry.z + cloud.anchor.y) * (1.0 / CLOUD_WEATHER_TILE));
    float exitWeather = cloudSampleWeather(
            float2(exitPoint.x + cloud.anchor.x, exitPoint.z + cloud.anchor.y)
                    * (1.0 / CLOUD_WEATHER_TILE));

    int steps = max(cloud.steps, 1);
    float stepLength = (tExit - tEnter) / float(steps);
    float3 position = rayOrigin + rayDirection * (tEnter + stepLength * jitter);
    float cosTheta = dot(rayDirection, cloud.lightDir);
    float topY = cloud.baseY + cloud.thickness;
    float transmittance = 1.0;
    float3 scattering = float3(0.0, 0.0, 0.0);

    for (int i = 0; i < steps; i++) {
        float weather = lerp(entryWeather, exitWeather, (float(i) + jitter) / float(steps));
        float density = cloudDensityAt(position, cloud, weather, true);
        // Dissolve the layer with distance before it is shaded, not after the march: a near-horizontal
        // ray exits the slab thousands of blocks away, but its cloud at 500 blocks must still be drawn.
        // Scaling the step's density fades both its scattering and its occlusion into the sky together.
        float t = tEnter + (float(i) + jitter) * stepLength;
        density *= 1.0 - smoothstep(CLOUD_FADE_START, CLOUD_MAX_DISTANCE, t);
        if (density > 0.002) {
            float stepTransmittance = exp(-CLOUD_EXTINCTION * density * stepLength);
            float sunOpticalDepth = CLOUD_EXTINCTION * cloudSunOpticalDepth(position, cloud, weather);
            float3 direct = cloudSunOctaves(cosTheta, sunOpticalDepth) * cloud.lightIlluminance;
            // NUMOS' two-stream diffusion for the sky/ground ambient: multiply scattered light crosses
            // a thick cloud as 1 / (1 + k tau), not exp(-tau), so a thick base still receives light.
            float tauUp = CLOUD_EXTINCTION * density * max(topY - position.y, 0.0) * 0.7;
            float tauDown = CLOUD_EXTINCTION * density * max(position.y - cloud.baseY, 0.0) * 0.7;
            float tauSide = CLOUD_EXTINCTION * density * cloud.thickness * 0.4;
            float kDiffusion = 0.15;
            float tUp = 1.0 / (1.0 + kDiffusion * tauUp);
            float tDown = 1.0 / (1.0 + kDiffusion * tauDown);
            float tSide = 1.0 / (1.0 + kDiffusion * tauSide);
            float3 sky = cloud.ambientRadiance;
            float3 below = sky * 0.3;
            float3 ambient = 0.4 * sky * tUp + 0.4 * below * tDown
                    + 0.1 * cloud.horizonSunward * tSide
                    + 0.1 * cloud.horizonAntiSunward * tSide;
            scattering += transmittance * (direct + ambient) * (1.0 - stepTransmittance);
            transmittance *= stepTransmittance;
            if (transmittance < 0.02) {
                break;
            }
        }
        position += rayDirection * stepLength;
    }

    result.scattering = scattering;
    result.transmittance = transmittance;
    return result;
}

// Tile-space cloud shadow map, one byte per texel. The bake integrates the body SDF toward the
// dominant body; terrain below the deck reads this instead of repeating the density march per NEE.
public float cloudSampleShadowMap(float2 tilePosition) {
    uint dim = uint(CLOUD_SHADOW_DIM);
    float2 uv = frac(tilePosition) * float(dim);
    int2 floorCell = min(int2(floor(uv)), int2(int(dim) - 1));
    float2 f = uv - float2(floorCell);
    uint2 cell = uint2(floorCell);
    uint2 next = cell + uint2(1u);
    next.x = next.x >= dim ? 0u : next.x;
    next.y = next.y >= dim ? 0u : next.y;

    ConstPtr<uint> words = ConstPtr<uint>(pc.cloudShadowAddr);
    float2 t00 = float2(words[cell.y * dim + cell.x] & 255u,
                        words[next.y * dim + cell.x] & 255u) * (1.0 / 255.0);
    float2 t10 = float2(words[cell.y * dim + next.x] & 255u,
                        words[next.y * dim + next.x] & 255u) * (1.0 / 255.0);
    float2 tx = lerp(t00, t10, f.x);
    return lerp(tx.x, tx.y, f.y);
}

// Fraction of the sun/moonlight that reaches `position` through the cloud deck. Used by the celestial
// NEE at every shaded vertex, so terrain under a cloud is lit by the same density field the cloud is
// drawn from. Positions above the deck are always lit.
public float cloudShadowTransmittance(float3 position, float3 lightDirection, CloudState cloud) {
    if (!cloud.enabled || lightDirection.y <= 0.02
            || cloud.coverage <= 0.0 || cloud.densityScale <= 0.0) {
        return 1.0;
    }
    float top = cloud.baseY + cloud.thickness;
    if (position.y >= top) {
        return 1.0;
    }
    // Below the deck the baked map is authoritative; inside the slab the live march is needed because
    // the map only holds the column integral.
    if (position.y < cloud.baseY && pc.cloudShadowAddr != uint64_t(0)) {
        float2 uv = frac((position.xz + cloud.anchor) * (1.0 / CLOUD_PLACEMENT_TILE));
        return cloudSampleShadowMap(uv);
    }
    float tEnter = max(0.0, (cloud.baseY - position.y) / lightDirection.y);
    float tExit = (top - position.y) / lightDirection.y;
    if (tExit <= tEnter) {
        return 1.0;
    }
    tExit = min(tExit, tEnter + cloud.thickness * 4.0);

    float weather = cloudSampleWeather(
            float2(position.x + cloud.anchor.x, position.z + cloud.anchor.y)
                    * (1.0 / CLOUD_WEATHER_TILE));
    float stepLength = (tExit - tEnter) / float(CLOUD_SHADOW_STEPS);
    float opticalDepth = 0.0;
    for (int i = 0; i < CLOUD_SHADOW_STEPS; i++) {
        float3 samplePosition = position
                + lightDirection * (tEnter + (float(i) + 0.5) * stepLength);
        opticalDepth += cloudDensityAt(samplePosition, cloud, weather, false);
    }
    return exp(-CLOUD_EXTINCTION * opticalDepth * stepLength);
}
