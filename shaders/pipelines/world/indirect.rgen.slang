// Indirect pass of the wavefront renderer.
//
// Resumes the continuations the primary pass queued and runs the bounce loop: NEE toward the sun, RIS
// emitter lighting, thin-surface SSS, BSDF continuation and Russian roulette. It captures no guides and
// owns no motion vectors — guides and primary are deliberately absent from the include list
// below, so guide state cannot leak into this pass by accident.
//
// Lighting is iterative here rather than recursive from closest-hit, so maxRayRecursionDepth stays 1.
// At each surface vertex:
//   * NEE toward the sun — a single analytic directional light, gated by one hit-object visibility
//     query (TerminateOnFirstHit | SkipClosestHit). Lambertian BRDF = albedo/PI.
//   * a cosine-weighted Lambertian continuation ray (throughput *= albedo; the PI and cos cancel).
// Sky / ambient light needs no separate AO pass: continuation rays that escape land in world.rmiss,
// which returns sky radiance — so soft sky fill and contact AO both fall out of the path integral.
//
// Each frame emits a single SPP-averaged (still noisy) estimate; temporal convergence is the denoiser's
// job. With RR disabled the output is the raw noisy path trace — a reference view, not a converged
// image. Output is scene-linear ACEScg HDR (R16G16B16A16_SFLOAT); display mapping happens afterward.
//
// The build emits an ordinary TraceRay fallback and an optional EXT SER variant from this source.
import world_common;
import world_core;
import math;
import medium;
import segment;
import water;
import trace;
#ifdef CAUSTICA_ENABLE_EXT_SER
import trace_ser;
#endif
import lighting;
import sky;
import clouds;
import bindings;

// Atmospheric transmittance LUT (RtSkyLut), also bound to world.rmiss at the same binding: raygen reads it
// to colour the NEE sun/moonlight, the miss shader reads it to tint the visible discs and stars.

// This dispatch's cloud layer, derived once beside the celestial light so every shaded vertex tests its
// sun/moon visibility against the same density field world.rmiss draws. Only the shadow query is read
// here; the layer's radiance reaches the path through the miss shader.
public static CloudState cloudFrame = {};

// Reported as a lane's hit distance when the ray escaped to sky. The renderer works in blocks and the
// scene is bounded far below this, so it reads as "no hit" without being a special case downstream.
public static const float SKY_HIT_DISTANCE = 1.0e5;
// Radiance split by the lobe that produced it, plus each lane's first-bounce hit distance in blocks. A
// path belongs to the lane of the FIRST lobe sampled at the primary vertex, because that is the lobe a
// denoiser reconstructs; direct lighting at that vertex is evaluated for both lobes at once (one shadow
// ray), so it is split by BRDF component instead. Adding the two lanes back together reproduces exactly
// the combined radiance the display chain consumes when no denoiser asks for the split.
public struct PathRadiance {
    public float3 diffuse;
    public float3 specular;
    public float diffHitDist;
    public float specHitDist;
    // The part of the two above that the path's own first shaded vertex contributed: everything reaching it
    // directly, and nothing that arrived by a further bounce. A trace stopped at that vertex returns exactly
    // this, which is what lets a budget-limited trace be combined with a full one without either being counted
    // twice -- see the aperture sweep, which spends its extra samples on this and keeps the indirect lighting
    // from the one full path.
    public float3 primaryDiffuse;
    public float3 primarySpecular;
};

// The camera ray of one aperture sample of a pixel, for the sweep below. A depth-of-field disc is the image of
// the lens, so it is only sampled where the primary rays land on it: a pixel's disc receives the contribution of
// the points of the pupil its own rays started from.
//
// Sample zero is NOT built here. The guide pass owns the primary ray and samples sample zero of the pupil with
// it, so this pass reads that ray back from the queue instead of deriving it a second time: deriving it would
// re-trace a ray the frame already paid for, and it would take the guides' ray away from the image the guides
// describe.
public PathSegment apertureCameraSegment(float3 axisEye, float3 axisDir, float rayConeSpread,
                                         uint2 pix, uint sampleIndex, uint seed) {
    float3 dir = axisDir;
    float3 origin = axisEye;
    if (worldPush.dofParams.z != 0.0) {
        float maxCocBlocks = max(worldPush.dofParams.w * rayConeSpread, RAY_CONE_MIN_WIDTH);
        float lensRadius = min(worldPush.dofParams.x, maxCocBlocks);
        float3 lens = axisEye + sampleLensDisc(lensRadius, float2(pix), worldPush.frameIndex, sampleIndex);
        dir = normalize(axisEye + axisDir * worldPush.dofParams.y - lens);
        origin = lens;
    }
    // The camera ray may see the sun/moon disc. Every continuation the path takes resets the flag itself, so
    // this cannot leak deeper into the path.
    MediumStack medium = makeMediumStack((worldPush.flags & 1u) != 0u
            ? makeDielectricMedium(true, WATER_IOR, worldPush.waterParams.xyz, 1.0)
            : airMedium());
    return makePathSegment(origin, dir, float3(1.0, 1.0, 1.0),
            medium, 0.0, rayConeSpread, seed, 0, true);
}

// Folds one traced path into the frame's running sums. Shared by the aperture sweep and the queue loop so a
// sample counts the same whichever loop produced it -- the two differ only in which point of the pupil the ray
// left from, which is not something the accumulators can see.
public void accumulatePath(PathRadiance radiance, inout float3 frameDiffuse, inout float3 frameSpecular,
                           inout float diffHitDistSum, inout float specHitDistSum,
                           inout uint diffHitDistCount, inout uint specHitDistCount) {
    frameDiffuse += radiance.diffuse;
    frameSpecular += radiance.specular;
    // Averaged over the samples that actually traced the lane, not over every sample: a lane is only taken by
    // the paths whose primary vertex sampled that lobe.
    if (radiance.diffHitDist >= 0.0) {
        diffHitDistSum += radiance.diffHitDist;
        diffHitDistCount++;
    }
    if (radiance.specHitDist >= 0.0) {
        specHitDistSum += radiance.specHitDist;
        specHitDistCount++;
    }
}

// `bounceBudget` is the last vertex this trace may shade, and it decides both what the trace costs and what it
// reports: at PRIMARY_ONLY_BOUNCES the loop cannot take a continuation, so the trace is the camera ray and its
// own vertex' lighting with no further NEE, RIS, roulette or second ray -- and the path beyond that vertex is
// reported as nothing rather than as a discarded estimate, so a budget-limited trace can be averaged together
// with a full one without either being counted twice.
public PathRadiance tracePath(PathSegment seg, uint2 pix, uint sampleIndex, int bounceBudget) {
    float3 Ldiff = float3(0.0, 0.0, 0.0);
    float3 Lspec = float3(0.0, 0.0, 0.0);
    // -1 until the primary vertex picks a continuation lobe, then 0 diffuse / 1 specular for the rest of
    // the path. Everything after that choice belongs to the chosen lane, including deeper bounces.
    int lane = -1;
    // Negative until the lane's own ray has been traced, so the first hit after the choice wins.
    float diffHitDist = -1.0;
    float specHitDist = -1.0;
    float3 ro = seg.ro;
    float3 rd = seg.rd;
    float3 throughput = seg.throughput;
    // Pass A is fixed at one sample per pixel. Decorrelate each Pass B resample here so configured SPP
    // produces independent lighting/BSDF paths from the shared terminal continuation.
    uint seed = seg.seed ^ sampleIndex * 2246822519u;
    seed = pcg(seed);
    float rayConeWidth = seg.rayConeWidth;
    float rayConeSpread = max(seg.rayConeSpread, RAY_CONE_MIN_SPREAD);
    int maxBounces = min(int(worldPush.maxBounces), max(bounceBudget, PRIMARY_ONLY_BOUNCES));
    // A trace that stops at the vertex the camera ray reached reports only that vertex' lighting. Subtracting the
    // primary contribution out of the total is exact for a full trace (it cancels) and leaves exactly the primary
    // contribution for a stopped one, so the two can be averaged without either being counted twice.
    float indirectWeight = maxBounces <= PRIMARY_ONLY_BOUNCES ? 0.0 : 1.0;
    int rrStart = maxBounces <= 3 ? 1 : 2;
    // RIS emitter NEE: direct lighting from block emitters is active when lights are published and the
    // candidate count is non-zero. Otherwise emitters contribute only when a path hits them.
    bool risOn = worldPush.risCandidates > 0u && worldPush.lightCount > 0u && pc.lightBufAddr != 0;
    // Independent light-proposal RNG. Screen-space coherent selection is intentionally deferred until
    // there is a real presampled RIS tile buffer; direct sharing made tile-shaped lighting noise visible.
    // Measured: making a 16x16 tile share this seed recovers at most 1.3ms, so a future pool should share
    // the POOL rather than the seed — the win is in removing dependent loads, not in coherence.
    uint proposalSeed = pix.x * 1973u + pix.y * 9277u + 26699u
            ^ worldPush.frameIndex * 2654435761u ^ sampleIndex * 2246822519u;
    proposalSeed = pcg(proposalSeed);
    // The medium the segment starts in. For the camera segment that is water when the eye is submerged
    // (so the first ray already carries the right relative index and absorption); for a segment split off
    // a dielectric it is whatever the parent was travelling through.
    MediumStack medium = seg.medium;
    bool waterWaves = (worldPush.flags & 16u) != 0u; // animated wave-normal perturbation
    // Sky-disc gate (see world.rmiss): the sun/moon disc is shown to the primary ray and to rays spawned
    // by a specular/dielectric bounce (mirror reflections + refractions of the sun/moon), but hidden from
    // diffuse-sampled continuation rays — those are already covered by the sun NEE, so the tiny disc
    // would be a double-counted firefly. Reset true on every specular/water bounce, false on a diffuse one.
    bool showCelestial = seg.showCelestial;
    // Pass B treats every hit it sees as one indirect-depth step, regardless of material or selected
    // continuation lobe. Deliberately start at zero instead of seg.bounce: the first hit of either
    // queued split branch is still the primary-terminal shading point. Hits 0 and 1 retain SSS; RIS
    // remains active with its full configured candidate count at every hit. Pass A's primary/interface
    // prefix is outside this SSS quality budget.
    int indirectDepth = 0;
    // Accumulated only while the path is still at the vertex the camera ray reached, which is the one whose
    // lighting a budget-limited trace reproduces. A dielectric interface keeps the path at that vertex for
    // another iteration, which is why this follows the bounce counter rather than the iteration.
    float3 primaryDiffuse = float3(0.0, 0.0, 0.0);
    float3 primarySpecular = float3(0.0, 0.0, 0.0);
    for (int bounce = seg.bounce; bounce <= maxBounces; bounce++) {
        bool atPrimaryVertex = bounce == 0;
        // Radiance SBT records run any-hit only for true alpha cutout. Translucent/water go straight to
        // closest-hit for dielectric handling. Geometry is double-sided; the chit flips the normal.
        // Primary (bounce 0) is the camera ray (CULL_PRIMARY): sees particles but not the first-person
        // player. Bounce rays are secondary (CULL_SECONDARY): exclude particles, include the player.
#ifdef CAUSTICA_ENABLE_EXT_SER
        // SER lifetime phase: keep paths that are not roulette-eligible, paths that may terminate via
        // roulette, and paths guaranteed to end at the bounce cap in separate coherence groups.
        uint pathPhaseHint = bounce >= maxBounces ? 2u : (bounce >= rrStart ? 1u : 0u);
        traceRadianceReordered(bounce == 0 ? CULL_PRIMARY : CULL_SECONDARY, ro, 0.0, rd, 10000.0,
                               showCelestial, rayConeWidth, rayConeSpread, pathPhaseHint);
#else
        traceRadiance(bounce == 0 ? CULL_PRIMARY : CULL_SECONDARY, ro, 0.0, rd, 10000.0,
                      showCelestial, rayConeWidth, rayConeSpread);
#endif

        // First hit along whichever lane the primary vertex chose: this trace is the ray that lobe
        // generated, so its length is that lane's hit distance. A miss reports the sky sentinel, which a
        // denoiser reads as "infinitely far" (maximum blur) rather than "close".
        if (lane == 1 && specHitDist < 0.0) {
            specHitDist = payload.hitT < 0.0 ? SKY_HIT_DISTANCE : payload.hitT;
        } else if (lane == 0 && diffHitDist < 0.0) {
            diffHitDist = payload.hitT < 0.0 ? SKY_HIT_DISTANCE : payload.hitT;
        }

        if (payload.hitT < 0.0) {
            // A ray still in water should leave through a water interface. A miss instead means the
            // streamed/open volume has no known exit; do not reinterpret that unknown region as air
            // and reveal sky.
            if (medium.current.water) {
                break;
            }

            // world.rmiss writes sky radiance (gradient + sun/moon disc + stars) into the payload's
            // surface words as a full float3, so this is absolute scene units with nothing to undo. The
            // packed show-celestial flag decides whether this path segment may see the bright disc.
            float3 sky = payloadSky();
            // Escaped to sky. A camera ray that hit nothing is the background: no lobe chose it, so it
            // belongs to the diffuse lane.
            if (lane == 1) {
                Lspec += throughput * sky;
                if (atPrimaryVertex) {
                    primarySpecular += throughput * sky;
                }
            } else {
                Ldiff += throughput * sky;
                if (atPrimaryVertex) {
                    primaryDiffuse += throughput * sky;
                }
            }
            break;
        }

        // Beer–Lambert: attenuate along the segment just travelled by the medium it lay inside. Applies
        // to every hit reached while inside a volume dielectric (its own exit face, or whatever content
        // lies within it), shifting the transmitted radiance with distance. Air's extinction is zero, so
        // the ordinary case costs one comparison.
        if (any(medium.current.extinction > 0.0)) {
            throughput *= exp(-medium.current.extinction * payload.hitT);
        }

        float3 n = payloadNormal(); // oriented toward the incoming ray by the closest-hit
        float3 hitPos = ro + rd * payload.hitT;
        rayConeWidth = max(rayConeWidth + rayConeSpread * max(payload.hitT, 0.0), RAY_CONE_MIN_WIDTH);
        uint material = payloadMaterial();
        int hitDepth = indirectDepth++;

        // ---- Dielectric interface: water, stained glass, ice. ONE handler and one behaviour — every
        // dielectric is a volume. Fresnel from the relative index across the face, then reflect or
        // refract; no diffuse response and no sun NEE, because the surface is specular. Transmission
        // pushes or pops a participating medium, and the segment travelled inside it is attenuated by
        // that medium's extinction rather than by a tint multiply at the interface.
        //
        // Water is the only material that adds anything: the animated wave normal, and a medium marked
        // for the caustic term. Which side of the face we are on comes from the geometry
        // (payloadDielectricEntering), so a stray or missing face cannot desync the medium for the rest
        // of the path.
        if (material == MATERIAL_WATER || material == MATERIAL_DIELECTRIC) {
            // A dielectric has no local shading term. At the path cap its continuation cannot be traced,
            // so avoid computing an unused Fresnel choice, medium update, and roulette step.
            if (bounce >= maxBounces) {
                break;
            }
            bool isWater = material == MATERIAL_WATER;
            bool entering = payloadDielectricEntering();
            float3 tint = payloadAlbedo();
            float transmission = clamp(payloadTransmission(), 0.0, 1.0);

            // Ripple the (near-flat) surface normal before any Fresnel/guide use → glints, broken
            // reflections, wobbling refraction. The wave domain is world-stable: rebased hit XZ plus the
            // pushed anchor (terrain origin mod 4096) survives the per-move terrain rebase.
            float3 geometricNormal = n; // unrippled; ray origins offset along this
            if (isWater && waterWaves) {
                float2 waterDomain = hitPos.xz + worldPush.waterAnchor.xy;
                float waterFootprint = rayConeWidth / max(abs(dot(-rd, geometricNormal)), 0.2);
                n = applyWaterWaves(geometricNormal, waterDomain,
                                    worldPush.waterParams.w, waterFootprint);
            }

            // The medium this face opens into, and the one it returns to on the way out — which is what
            // the stack remembers.
            Medium entered = makeDielectricMedium(isWater, max(payloadIor(), 1.0), tint, transmission);
            float etaI = medium.current.ior;
            float etaT = entering ? entered.ior : medium.outer.ior;

            float cosI = clamp(dot(-rd, n), 0.0, 1.0);
            float F = fresnelDielectric(cosI, etaI, etaT);
            float3 transmittedDir = refract(rd, n, etaI / etaT);
            // The translucent terrain layer is recessed by TRANSLUCENT_INSET (RtTerrainMesher), so a
            // glass/ice face touching a slab or stair sits a hair behind that neighbour's surface. A full
            // SURF_BIAS would restart the transmitted ray past the neighbour and see through it. Water
            // comes from the fluid mesher, which applies no inset, so it takes the ordinary bias.
            float transmitBias = isWater ? SURF_BIAS : INSET_TRANSMIT_BIAS;

            bool chooseReflection = rndf(seed) < F;
            if (chooseReflection) {
                rd = reflect(rd, n);
                ro = offsetSurfaceOrigin(hitPos, geometricNormal, rd, SURF_BIAS);
            } else {
                if (dot(transmittedDir, transmittedDir) <= 0.0) break; // TIR with F < 1 cannot happen
                rd = normalize(transmittedDir);
                ro = offsetSurfaceOrigin(hitPos, geometricNormal, rd, transmitBias);
                // Crossing into or out of the volume. Absorption is the medium's job from here, so the
                // tint is NOT also multiplied into the throughput — that would double-count it.
                if (entering) {
                    mediumPush(medium, entered);
                } else {
                    mediumPop(medium);
                }
            }
            showCelestial = true; // specular interface: the continuation ray may see the sun/moon disc
            // A dielectric reflects or refracts, never scatters diffusely, so a primary ray that lands on
            // one makes the whole path specular.
            if (lane < 0) {
                lane = 1;
            }
            if (bounce >= rrStart) {
                float q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.02, 1.0);
                if (rndf(seed) > q) {
                    break;
                }
                throughput /= q;
            }
            continue;
        }

        // Particles (material == 2): camera-facing receiver billboard. The instance mask keeps particles
        // off secondary rays, so this only fires at bounce 0; they receive direct light and one diffuse
        // path continuation into the scene, but never appear in shadows/reflections/GI or emit radiance.
        if (material == MATERIAL_PARTICLE) {
            float3 albedo = payloadAlbedo();
            // Billboards are diffuse receivers.
            if (lane < 0) {
                lane = 0;
            }

            float3 lightDir = celestialLight.dir;
            float lightHalfAngle = celestialLight.halfAngle;
            if (lightHalfAngle > 0.0) {
                lightDir = sampleSquare(lightDir, lightHalfAngle, seed);
            }
            // Billboards are effectively two-sided receivers. Bias from the side facing the light; shadow
            // rays exclude particles anyway, but this keeps the origin sane when the light is behind the
            // camera side.
            float signedNdl = dot(n, lightDir);
            float ndl = abs(signedNdl);
            if (ndl > 0.0) {
                float3 shadowOrigin = hitPos + (signedNdl >= 0.0 ? n : -n) * SURF_BIAS;
                float3 vis = visibility(shadowOrigin, lightDir, 10000.0).transmittance;
                vis *= cloudShadowTransmittance(shadowOrigin, lightDir, cloudFrame);
                if (max(vis.r, max(vis.g, vis.b)) > 0.0) {
                    float3 billboard = throughput * albedo * INV_PI * celestialLight.illuminance * ndl * vis;
                    Ldiff += billboard;
                    if (atPrimaryVertex) {
                        primaryDiffuse += billboard;
                    }
                }
            }

            // RIS direct lighting from block emitters (lava, glowstone, torches, ...) — same reservoir
            // sampler as terrain/entities, but two-sided: a billboard has no back face, so light
            // striking either side of the quad should still land (matches the sun/moon NEE above).
            if (risOn) {
                float3 v = -rd;
                Reservoir r = risInitial(hitPos, n, v, rd, albedo, float3(0.0, 0.0, 0.0), 1.0,
                                         true, 0.0, seed, proposalSeed);
                float3 unusedSpecular;
                float3 billboardRis = throughput * shadeReservoir(r, hitPos, n, v, rd, albedo,
                        float3(0.0, 0.0, 0.0), 1.0, true, 0.0, unusedSpecular);
                Ldiff += billboardRis;
                if (atPrimaryVertex) {
                    primaryDiffuse += billboardRis;
                }
            }

            if (bounce >= maxBounces) {
                break;
            }
            throughput *= albedo;
            ro = hitPos + n * SURF_BIAS;
            rd = cosineDir(n, seed);
            rayConeSpread = max(rayConeSpread, RAY_CONE_DIFFUSE_SPREAD);
            showCelestial = false; // diffuse receiver: direct sun/moon was handled by NEE above
            continue;
        }

        float3 albedo = payloadAlbedo();
        float sss = payloadSss(); // LabPBR SSS strength (0 when absent)
        float3 p = hitPos + n * SURF_BIAS;
        float3 v = -rd; // view direction (toward the camera / incoming ray)

        // Material split. Metals tint F0 by albedo and have no diffuse lobe; dielectrics use a fixed
        // 0.04 F0, sourced from the chit (LabPBR custom/metal F0, or the dielectric default it applies).
        float rough = clamp(payloadRoughness(), 0.0, 1.0);
        float metal = clamp(payloadMetalness(), 0.0, 1.0);
        // Unresolvably tight lobes become exact delta mirrors; everything else keeps its authored value.
        // Snapping to 0 also propagates into the guide buffers, which is what tells RR to hold the
        // reflection sharp instead of accumulating it as a glossy lobe.
        bool exactSpecular = isDeltaAlpha(rough);
        rough = exactSpecular ? 0.0 : rough;
        float3 diffAlb = albedo * (1.0 - metal);
        float3 F0 = payload.f0;

        // Emissive surfaces (lava, glowstone, torches, ...) add radiance directly, colored by albedo.
        // RIS emitter NEE: the direct-hit emission term is gated only for emitters RIS actually samples
        // (payload emitter-in-list bit) — and then only off diffuse continuation rays (showCelestial
        // false), which the previous vertex's RIS already accounted for; the primary ray and
        // specular/dielectric bounces (showCelestial true) still add it, since no emitter NEE happened
        // along those. Emitters NOT in the light buffer (sparse/sub-threshold footprints, entities) are
        // never NEE-sampled, so they always gather — bit-identical to the no-NEE path, no energy lost.
        // Emission is already the full HDR luminance (look-package baseline or absolute cd/m² override
        // multiplier baked in Java at material-compile time — see MaterialHeader.features packing).
        float emission = payloadEmission();
        // One gate drives both the RIS call below and the direct-hit emission suppression here. They must
        // agree: gateEmitter exists only because RIS already accounted for this emitter, so a bounce where
        // RIS is skipped must also gather emission directly or that light is lost outright.
        bool gateEmitter = risOn && payloadEmitterInList();
        if (emission > 0.0 && (!gateEmitter || showCelestial)) {
            // A surface's own emission is a diffuse-class signal at the primary vertex; deeper in the path
            // it is simply radiance arriving through the lane already chosen.
            if (lane == 1) {
                Lspec += throughput * albedo * emission;
                if (atPrimaryVertex) {
                    primarySpecular += throughput * albedo * emission;
                }
            } else {
                Ldiff += throughput * albedo * emission;
                if (atPrimaryVertex) {
                    primaryDiffuse += throughput * albedo * emission;
                }
            }
        }

        // NEE: direct light from the dominant celestial body (sun by day, moon by night) — Lambert
        // diffuse + GGX specular, one shadow ray. Jitter the light direction within its square angular
        // extent (celestialLight.halfAngle) so the shadow ray samples the light's quad — soft penumbrae
        // over accumulation. The same sampled direction drives ndl, the shadow ray, and the specular
        // half-vector.
        float3 lightDir = celestialLight.dir;
        float lightHalfAngle = celestialLight.halfAngle;
        if (lightHalfAngle > 0.0) {
            lightDir = sampleSquare(lightDir, lightHalfAngle, seed);
        }
        float ndl = max(0.0, dot(n, lightDir));
        if (ndl > 0.0) {
            VisibilityResult shadow = visibility(p, lightDir, 10000.0);
            float3 vis = shadow.transmittance;
            // The shadow ray only proves terrain is clear; the cloud deck it passes through attenuates
            // the same light. Shading and the drawn cloud therefore share one density field.
            vis *= cloudShadowTransmittance(p, lightDir, cloudFrame);
            // Underwater receiver whose shadow ray crossed a water surface: scale the direct light by
            // the wave-refraction caustic at the exit point. Focusing scales the incident irradiance,
            // so it applies to the whole NEE term (diffuse + specular).
            if (medium.current.water && waterWaves && shadow.waterHitT > 0.0) {
                vis *= waterCaustic(p + lightDir * shadow.waterHitT, lightDir, shadow.waterHitT);
            }
            if (max(vis.r, max(vis.g, vis.b)) > 0.0) {
                float3 diffBrdf = diffAlb * INV_PI; // Lambertian diffuse (f = albedo/PI)
                float3 h = normalize(lightDir + v);
                float ndh = max(0.0, dot(n, h));
                float ndv = max(1.0e-4, dot(n, v));
                float vdh = max(0.0, dot(v, h));
                float D = ggxD(ndh, rough);
                float G = ggxG1(ndv, rough) * ggxG1(ndl, rough); // separable Smith masking
                float3 F = fresnelSchlick(vdh, F0);
                float3 specBrdf = (D * G) * F / (4.0 * ndv * ndl); // Cook–Torrance specular
                float3 direct = throughput * celestialLight.illuminance * ndl * vis;
                // One shadow ray serves both lobes, so at the primary vertex the term is split by BRDF
                // component; anywhere deeper the whole term belongs to the lane already chosen.
                float3 neeDiffuse = lane == 1
                        ? float3(0.0) : direct * (lane < 0 ? diffBrdf : diffBrdf + specBrdf);
                float3 neeSpecular = lane == 0
                        ? float3(0.0) : direct * (lane < 0 ? specBrdf : diffBrdf + specBrdf);
                Ldiff += neeDiffuse;
                Lspec += neeSpecular;
                if (atPrimaryVertex) {
                    primaryDiffuse += neeDiffuse;
                    primarySpecular += neeSpecular;
                }
            }
        }

        // RIS direct lighting from block emitters (single-frame only, no temporal reservoir reuse).
        // Front lobe + SSS backlighting are resampled together and share ONE shadow ray — a light can
        // only be in front of n or behind it, never both, so combining them into one target function is
        // exact, not an approximation. The SSS term is gated off past MAX_SSS_INDIRECT_DEPTH by passing
        // sss=0 there (falls back to plain front-only RIS).
        if (risOn) {
            float activeSss = hitDepth <= MAX_SSS_INDIRECT_DEPTH ? sss : 0.0;
            Reservoir r = risInitial(hitPos, n, v, rd, diffAlb, F0, rough, false, activeSss,
                                     seed, proposalSeed);
            float3 risSpecular;
            float3 risDiffuse = shadeReservoir(r, hitPos, n, v, rd, diffAlb, F0, rough, false,
                                               activeSss, risSpecular);
            float3 direct = throughput;
            float3 risDiffuseTerm = lane == 1
                    ? float3(0.0) : direct * (lane < 0 ? risDiffuse : risDiffuse + risSpecular);
            float3 risSpecularTerm = lane == 0
                    ? float3(0.0) : direct * (lane < 0 ? risSpecular : risDiffuse + risSpecular);
            Ldiff += risDiffuseTerm;
            Lspec += risSpecularTerm;
            if (atPrimaryVertex) {
                primaryDiffuse += risDiffuseTerm;
                primarySpecular += risSpecularTerm;
            }
        }

        // Thin-surface SSS transmission. Light entering from the back face scatters through toward the
        // viewer via a forward-biased HG phase (leaves/grass backlit by the sun glow when you look
        // through them toward the light source). Shadow ray fires from the back face to avoid
        // self-occlusion. cosT = dot(lightDir, rd): rd tracks the ray toward the sun when the sun is
        // behind the slab, giving cosT ≈ 1 → forward-scatter peak.
        if (sss > 0.0 && hitDepth <= MAX_SSS_INDIRECT_DEPTH) {
            float backNdl = max(0.0, dot(-n, lightDir));
            if (backNdl > 0.0) {
                VisibilityResult shadowBack = visibility(hitPos - n * SURF_BIAS, lightDir, 10000.0);
                float3 visB = shadowBack.transmittance;
                visB *= cloudShadowTransmittance(hitPos - n * SURF_BIAS, lightDir, cloudFrame);
                // Same caustic as the front-face NEE — underwater kelp/seagrass transmission should
                // flicker with the same light bands as the floor around it.
                if (medium.current.water && waterWaves && shadowBack.waterHitT > 0.0) {
                    visB *= waterCaustic(hitPos + lightDir * shadowBack.waterHitT,
                                         lightDir, shadowBack.waterHitT);
                }
                if (max(visB.r, max(visB.g, visB.b)) > 0.0) {
                    float cosT = dot(lightDir, rd);
                    // Forward-scatter transmission is diffuse-class energy.
                    float3 transmission = throughput * diffAlb * sss * SSS_STRENGTH * hg(cosT, SSS_G)
                            * backNdl * celestialLight.illuminance * visB;
                    if (lane == 1) {
                        Lspec += transmission;
                        if (atPrimaryVertex) {
                            primarySpecular += transmission;
                        }
                    } else {
                        Ldiff += transmission;
                        if (atPrimaryVertex) {
                            primaryDiffuse += transmission;
                        }
                    }
                }
            }
        }

        // Preserve all local lighting at the terminal hit, but do not sample a continuation that the
        // bounce loop cannot trace.
        if (bounce >= maxBounces) {
            break;
        }

        // Indirect continuation: importance-sample the diffuse OR the GGX specular lobe, chosen by their
        // relative reflectance.
        float ps = exactSpecular && luminance(diffAlb) <= 1.0e-6
                       ? 1.0
                       : clamp(luminance(F0) / (luminance(F0) + luminance(diffAlb) + 1.0e-4), 0.1, 0.9);
        if (rndf(seed) < ps) {
            float3 l;
            if (exactSpecular) {
                // Authored zero is a delta distribution, not a narrow finite GGX lobe. This avoids the
                // degenerate VNDF at alpha=0 and preserves a genuinely sharp reflection without a floor.
                l = reflect(rd, n);
                throughput *= fresnelSchlick(clamp(dot(n, v), 0.0, 1.0), F0) / ps;
            } else {
                float3 h = sampleGGXVNDF(n, v, rough, seed);
                l = reflect(rd, h); // rd = -v: reflect the incoming ray about the microfacet normal
                float ndl2 = dot(n, l);
                if (ndl2 <= 0.0) {
                    break; // microfacet reflects below the surface: terminate this path
                }
                // VNDF + separable Smith ⇒ weight = F · G2/G1(v) = F · G1(l); divide by the lobe pdf ps.
                float3 F = fresnelSchlick(max(0.0, dot(v, h)), F0);
                throughput *= F * ggxG1(ndl2, rough) / ps;
                rayConeSpread = max(rayConeSpread, rough * RAY_CONE_GLOSSY_SPREAD_SCALE);
            }
            ro = p;
            rd = l;
            showCelestial = true; // specular lobe: this ray is a (glossy) mirror — show the disc
            if (lane < 0) {
                lane = 1;
            }
        } else {
            throughput *= diffAlb / (1.0 - ps);
            ro = p;
            rd = cosineDir(n, seed);
            rayConeSpread = max(rayConeSpread, RAY_CONE_DIFFUSE_SPREAD);
            showCelestial = false; // diffuse lobe: disc covered by NEE, hide it (anti-firefly)
            if (lane < 0) {
                lane = 0;
            }
        }

        // Russian roulette on deeper bounces: survive with prob = max channel, rescale survivors.
        if (bounce >= rrStart) {
            float q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.02, 1.0);
            if (rndf(seed) > q) {
                break;
            }
            throughput /= q;
        }
    }
    PathRadiance result;
    result.diffuse = primaryDiffuse + (Ldiff - primaryDiffuse) * indirectWeight;
    result.specular = primarySpecular + (Lspec - primarySpecular) * indirectWeight;
    // Negative when this sample never traced that lane (the other lobe was chosen, or the path ended at
    // the primary vertex). The caller averages only over the samples that did trace it — dividing by the
    // full sample count would bias both distances toward zero.
    result.diffHitDist = diffHitDist;
    result.specHitDist = specHitDist;
    result.primaryDiffuse = primaryDiffuse;
    result.primarySpecular = primarySpecular;
    return result;
}

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    // Derive this dispatch's directional light once. The same transmittance LUT drives terrain lighting
    // and the visible sun/moon, keeping atmospheric tint consistent across both consumers.
    SkyState sky = skyState(worldPush);
    celestialLight = dominantCelestialLight(sky, transmittanceLut);
    // The path integral is ACEScg end to end; sky.slang works in linear BT.709, so this is the one seam.
    celestialLight.illuminance = bt709ToAcesCg(celestialLight.illuminance);
    cloudFrame = cloudState(worldPush, sky);
    uint2 dispatchIndex = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    int2 pix = int2(dispatchIndex);
    uint spp = max(worldPush.spp, 1u);
    uint apertureSamples = max(uint(worldPush.dofSamples), 1u);
    // The camera ray of every aperture sample, rebuilt here rather than read from the guide pass: the guide
    // pass keeps one sample per pixel because a guide is a per-pixel surface property, and a lens-offset sample
    // would move the depth and motion a denoiser reprojects with. Both passes derive the ray from the same
    // pixel, jitter, frame and lens parameters, so they agree on where the eye and the pupil are.
    float2 size = float2(dimensions);
    float2 uv = (float2(pix) + 0.5) / size;
    float2 jndc = (uv + worldPush.jitter / size) * 2.0 - 1.0;
    float4 nearH = mul(worldPush.invViewProj, float4(jndc.x, jndc.y, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(jndc.x, jndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    float3 axisEye = nearP + worldPush.camOffset;
    float3 axisDir = normalize(farP - nearP);
    float rayConeSpread = primaryRayConeSpread(jndc, size, axisDir);
    uint baseSeed = (dispatchIndex.x * 1973u + dispatchIndex.y * 9277u + 26699u)
            ^ (worldPush.frameIndex * 2654435761u);
    uint pixelIndex = dispatchIndex.y * dimensions.x + dispatchIndex.x;
    ConstPtr<PackedPathSegment> queue = ConstPtr<PackedPathSegment>(pc.pathQueueAddr);
    float3 frameDiffuse = float3(0.0, 0.0, 0.0);
    float3 frameSpecular = float3(0.0, 0.0, 0.0);
    float diffHitDistSum = 0.0;
    float specHitDistSum = 0.0;
    uint diffHitDistCount = 0u;
    uint specHitDistCount = 0u;
    // One path per aperture sample beyond the first.
    //
    // The per-frame sample the platform has always taken is the guide pass's, one point of the pupil per pixel,
    // resolved over time by the denoiser. At a large circle of confusion that point is most of what a pixel sees,
    // so the disc stays speckled and its edge stair-steps however long the history is. Sweeping the pupil within
    // the frame is what fixes that, and it is the only thing that does: the disc exists solely at the points of
    // the pupil a pixel's own rays started from, so a broader filter has nothing more to gather from.
    //
    // Sample zero is the segment the guide pass queued and is traced by the queue loop below, so this loop only
    // traces the samples after it and the frame's cost is the guide pass's ray plus one path per extra sample.
    // The samples are divided out below, so their mean is the frame's estimate of the aperture integral.
    // An extra aperture sample exists to change which surface the pixel sees through that part of the pupil, and
    // the lighting of that surface is what it has to get right. Tracing a whole path for it would multiply the
    // frame's entire ray budget -- every extra bounce, NEE and RIS estimate -- to re-answer a question the first
    // sample already answered, so these samples stop at the vertex the camera ray reached and contribute no
    // indirect lighting; the first sample's indirect lighting stands for all of them. Both are averaged over the
    // sample count below, which is what makes the total the aperture's mean over the whole path rather than a
    // brightened sum or a darkened one.
    uint recordIndex = pixelIndex;
    for (uint apertureSample = 1u; apertureSample < apertureSamples; ++apertureSample) {
        uint apertureSeed = baseSeed ^ (apertureSample * 2654435761u);
        apertureSeed = pcg(apertureSeed);
        PathSegment camera = apertureCameraSegment(axisEye, axisDir, rayConeSpread, uint2(pix),
                apertureSample, apertureSeed);
        accumulatePath(tracePath(camera, uint2(pix), apertureSample * 0x9e3779b9u, PRIMARY_ONLY_BOUNCES),
                frameDiffuse, frameSpecular, diffHitDistSum, specHitDistSum,
                diffHitDistCount, specHitDistCount);
    }
    // Sample zero of the pupil, as the guide pass traced it. Its radiance carries the guides' own surface, which
    // is the one a denoiser reconstructs, so this is the lane that has to be in the frame.
    for (uint leaf = 0u; leaf < MAX_PATH_SEGMENTS; ++leaf) {
        PackedPathSegment packed = queue[recordIndex];
        PathSegment segment = unpackPathSegment(packed);
        for (uint s = 0u; s < spp; ++s) {
            accumulatePath(tracePath(segment, uint2(pix), s + leaf * spp, int(worldPush.maxBounces)),
                    frameDiffuse, frameSpecular, diffHitDistSum, specHitDistSum,
                    diffHitDistCount, specHitDistCount);
        }
        if (packed.nextRecord == PATH_NO_NEXT) {
            break;
        }
        recordIndex = packed.nextRecord;
    }
    // Pre-exposure is applied here and nowhere else: the entire path-traced result above is in absolute
    // scene units, and only the fp16 store is normalized.
    //
    // The three outputs are rgba16f, so anything over 65504 would round to +inf and propagate as NaN
    // through the denoiser. Pre-exposed, it only bites when a pixel would render >65504x mid-grey — the
    // sun mid-transition, a firefly off a tiny emitter — and ACES 2.0 renders anything a few EV over
    // white as white, so it costs nothing visible.
    float scale = worldPush.preExposure / (float(spp) * float(apertureSamples));

    float3 storedDiffuse = frameDiffuse * scale;
    float3 storedSpecular = frameSpecular * scale;
    outImage[pix] = float4(clamp(storedDiffuse + storedSpecular, float3(0.0), float3(HALF_MAX)), 1.0);
    // The split lanes, for a denoiser that reconstructs the lobes separately. Their alpha carries the
    // lane's mean first-bounce hit distance in blocks; radiance is clamped for the same reason as above.
    // A lane no sample took reports zero distance, which reads as "hit at the primary surface".
    float diffHitDist = diffHitDistCount > 0u ? diffHitDistSum / float(diffHitDistCount) : 0.0;
    float specHitDist = specHitDistCount > 0u ? specHitDistSum / float(specHitDistCount) : 0.0;
    gDiffRadianceHitDist[pix] = float4(clamp(storedDiffuse, float3(0.0), float3(HALF_MAX)), diffHitDist);
    gSpecRadianceHitDist[pix] = float4(clamp(storedSpecular, float3(0.0), float3(HALF_MAX)), specHitDist);
}
