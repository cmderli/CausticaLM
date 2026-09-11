// NRD back end: unpacks the denoised lanes and recombines them into the pre-exposed scene-linear radiance
// the rest of the frame graph consumes. Radiance arrives YCoCg-packed and with the material factor divided
// out (NRD's demodulation contract), so this undoes both.
//
// Sky pixels are special: NRD does not denoise them at all — REBLUR's passes early-out for pixels outside
// its denoising range, leaving its outputs unwritten there — so the sky is taken from the tracer's own
// lanes instead. That is the same division of labour the official NRD sample uses, where the application
// composites the sky itself; reading the denoiser's output for those pixels yields black.
import recon_common;
import bindings;

[[vk::push_constant]] NrdBackendPush pc;

// NRD.hlsli:407-417.
float3 yCoCgToLinear(float3 color) {
    float t = color.x - color.z;
    float3 r;
    r.y = color.x + color.z;
    r.x = t + color.y;
    r.z = t - color.y;
    return max(r, 0.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int2 pix = int2(id.xy);
    if (pix.x >= pc.width || pix.y >= pc.height) {
        return;
    }
    float2 uv = (float2(pix) + 0.5) / float2(float(pc.width), float(pc.height));

    float4 normal = gNormal.SampleLevel(uv, 0.0);
    float depth = gDepth.SampleLevel(uv, 0.0).x;
    bool sky = dot(normal.xyz, normal.xyz) < 1.0e-8 || depth <= 0.0;

    float3 radiance;
    if (sky) {
        // The tracer's own lanes, which the denoiser never touched and which carry no material factor to
        // restore: they are the pre-exposed radiance the rest of the frame graph consumes.
        radiance = rawDiffRadianceHitDist.SampleLevel(uv, 0.0).rgb
                 + rawSpecRadianceHitDist.SampleLevel(uv, 0.0).rgb;
    } else {
        // Undo the front end: unpack YCoCg, then re-apply the material factor it divided out, with the same
        // floor, so the two ends cancel exactly for every texel.
        float3 diffuse = yCoCgToLinear(inDiffRadianceHitDist.SampleLevel(uv, 0.0).rgb);
        float3 specular = yCoCgToLinear(inSpecRadianceHitDist.SampleLevel(uv, 0.0).rgb);
        diffuse *= nrdDemodFactor(gAlbedo.SampleLevel(uv, 0.0).rgb);
        specular *= nrdDemodFactor(gSpecAlbedo.SampleLevel(uv, 0.0).rgb);
        radiance = diffuse + specular;
    }
    outCombined[pix] = float4(radiance, 1.0);
}
