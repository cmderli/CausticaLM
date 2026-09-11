// NRD inputs packed from the renderer's own guide buffers.
//
// NRD does not take the renderer's guide layout directly: it wants world-space normals, linear roughness
// (sqrt of the renderer's alpha), camera-space view-Z, exposure-free radiance and hit distances normalized
// through its own parameters. Every conversion here is a port of the matching helper in NRD's NRD.hlsli,
// kept in step with the encodings the library was built with (NRD_NORMAL_ENCODING = RGBA8_UNORM,
// NRD_ROUGHNESS_ENCODING = LINEAR). A mismatch shows up as a wrong image, not as an error, so the
// encodings are pinned on both sides and the shim refuses to create the denoiser if the linked library
// reports anything else.
[[vk::binding(0, 0)]] public Sampler2D gNormal;
[[vk::binding(1, 0)]] public Sampler2D gDepth;
[[vk::binding(2, 0)]] public Sampler2D inDiffRadianceHitDist;
[[vk::binding(3, 0)]] public Sampler2D inSpecRadianceHitDist;
[[vk::binding(4, 0)]] [format("rgba8")] public RWTexture2D<float4> outNormalRoughness;
[[vk::binding(5, 0)]] [format("r32f")] public RWTexture2D<float> outViewZ;
[[vk::binding(6, 0)]] [format("rgba16f")] public RWTexture2D<float4> outDiffRadianceHitDist;
[[vk::binding(7, 0)]] [format("rgba16f")] public RWTexture2D<float4> outSpecRadianceHitDist;
// Material guides, read only to divide the material factor out of both lanes (see nrdDemodFactor).
[[vk::binding(8, 0)]] public Sampler2D gAlbedo;
[[vk::binding(9, 0)]] public Sampler2D gSpecAlbedo;
