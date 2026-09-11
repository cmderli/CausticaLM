// Reconstruction-chain inputs: the noisy render-resolution trace plus the guide buffers the world pass
// already produces. Everything is read-only here; the only write is this pass's own output, which is
// render-res for an atrous iteration and display-res for the resolve.
[[vk::binding(0, 0)]] public Sampler2D srcImage;
[[vk::binding(1, 0)]] [format("rgba16f")] public RWTexture2D<float4> dstImage;
[[vk::binding(2, 0)]] public Sampler2D gDepth;
[[vk::binding(3, 0)]] public Sampler2D gNormal;
[[vk::binding(4, 0)]] public Sampler2D gAlbedo;
[[vk::binding(5, 0)]] public Sampler2D gSpecAlbedo;
