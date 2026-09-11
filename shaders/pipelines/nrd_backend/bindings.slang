// NRD back end inputs. The two denoised lanes come from the denoiser; the raw lanes and the guides are
// here because NRD skips sky pixels outright (REBLUR early-outs outside its denoising range), so those
// pixels have to be reconstructed from the un-denoised signal instead of read back from the denoiser. The
// material guides re-apply the factor the front end divided out.
[[vk::binding(0, 0)]] public Sampler2D inDiffRadianceHitDist;
[[vk::binding(1, 0)]] public Sampler2D inSpecRadianceHitDist;
[[vk::binding(2, 0)]] public Sampler2D rawDiffRadianceHitDist;
[[vk::binding(3, 0)]] public Sampler2D rawSpecRadianceHitDist;
[[vk::binding(4, 0)]] public Sampler2D gNormal;
[[vk::binding(5, 0)]] public Sampler2D gDepth;
[[vk::binding(6, 0)]] [format("rgba16f")] public RWTexture2D<float4> outCombined;
[[vk::binding(7, 0)]] public Sampler2D gAlbedo;
[[vk::binding(8, 0)]] public Sampler2D gSpecAlbedo;
