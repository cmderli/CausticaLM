/*
 * caustica_nrd - flat C ABI exposing NVIDIA Real-time Denoisers (NRD) to a
 * java.lang.foreign (FFM) caller.
 *
 * The ABI below is fixed; the Java side is written against this declaration
 * verbatim (see README.md for the same text in copy-pasteable form):
 *
 *   #define CAUSTICA_NRD_OK 0
 *
 *   typedef struct CausticaNrdImageView { uint64_t image; uint64_t view; } CausticaNrdImageView;
 *
 *   typedef struct CausticaNrdCreateInfo {
 *       uint64_t device;            // VkDevice
 *       uint64_t physicalDevice;    // VkPhysicalDevice
 *       uint64_t getDeviceProcAddr; // PFN_vkGetDeviceProcAddr
 *       uint32_t renderWidth;
 *       uint32_t renderHeight;
 *       uint32_t denoiser;          // 0 = REBLUR_DIFFUSE_SPECULAR, 1 = RELAX_DIFFUSE_SPECULAR
 *       uint32_t reserved;
 *   } CausticaNrdCreateInfo;
 *
 *   typedef struct CausticaNrdFrameInfo {
 *       uint64_t commandBuffer;     // VkCommandBuffer, already recording
 *       uint32_t frameIndex;        // advances by exactly 1 per denoised frame
 *       float    jitterX, jitterY;  // render-pixel jitter applied to the primary ray this frame
 *       float    denoisingRange;
 *       float    viewToClip[16];
 *       float    clipToView[16];
 *       float    worldToView[16];
 *       float    viewToWorld[16];
 *       CausticaNrdImageView viewZ;                 // r32f, render res
 *       CausticaNrdImageView normalRoughness;       // render res
 *       CausticaNrdImageView motion;                // rg16f, render res
 *       CausticaNrdImageView diffRadianceHitDist;   // rgba16f, render res
 *       CausticaNrdImageView specRadianceHitDist;   // rgba16f, render res
 *       CausticaNrdImageView outDiffRadianceHitDist;
 *       CausticaNrdImageView outSpecRadianceHitDist;
 *       float worldPrevToWorld[16]; // previous world -> current world; all-zero = derive
 *   } CausticaNrdFrameInfo;
 *
 *   int32_t caustica_nrd_create(const CausticaNrdCreateInfo* info);
 *   int32_t caustica_nrd_denoise(const CausticaNrdFrameInfo* frame);
 *   void    caustica_nrd_destroy(void);
 *
 * Implementation notes:
 *
 *  - NRD 4.17 makes no graphics API calls. It hands out POD pipeline/POD
 *    resource descriptions plus embedded SPIR-V bytecode; this translation
 *    unit creates and owns every Vulkan object they require (shader modules,
 *    pipelines, descriptor set layouts, pipeline layout, descriptor pools and
 *    sets, samplers, the constants UBO and the NRD pool textures).
 *
 *  - Only PFN_vkGetDeviceProcAddr is used to reach Vulkan. The library never
 *    calls vkGetInstanceProcAddr, never links libvulkan and never queries
 *    instance/physical-device level state, which keeps it usable on top of any
 *    embedder that already owns the VkInstance.
 *
 *  - Constants are pushed into a uniform buffer with vkCmdUpdateBuffer. That
 *    needs no host-visible memory (the ABI cannot deliver
 *    vkGetPhysicalDeviceMemoryProperties, and a uniform buffer must stay valid
 *    until the frame's submission completes).
 *
 *  - Matrices are handed to NRD unchanged: NRD expects column-major storage and
 *    column-vector usage (clip = matrix * position), i.e. m[column * 4 + row].
 */

#include <vulkan/vulkan.h>

#include "NRD.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

#if defined(_WIN32)
#  define CAUSTICA_NRD_EXPORT extern "C" __declspec(dllexport)
#else
#  define CAUSTICA_NRD_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define CAUSTICA_NRD_OK 0

// Encodings the linked NRD library must have been built with. They come from the
// shim's CMakeLists (NRD_NORMAL_ENCODING / NRD_ROUGHNESS_ENCODING, pinned there
// because the host writes IN_NORMAL_ROUGHNESS itself) and are verified against
// nrd::LibraryDesc at runtime.
#ifndef CAUSTICA_NRD_EXPECTED_NORMAL_ENCODING
#  define CAUSTICA_NRD_EXPECTED_NORMAL_ENCODING 0 // nrd::NormalEncoding::RGBA8_UNORM
#endif
#ifndef CAUSTICA_NRD_EXPECTED_ROUGHNESS_ENCODING
#  define CAUSTICA_NRD_EXPECTED_ROUGHNESS_ENCODING 1 // nrd::RoughnessEncoding::LINEAR
#endif

typedef struct CausticaNrdImageView { uint64_t image; uint64_t view; } CausticaNrdImageView;

typedef struct CausticaNrdCreateInfo {
    uint64_t device;            // VkDevice
    uint64_t physicalDevice;    // VkPhysicalDevice
    uint64_t getDeviceProcAddr; // PFN_vkGetDeviceProcAddr
    uint32_t renderWidth;
    uint32_t renderHeight;
    uint32_t denoiser;          // 0 = REBLUR_DIFFUSE_SPECULAR, 1 = RELAX_DIFFUSE_SPECULAR
    uint32_t reserved;
} CausticaNrdCreateInfo;

typedef struct CausticaNrdFrameInfo {
    uint64_t commandBuffer;     // VkCommandBuffer, already recording
    uint32_t frameIndex;        // advances by exactly 1 per denoised frame
    float    jitterX, jitterY;  // render-pixel jitter applied to the primary ray this frame
    float    denoisingRange;
    float    viewToClip[16];
    float    clipToView[16];
    float    worldToView[16];
    float    viewToWorld[16];
    CausticaNrdImageView viewZ;                 // r32f, render res
    CausticaNrdImageView normalRoughness;       // render res
    CausticaNrdImageView motion;                // rg16f, render res
    CausticaNrdImageView diffRadianceHitDist;   // rgba16f, render res
    CausticaNrdImageView specRadianceHitDist;   // rgba16f, render res
    CausticaNrdImageView outDiffRadianceHitDist;
    CausticaNrdImageView outSpecRadianceHitDist;
    float worldPrevToWorld[16]; // previous world -> current world; all-zero = derive as documented
    float worldToViewPrev[16];  // current world -> PREVIOUS view; all-zero = use the cached previous matrix
} CausticaNrdFrameInfo;

namespace {

// Error codes returned by the ABI functions (only "0 == success" is contractual).
enum : int32_t {
    kErrArgument   = -1, // bad CausticaNrd* arguments or no instance created yet
    kErrVulkan     = -2, // a Vulkan object could not be created
    kErrNrd        = -3, // an nrd:: call failed
    kErrInternal   = -4, // C++ exception or an inconsistent NRD description
};

// Frames in flight the shim sizes its descriptor pools and constants ring for.
// The host must not have more than this many denoised frames recording or
// executing at the same time; 3 matches NRD's own integration default.
constexpr uint32_t kQueuedFrameNum = 3;

// The largest value "VkPhysicalDeviceLimits::minUniformBufferOffsetAlignment" is
// allowed to have, so it is always a legal alignment for a uniform buffer slice.
constexpr VkDeviceSize kConstantSliceAlignment = 256;

// Every barrier the shim records for NRD's own dispatches is compute -> compute
// (NRD's own integration does the same). The hand-off between the host's
// g-buffer/material passes and the NRD block is the caller's responsibility.
constexpr VkPipelineStageFlags kComputeStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
constexpr VkAccessFlags kReadAccess = VK_ACCESS_SHADER_READ_BIT;
constexpr VkAccessFlags kReadWriteAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

constexpr uint32_t kHostImageNum = 7;

// Which CausticaNrdFrameInfo image field serves each nrd::ResourceType REBLUR /
// RELAX consume. "nrd::ResourceType" values are not contiguous (IN_VIEWZ is 2,
// IN_DIFF_RADIANCE_HITDIST is 6, OUT_DIFF_RADIANCE_HITDIST is 18, ...), so the
// mapping is explicit. The indices match the hostImages[] array in
// caustica_nrd_denoise, which follows the ABI's field order.
struct HostResourceSlot {
    nrd::ResourceType type;
    uint32_t imageIndex;
};

const HostResourceSlot kHostResourceSlots[kHostImageNum] = {
    {nrd::ResourceType::IN_VIEWZ, 0},
    {nrd::ResourceType::IN_NORMAL_ROUGHNESS, 1},
    {nrd::ResourceType::IN_MV, 2},
    {nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, 3},
    {nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST, 4},
    {nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, 5},
    {nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, 6},
};

// -1 when the denoiser asked for a texture the ABI does not carry (confidence,
// disocclusion-threshold mix, validation, SH inputs, ...).
int32_t HostImageIndexOf(nrd::ResourceType type) {
    for (uint32_t i = 0; i < kHostImageNum; i++) {
        if (kHostResourceSlots[i].type == type)
            return static_cast<int32_t>(kHostResourceSlots[i].imageIndex);
    }
    return -1;
}

struct ImageState {
    VkImage image = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkAccessFlags access = 0;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
};

struct PendingImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkAccessFlags access = 0;
};

// NRD format -> VkFormat. Mirrors nvpro-samples/vk_denoise_nrd
// (RealtimeDenoiser/src/nrd_wrapper.cpp, Apache-2.0); the A2B10G10R10 /
// A8B8G8R8 entries are deliberately swizzled by NRD.
const VkFormat kNrdToVkFormat[] = {
    VK_FORMAT_R8_UNORM,
    VK_FORMAT_R8_SNORM,
    VK_FORMAT_R8_UINT,
    VK_FORMAT_R8_SINT,

    VK_FORMAT_R8G8_UNORM,
    VK_FORMAT_R8G8_SNORM,
    VK_FORMAT_R8G8_UINT,
    VK_FORMAT_R8G8_SINT,

    VK_FORMAT_R8G8B8A8_UNORM,
    VK_FORMAT_R8G8B8A8_SNORM,
    VK_FORMAT_A8B8G8R8_UINT_PACK32,
    VK_FORMAT_R8G8B8A8_SINT,
    VK_FORMAT_R8G8B8A8_SRGB,

    VK_FORMAT_R16_UNORM,
    VK_FORMAT_R16_SNORM,
    VK_FORMAT_R16_UINT,
    VK_FORMAT_R16_SINT,
    VK_FORMAT_R16_SFLOAT,

    VK_FORMAT_R16G16_UNORM,
    VK_FORMAT_R16G16_SNORM,
    VK_FORMAT_R16G16_UINT,
    VK_FORMAT_R16G16_SINT,
    VK_FORMAT_R16G16_SFLOAT,

    VK_FORMAT_R16G16B16A16_UNORM,
    VK_FORMAT_R16G16B16A16_SNORM,
    VK_FORMAT_R16G16B16A16_UINT,
    VK_FORMAT_R16G16B16A16_SINT,
    VK_FORMAT_R16G16B16A16_SFLOAT,

    VK_FORMAT_R32_UINT,
    VK_FORMAT_R32_SINT,
    VK_FORMAT_R32_SFLOAT,

    VK_FORMAT_R32G32_UINT,
    VK_FORMAT_R32G32_SINT,
    VK_FORMAT_R32G32_SFLOAT,

    VK_FORMAT_R32G32B32_UINT,
    VK_FORMAT_R32G32B32_SINT,
    VK_FORMAT_R32G32B32_SFLOAT,

    VK_FORMAT_R32G32B32A32_UINT,
    VK_FORMAT_R32G32B32A32_SINT,
    VK_FORMAT_R32G32B32A32_SFLOAT,

    VK_FORMAT_A2B10G10R10_UNORM_PACK32,
    VK_FORMAT_A2R10G10B10_UINT_PACK32,
    VK_FORMAT_B10G11R11_UFLOAT_PACK32,
    VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
};
static_assert(sizeof(kNrdToVkFormat) / sizeof(kNrdToVkFormat[0]) == size_t(nrd::Format::MAX_NUM),
              "NRD format table is out of sync with nrd::Format");

struct PendingDispatch {
    uint16_t pipelineIndex = 0;
    uint16_t gridWidth = 0;
    uint16_t gridHeight = 0;
    VkDescriptorSet set = VK_NULL_HANDLE;
    uint32_t barrierFirst = 0;
    uint32_t barrierCount = 0;
    uint32_t uboSlice = 0;
};

// ---------------------------------------------------------------------------
// Loaded device entry points
// ---------------------------------------------------------------------------
struct DeviceApi {
    PFN_vkDestroyPipeline vkDestroyPipeline = nullptr;
    PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = nullptr;
    PFN_vkResetDescriptorPool vkResetDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets = nullptr;
    PFN_vkCreateSampler vkCreateSampler = nullptr;
    PFN_vkDestroySampler vkDestroySampler = nullptr;
    PFN_vkCreateBuffer vkCreateBuffer = nullptr;
    PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
    PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
    PFN_vkCreateImage vkCreateImage = nullptr;
    PFN_vkDestroyImage vkDestroyImage = nullptr;
    PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements = nullptr;
    PFN_vkBindImageMemory vkBindImageMemory = nullptr;
    PFN_vkCreateImageView vkCreateImageView = nullptr;
    PFN_vkDestroyImageView vkDestroyImageView = nullptr;
    PFN_vkAllocateMemory vkAllocateMemory = nullptr;
    PFN_vkFreeMemory vkFreeMemory = nullptr;
    PFN_vkCreateShaderModule vkCreateShaderModule = nullptr;
    PFN_vkDestroyShaderModule vkDestroyShaderModule = nullptr;
    PFN_vkCreateComputePipelines vkCreateComputePipelines = nullptr;
    PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = nullptr;
    PFN_vkCreatePipelineLayout vkCreatePipelineLayout = nullptr;
    PFN_vkCreateDescriptorPool vkCreateDescriptorPool = nullptr;
    PFN_vkCmdBindPipeline vkCmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = nullptr;
    PFN_vkCmdDispatch vkCmdDispatch = nullptr;
    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;
    PFN_vkCmdUpdateBuffer vkCmdUpdateBuffer = nullptr;
};

// ---------------------------------------------------------------------------
// Instance
// ---------------------------------------------------------------------------
struct Instance {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr getDeviceProcAddr = nullptr;
    DeviceApi api;

    uint32_t width = 0;
    uint32_t height = 0;
    nrd::Denoiser denoiser = nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR;
    nrd::Identifier identifier = 0;

    nrd::Instance* nrd = nullptr;
    const nrd::InstanceDesc* instanceDesc = nullptr;

    // Descriptor geometry taken from InstanceDesc/LibraryDesc. Never hardcoded.
    uint32_t resourcesSpaceIndex = 0;
    uint32_t globalSpaceIndex = 1;
    uint32_t resourcesBaseRegisterIndex = 0;
    uint32_t samplerOffset = 0;
    uint32_t textureOffset = 0;
    uint32_t constantBufferOffset = 0;
    uint32_t storageTextureOffset = 0;
    uint32_t samplersNum = 0;
    uint32_t constantBufferRegisterIndex = 0;
    uint32_t perSetTexturesMaxNum = 0;
    uint32_t perSetStorageTexturesMaxNum = 0;
    uint32_t setsMaxNum = 0;
    uint32_t constantBufferMaxDataSize = 0;
    VkDeviceSize constantBufferSliceSize = 0;
    uint32_t constantsSlicesPerFrame = 0;

    std::vector<VkPipeline> pipelines;
    std::vector<nrd::PipelineDesc> pipelineDescs;

    VkDescriptorSetLayout resourcesSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout globalSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

    VkDescriptorPool framePools[kQueuedFrameNum] = {};
    VkDescriptorPool globalPool = VK_NULL_HANDLE;
    VkDescriptorSet globalSet = VK_NULL_HANDLE;

    VkSampler samplers[static_cast<size_t>(nrd::Sampler::MAX_NUM)] = {};

    VkBuffer constantBuffer = VK_NULL_HANDLE;
    VkDeviceMemory constantBufferMemory = VK_NULL_HANDLE;

    // Pool textures: [0, permanentPoolSize) permanent, then transient.
    std::vector<VkImage> poolImages;
    std::vector<VkDeviceMemory> poolImageMemories;
    std::vector<VkImageView> poolImageViews;
    std::vector<ImageState> poolImageStates;
    uint32_t permanentPoolSize = 0;

    // Previous-frame data derived by the shim (the ABI only carries the current frame).
    bool hasPrev = false;
    float prevViewToClip[16] = {};
    float prevWorldToView[16] = {};
    float prevJitter[2] = {};

    // Scratch reused by every denoise call.
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorImageInfo> imageInfos;
    std::vector<VkImage> writeImages;
    std::vector<PendingImage> pendingImages;
    std::vector<VkImageMemoryBarrier> barriers;
    std::vector<PendingDispatch> dispatches;
};

Instance* g_instance = nullptr;

// Declared here because caustica_nrd_create() reuses the teardown path.
void DestroyInstance(Instance* inst);

CAUSTICA_NRD_EXPORT void caustica_nrd_destroy(void);

void LogError(const char* what) {
    std::fprintf(stderr, "[caustica_nrd] %s\n", what);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
template <typename T>
bool LoadDeviceProc(Instance* inst, PFN_vkGetDeviceProcAddr gdpa, T& fn, const char* name) {
    fn = reinterpret_cast<T>(gdpa(inst->device, name));
    if (!fn) {
        std::fprintf(stderr, "[caustica_nrd] vkGetDeviceProcAddr(\"%s\") returned null\n", name);
        return false;
    }
    return true;
}

#define CAUSTICA_LOAD_DEVICE_PROC(inst, member, name) \
    do { \
        if (!LoadDeviceProc((inst), (inst)->getDeviceProcAddr, (inst)->api.member, name)) return false; \
    } while (0)

bool LoadDeviceApi(Instance* inst) {
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkDestroyPipeline, "vkDestroyPipeline");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkDestroyPipelineLayout, "vkDestroyPipelineLayout");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkDestroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkDestroyDescriptorPool, "vkDestroyDescriptorPool");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkResetDescriptorPool, "vkResetDescriptorPool");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkAllocateDescriptorSets, "vkAllocateDescriptorSets");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkUpdateDescriptorSets, "vkUpdateDescriptorSets");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkCreateSampler, "vkCreateSampler");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkDestroySampler, "vkDestroySampler");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkCreateBuffer, "vkCreateBuffer");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkDestroyBuffer, "vkDestroyBuffer");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkGetBufferMemoryRequirements, "vkGetBufferMemoryRequirements");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkBindBufferMemory, "vkBindBufferMemory");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkCreateImage, "vkCreateImage");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkDestroyImage, "vkDestroyImage");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkGetImageMemoryRequirements, "vkGetImageMemoryRequirements");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkBindImageMemory, "vkBindImageMemory");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkCreateImageView, "vkCreateImageView");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkDestroyImageView, "vkDestroyImageView");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkAllocateMemory, "vkAllocateMemory");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkFreeMemory, "vkFreeMemory");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkCreateShaderModule, "vkCreateShaderModule");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkDestroyShaderModule, "vkDestroyShaderModule");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkCreateComputePipelines, "vkCreateComputePipelines");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkCreateDescriptorSetLayout, "vkCreateDescriptorSetLayout");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkCreatePipelineLayout, "vkCreatePipelineLayout");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkCreateDescriptorPool, "vkCreateDescriptorPool");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkCmdBindPipeline, "vkCmdBindPipeline");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkCmdDispatch, "vkCmdDispatch");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
    CAUSTICA_LOAD_DEVICE_PROC(inst, vkCmdUpdateBuffer, "vkCmdUpdateBuffer");
    return true;
}

#undef CAUSTICA_LOAD_DEVICE_PROC

// Lowest memory type accepted by the resource. vkGetImage/BufferMemoryRequirements
// is the only device-level source of allocation information, and the lowest
// accepted bit is by construction a legal choice.
uint32_t PickMemoryType(uint32_t memoryTypeBits) {
    for (uint32_t i = 0; i < 32; i++) {
        if (memoryTypeBits & (1u << i))
            return i;
    }
    return 0;
}

VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

bool AllZero(const float* values, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        if (values[i] != 0.0f)
            return false;
    }
    return true;
}

// out = a * b for column-major matrices (m[column * 4 + row]).
void MultiplyColumnMajor(float* out, const float* a, const float* b) {
    float result[16];
    for (uint32_t column = 0; column < 4; column++) {
        for (uint32_t row = 0; row < 4; row++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < 4; k++)
                sum += a[k * 4 + row] * b[column * 4 + k];
            result[column * 4 + row] = sum;
        }
    }
    std::memcpy(out, result, sizeof(result));
}

const char* ResultName(nrd::Result result) {
    switch (result) {
        case nrd::Result::SUCCESS: return "SUCCESS";
        case nrd::Result::FAILURE: return "FAILURE";
        case nrd::Result::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case nrd::Result::UNSUPPORTED: return "UNSUPPORTED";
        case nrd::Result::NON_UNIQUE_IDENTIFIER: return "NON_UNIQUE_IDENTIFIER";
        default: return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// Teardown
//
// Every entry point is checked for null: teardown must survive a partially
// loaded DeviceApi (a failing create can return before all of them resolve).
// ---------------------------------------------------------------------------
void DestroyInstance(Instance* inst) {
    if (!inst)
        return;

    VkDevice device = inst->device;
    DeviceApi& api = inst->api;

    if (device) {
        for (size_t i = 0; i < inst->poolImages.size(); i++) {
            if (inst->poolImageViews[i] != VK_NULL_HANDLE && api.vkDestroyImageView)
                api.vkDestroyImageView(device, inst->poolImageViews[i], nullptr);
            if (inst->poolImages[i] != VK_NULL_HANDLE && api.vkDestroyImage)
                api.vkDestroyImage(device, inst->poolImages[i], nullptr);
            if (inst->poolImageMemories[i] != VK_NULL_HANDLE && api.vkFreeMemory)
                api.vkFreeMemory(device, inst->poolImageMemories[i], nullptr);
        }
        if (inst->constantBuffer != VK_NULL_HANDLE && api.vkDestroyBuffer)
            api.vkDestroyBuffer(device, inst->constantBuffer, nullptr);
        if (inst->constantBufferMemory != VK_NULL_HANDLE && api.vkFreeMemory)
            api.vkFreeMemory(device, inst->constantBufferMemory, nullptr);

        for (VkDescriptorPool pool : inst->framePools) {
            if (pool != VK_NULL_HANDLE && api.vkDestroyDescriptorPool)
                api.vkDestroyDescriptorPool(device, pool, nullptr);
        }
        if (inst->globalPool != VK_NULL_HANDLE && api.vkDestroyDescriptorPool)
            api.vkDestroyDescriptorPool(device, inst->globalPool, nullptr);

        for (VkPipeline pipeline : inst->pipelines) {
            if (pipeline != VK_NULL_HANDLE && api.vkDestroyPipeline)
                api.vkDestroyPipeline(device, pipeline, nullptr);
        }
        if (inst->pipelineLayout != VK_NULL_HANDLE && api.vkDestroyPipelineLayout)
            api.vkDestroyPipelineLayout(device, inst->pipelineLayout, nullptr);
        if (inst->resourcesSetLayout != VK_NULL_HANDLE && api.vkDestroyDescriptorSetLayout)
            api.vkDestroyDescriptorSetLayout(device, inst->resourcesSetLayout, nullptr);
        if (inst->globalSetLayout != VK_NULL_HANDLE && api.vkDestroyDescriptorSetLayout)
            api.vkDestroyDescriptorSetLayout(device, inst->globalSetLayout, nullptr);

        // Immutable samplers may be released after the layouts that embed them.
        for (VkSampler sampler : inst->samplers) {
            if (sampler != VK_NULL_HANDLE && api.vkDestroySampler)
                api.vkDestroySampler(device, sampler, nullptr);
        }
    }

    if (inst->nrd)
        nrd::DestroyInstance(*inst->nrd);

    delete inst;
}

// ---------------------------------------------------------------------------
// Creation steps
// ---------------------------------------------------------------------------
bool CreateSamplers(Instance* inst) {
    for (uint32_t i = 0; i < inst->samplersNum; i++) {
        const nrd::Sampler nrdSampler = inst->instanceDesc->samplers[i];

        VkSamplerCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = (nrdSampler == nrd::Sampler::NEAREST_CLAMP) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        info.minFilter = info.magFilter;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        // nrd::Sampler only has the two CLAMP variants.
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = info.addressModeU;
        info.addressModeW = info.addressModeU;
        info.minLod = 0.0f;
        info.maxLod = 0.0f; // NRD textures are single-mip
        info.maxAnisotropy = 1.0f;
        info.compareEnable = VK_FALSE;
        info.unnormalizedCoordinates = VK_FALSE;

        if (inst->api.vkCreateSampler(inst->device, &info, nullptr, &inst->samplers[i]) != VK_SUCCESS) {
            LogError("vkCreateSampler failed");
            return false;
        }
    }
    return true;
}

bool CreateDescriptorSetLayouts(Instance* inst) {
    // Set 0: resources. One binding per SRV / UAV slot NRD reports as its maxima.
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(inst->perSetTexturesMaxNum + inst->perSetStorageTexturesMaxNum);

    for (uint32_t i = 0; i < inst->perSetTexturesMaxNum; i++) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = inst->textureOffset + inst->resourcesBaseRegisterIndex + i;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(binding);
    }
    for (uint32_t i = 0; i < inst->perSetStorageTexturesMaxNum; i++) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = inst->storageTextureOffset + inst->resourcesBaseRegisterIndex + i;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(binding);
    }

    VkDescriptorSetLayoutCreateInfo resourcesInfo{};
    resourcesInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    resourcesInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    resourcesInfo.pBindings = bindings.empty() ? nullptr : bindings.data();
    if (inst->api.vkCreateDescriptorSetLayout(inst->device, &resourcesInfo, nullptr,
                                              &inst->resourcesSetLayout) != VK_SUCCESS) {
        LogError("vkCreateDescriptorSetLayout(resources) failed");
        return false;
    }

    // Set 1: immutable samplers + the constants uniform buffer.
    std::vector<VkDescriptorSetLayoutBinding> globalBindings(inst->samplersNum + 1);
    for (uint32_t i = 0; i < inst->samplersNum; i++) {
        globalBindings[i].binding = inst->samplerOffset + i;
        globalBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        globalBindings[i].descriptorCount = 1;
        globalBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        globalBindings[i].pImmutableSamplers = &inst->samplers[i];
    }
    globalBindings[inst->samplersNum].binding = inst->constantBufferOffset + inst->constantBufferRegisterIndex;
    globalBindings[inst->samplersNum].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    globalBindings[inst->samplersNum].descriptorCount = 1;
    globalBindings[inst->samplersNum].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo globalInfo{};
    globalInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    globalInfo.bindingCount = static_cast<uint32_t>(globalBindings.size());
    globalInfo.pBindings = globalBindings.data();
    if (inst->api.vkCreateDescriptorSetLayout(inst->device, &globalInfo, nullptr,
                                              &inst->globalSetLayout) != VK_SUCCESS) {
        LogError("vkCreateDescriptorSetLayout(constants/samplers) failed");
        return false;
    }

    // One shared pipeline layout: 2 sets, no push constants (NRD passes constants
    // through a uniform buffer, the SPIR-V has no push-constant block).
    const VkDescriptorSetLayout setLayouts[2] = {inst->resourcesSetLayout, inst->globalSetLayout};
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 2;
    layoutInfo.pSetLayouts = setLayouts;
    layoutInfo.pushConstantRangeCount = 0;
    if (inst->api.vkCreatePipelineLayout(inst->device, &layoutInfo, nullptr, &inst->pipelineLayout) != VK_SUCCESS) {
        LogError("vkCreatePipelineLayout failed");
        return false;
    }
    return true;
}

bool CreatePipelines(Instance* inst) {
    const uint32_t pipelinesNum = inst->instanceDesc->pipelinesNum;
    inst->pipelines.assign(pipelinesNum, VK_NULL_HANDLE);
    inst->pipelineDescs.assign(inst->instanceDesc->pipelines, inst->instanceDesc->pipelines + pipelinesNum);

    for (uint32_t i = 0; i < pipelinesNum; i++) {
        const nrd::PipelineDesc& pipelineDesc = inst->pipelineDescs[i];

        if (!pipelineDesc.computeShaderSPIRV.bytecode || pipelineDesc.computeShaderSPIRV.size == 0) {
            std::fprintf(stderr,
                         "[caustica_nrd] pipeline %u ('%s') carries no SPIR-V; NRD must be built with "
                         "NRD_EMBEDS_SPIRV_SHADERS=ON\n",
                         i, pipelineDesc.shaderIdentifier);
            return false;
        }
        if ((pipelineDesc.computeShaderSPIRV.size % 4) != 0) {
            LogError("SPIR-V blob size is not a multiple of 4");
            return false;
        }

        VkShaderModuleCreateInfo moduleInfo{};
        moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        moduleInfo.codeSize = static_cast<size_t>(pipelineDesc.computeShaderSPIRV.size);
        moduleInfo.pCode = static_cast<const uint32_t*>(pipelineDesc.computeShaderSPIRV.bytecode);
        VkShaderModule module = VK_NULL_HANDLE;
        if (inst->api.vkCreateShaderModule(inst->device, &moduleInfo, nullptr, &module) != VK_SUCCESS) {
            std::fprintf(stderr, "[caustica_nrd] vkCreateShaderModule failed for '%s'\n",
                         pipelineDesc.shaderIdentifier);
            return false;
        }

        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = module;
        stageInfo.pName = inst->instanceDesc->shaderEntryPoint;

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = inst->pipelineLayout;

        const VkResult result = inst->api.vkCreateComputePipelines(inst->device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                                                   nullptr, &inst->pipelines[i]);
        inst->api.vkDestroyShaderModule(inst->device, module, nullptr);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr, "[caustica_nrd] vkCreateComputePipelines failed (%d) for '%s'\n",
                         static_cast<int>(result), pipelineDesc.shaderIdentifier);
            return false;
        }
    }
    return true;
}

bool CreateConstantBuffer(Instance* inst) {
    inst->constantBufferSliceSize = AlignUp(inst->constantBufferMaxDataSize, kConstantSliceAlignment);
    inst->constantsSlicesPerFrame = inst->setsMaxNum;
    const VkDeviceSize totalSize = inst->constantBufferSliceSize * inst->constantsSlicesPerFrame * kQueuedFrameNum;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = totalSize;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (inst->api.vkCreateBuffer(inst->device, &bufferInfo, nullptr, &inst->constantBuffer) != VK_SUCCESS) {
        LogError("vkCreateBuffer(constants) failed");
        return false;
    }

    VkMemoryRequirements requirements{};
    inst->api.vkGetBufferMemoryRequirements(inst->device, inst->constantBuffer, &requirements);

    VkMemoryAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = requirements.size;
    allocateInfo.memoryTypeIndex = PickMemoryType(requirements.memoryTypeBits);
    if (inst->api.vkAllocateMemory(inst->device, &allocateInfo, nullptr, &inst->constantBufferMemory) != VK_SUCCESS) {
        LogError("vkAllocateMemory(constants) failed");
        return false;
    }
    if (inst->api.vkBindBufferMemory(inst->device, inst->constantBuffer, inst->constantBufferMemory, 0) != VK_SUCCESS) {
        LogError("vkBindBufferMemory(constants) failed");
        return false;
    }
    return true;
}

bool CreateDescriptorPools(Instance* inst) {
    // The global set is written once and never reset.
    const VkDescriptorPoolSize globalSizes[2] = {
        {VK_DESCRIPTOR_TYPE_SAMPLER, inst->samplersNum},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1},
    };
    VkDescriptorPoolCreateInfo globalInfo{};
    globalInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    globalInfo.maxSets = 1;
    globalInfo.poolSizeCount = 2;
    globalInfo.pPoolSizes = globalSizes;
    if (inst->api.vkCreateDescriptorPool(inst->device, &globalInfo, nullptr, &inst->globalPool) != VK_SUCCESS) {
        LogError("vkCreateDescriptorPool(constants/samplers) failed");
        return false;
    }

    VkDescriptorSetAllocateInfo setInfo{};
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setInfo.descriptorPool = inst->globalPool;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts = &inst->globalSetLayout;
    if (inst->api.vkAllocateDescriptorSets(inst->device, &setInfo, &inst->globalSet) != VK_SUCCESS) {
        LogError("vkAllocateDescriptorSets(constants/samplers) failed");
        return false;
    }

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = inst->constantBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = inst->constantBufferSliceSize;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = inst->globalSet;
    write.dstBinding = inst->constantBufferOffset + inst->constantBufferRegisterIndex;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    write.pBufferInfo = &bufferInfo;
    // Immutable samplers are part of the layout, so only the constants need a write.
    inst->api.vkUpdateDescriptorSets(inst->device, 1, &write, 0, nullptr);

    // Per-frame pools, reset (not recreated) before each reuse.
    const VkDescriptorPoolSize frameSizes[3] = {
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, inst->setsMaxNum * inst->perSetTexturesMaxNum},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, inst->setsMaxNum * inst->perSetStorageTexturesMaxNum},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, inst->setsMaxNum},
    };
    for (uint32_t i = 0; i < kQueuedFrameNum; i++) {
        VkDescriptorPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        info.maxSets = inst->setsMaxNum;
        info.poolSizeCount = 3;
        info.pPoolSizes = frameSizes;
        if (inst->api.vkCreateDescriptorPool(inst->device, &info, nullptr, &inst->framePools[i]) != VK_SUCCESS) {
            LogError("vkCreateDescriptorPool(frame) failed");
            return false;
        }
    }
    return true;
}

bool CreatePoolTexture(Instance* inst, uint32_t width, uint32_t height, const nrd::TextureDesc& desc,
                       VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
    const VkFormat format = kNrdToVkFormat[static_cast<uint32_t>(desc.format)];
    const uint32_t downsample = desc.downsampleFactor ? desc.downsampleFactor : 1;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent.width = (width + downsample - 1) / downsample;
    imageInfo.extent.height = (height + downsample - 1) / downsample;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;   // NRD textures are strictly 2D, 1 layer, 1 mip
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (imageInfo.extent.width == 0 || imageInfo.extent.height == 0) {
        LogError("NRD pool texture has zero extent");
        return false;
    }
    if (inst->api.vkCreateImage(inst->device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        std::fprintf(stderr, "[caustica_nrd] vkCreateImage failed (format %d, %ux%u)\n", static_cast<int>(format),
                     imageInfo.extent.width, imageInfo.extent.height);
        return false;
    }

    VkMemoryRequirements requirements{};
    inst->api.vkGetImageMemoryRequirements(inst->device, image, &requirements);

    VkMemoryAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = requirements.size;
    allocateInfo.memoryTypeIndex = PickMemoryType(requirements.memoryTypeBits);
    if (inst->api.vkAllocateMemory(inst->device, &allocateInfo, nullptr, &memory) != VK_SUCCESS) {
        LogError("vkAllocateMemory(pool texture) failed");
        return false;
    }
    if (inst->api.vkBindImageMemory(inst->device, image, memory, 0) != VK_SUCCESS) {
        LogError("vkBindImageMemory(pool texture) failed");
        return false;
    }

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    if (inst->api.vkCreateImageView(inst->device, &viewInfo, nullptr, &view) != VK_SUCCESS) {
        LogError("vkCreateImageView(pool texture) failed");
        return false;
    }
    return true;
}

bool CreatePoolTextures(Instance* inst) {
    const uint32_t permanentNum = inst->instanceDesc->permanentPoolSize;
    const uint32_t transientNum = inst->instanceDesc->transientPoolSize;
    const uint32_t total = permanentNum + transientNum;

    inst->permanentPoolSize = permanentNum;
    inst->poolImages.assign(total, VK_NULL_HANDLE);
    inst->poolImageMemories.assign(total, VK_NULL_HANDLE);
    inst->poolImageViews.assign(total, VK_NULL_HANDLE);
    inst->poolImageStates.assign(total, ImageState{});

    for (uint32_t i = 0; i < total; i++) {
        const nrd::TextureDesc& desc =
            (i < permanentNum) ? inst->instanceDesc->permanentPool[i] : inst->instanceDesc->transientPool[i - permanentNum];
        if (!CreatePoolTexture(inst, inst->width, inst->height, desc, inst->poolImages[i],
                               inst->poolImageMemories[i], inst->poolImageViews[i])) {
            return false;
        }
        inst->poolImageStates[i].image = inst->poolImages[i];
    }
    return true;
}

bool ApplyDenoiserSettings(Instance* inst) {
    nrd::Result result = nrd::Result::FAILURE;
    if (inst->denoiser == nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR) {
        // NRD's defaults, including hitDistanceParameters {A=3.0, B=0.1, C=20.0} which
        // the front-end must mirror through REBLUR_FrontEnd_GetNormHitDist().
        nrd::ReblurSettings settings{};
        result = nrd::SetDenoiserSettings(*inst->nrd, inst->identifier, &settings);
    } else {
        nrd::RelaxSettings settings{};
        result = nrd::SetDenoiserSettings(*inst->nrd, inst->identifier, &settings);
    }
    if (result != nrd::Result::SUCCESS) {
        std::fprintf(stderr, "[caustica_nrd] nrd::SetDenoiserSettings failed: %s\n", ResultName(result));
        return false;
    }
    return true;
}

bool CreateAll(Instance* inst) {
    if (!LoadDeviceApi(inst))
        return false;

    const nrd::LibraryDesc* libraryDesc = nrd::GetLibraryDesc();
    if (!libraryDesc) {
        LogError("nrd::GetLibraryDesc returned null");
        return false;
    }
    if (libraryDesc->versionMajor != NRD_VERSION_MAJOR || libraryDesc->versionMinor != NRD_VERSION_MINOR ||
        libraryDesc->versionBuild != NRD_VERSION_BUILD) {
        std::fprintf(stderr,
                     "[caustica_nrd] NRD library version mismatch: library %u.%u.%u, headers %u.%u.%u "
                     "(ReblurSettings/RelaxSettings are memcpy'd by size, so they must match)\n",
                     libraryDesc->versionMajor, libraryDesc->versionMinor, libraryDesc->versionBuild,
                     NRD_VERSION_MAJOR, NRD_VERSION_MINOR, NRD_VERSION_BUILD);
        return false;
    }

    // The host's Slang shaders write IN_NORMAL_ROUGHNESS (and REBLUR's front-end
    // normalization) against fixed encodings, so the linked library must have been
    // built with exactly these. Both values are always reported.
    std::fprintf(stderr,
                 "[caustica_nrd] NRD v%u.%u.%u: normalEncoding=%u (expected %d, 0 = RGBA8_UNORM), "
                 "roughnessEncoding=%u (expected %d, 1 = LINEAR)\n",
                 libraryDesc->versionMajor, libraryDesc->versionMinor, libraryDesc->versionBuild,
                 static_cast<uint32_t>(libraryDesc->normalEncoding), CAUSTICA_NRD_EXPECTED_NORMAL_ENCODING,
                 static_cast<uint32_t>(libraryDesc->roughnessEncoding), CAUSTICA_NRD_EXPECTED_ROUGHNESS_ENCODING);
    if (static_cast<uint32_t>(libraryDesc->normalEncoding) != CAUSTICA_NRD_EXPECTED_NORMAL_ENCODING ||
        static_cast<uint32_t>(libraryDesc->roughnessEncoding) != CAUSTICA_NRD_EXPECTED_ROUGHNESS_ENCODING) {
        LogError("NRD encoding mismatch: rebuild NRD with the encodings this shim pins in CMakeLists.txt");
        return false;
    }

    const nrd::DenoiserDesc denoiserDesc = {inst->identifier, inst->denoiser};
    nrd::InstanceCreationDesc creationDesc{};
    creationDesc.denoisers = &denoiserDesc;
    creationDesc.denoisersNum = 1;

    nrd::Result result = nrd::CreateInstance(creationDesc, inst->nrd);
    if (result != nrd::Result::SUCCESS || !inst->nrd) {
        std::fprintf(stderr, "[caustica_nrd] nrd::CreateInstance failed: %s\n", ResultName(result));
        return false;
    }

    inst->instanceDesc = nrd::GetInstanceDesc(*inst->nrd);
    if (!inst->instanceDesc) {
        LogError("nrd::GetInstanceDesc returned null");
        return false;
    }

    const nrd::InstanceDesc* desc = inst->instanceDesc;
    inst->resourcesSpaceIndex = desc->resourcesSpaceIndex;
    inst->globalSpaceIndex = desc->constantBufferAndSamplersSpaceIndex;
    inst->resourcesBaseRegisterIndex = desc->resourcesBaseRegisterIndex;
    inst->samplerOffset = libraryDesc->spirvBindingOffsets.samplerOffset;
    inst->textureOffset = libraryDesc->spirvBindingOffsets.textureOffset;
    inst->constantBufferOffset = libraryDesc->spirvBindingOffsets.constantBufferOffset;
    inst->storageTextureOffset = libraryDesc->spirvBindingOffsets.storageTextureAndBufferOffset;
    inst->samplersNum = desc->samplersNum;
    inst->constantBufferRegisterIndex = desc->constantBufferRegisterIndex;
    inst->perSetTexturesMaxNum = desc->descriptorPoolDesc.perSetTexturesMaxNum;
    inst->perSetStorageTexturesMaxNum = desc->descriptorPoolDesc.perSetStorageTexturesMaxNum;
    inst->setsMaxNum = desc->descriptorPoolDesc.setsMaxNum;
    inst->constantBufferMaxDataSize = desc->constantBufferMaxDataSize;

    if (inst->resourcesSpaceIndex != 0 || inst->globalSpaceIndex != 1) {
        std::fprintf(stderr, "[caustica_nrd] unexpected NRD set indices (resources=%u, constants=%u)\n",
                     inst->resourcesSpaceIndex, inst->globalSpaceIndex);
        return false;
    }
    if (inst->samplersNum != 2 || desc->pipelinesNum == 0 || !desc->shaderEntryPoint || inst->setsMaxNum == 0 ||
        inst->constantBufferMaxDataSize == 0) {
        LogError("unexpected NRD instance description");
        return false;
    }
    if (inst->perSetTexturesMaxNum > 64 || inst->perSetStorageTexturesMaxNum > 64) {
        LogError("unexpected NRD descriptor maxima");
        return false;
    }

    // Scratch buffers sized for the worst case NRD reports.
    const uint32_t maxDescriptors = inst->perSetTexturesMaxNum + inst->perSetStorageTexturesMaxNum;
    inst->writes.resize(maxDescriptors);
    inst->imageInfos.resize(maxDescriptors);
    inst->writeImages.resize(maxDescriptors);
    inst->pendingImages.resize(maxDescriptors);
    inst->barriers.reserve(static_cast<size_t>(desc->pipelinesNum) * 4);
    inst->dispatches.reserve(inst->setsMaxNum);

    if (!CreateConstantBuffer(inst))
        return false;

    // The constants are written from the CPU with vkCmdUpdateBuffer, so no
    // persistent mapping is required or taken.

    if (!CreateSamplers(inst))
        return false;
    if (!CreateDescriptorSetLayouts(inst))
        return false;
    if (!CreatePipelines(inst))
        return false;
    if (!CreateDescriptorPools(inst))
        return false;
    if (!CreatePoolTextures(inst))
        return false;
    if (!ApplyDenoiserSettings(inst))
        return false;

    return true;
}

// ---------------------------------------------------------------------------
// Per-frame helpers
// ---------------------------------------------------------------------------
ImageState* FindImageState(Instance* inst, ImageState* hostStates, VkImage image) {
    for (uint32_t i = 0; i < kHostImageNum; i++) {
        if (hostStates[i].image == image)
            return &hostStates[i];
    }
    for (ImageState& state : inst->poolImageStates) {
        if (state.image == image)
            return &state;
    }
    return nullptr;
}

void AppendBarrier(Instance* inst, VkImage image, VkImageLayout oldLayout, VkAccessFlags srcAccess,
                   VkImageLayout newLayout, VkAccessFlags dstAccess) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    // Every barrier is recorded as compute -> compute, so the batch's source
    // stage is COMPUTE and only the access masks differ. For an image whose
    // tracked layout is UNDEFINED the source access is empty: the contents are
    // discarded by the transition.
    barrier.srcAccessMask = (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) ? 0 : srcAccess;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    inst->barriers.push_back(barrier);
}

} // namespace

// ---------------------------------------------------------------------------
// ABI
// ---------------------------------------------------------------------------
CAUSTICA_NRD_EXPORT int32_t caustica_nrd_create(const CausticaNrdCreateInfo* info) {
    if (!info)
        return kErrArgument;
    if (!info->device || !info->physicalDevice || !info->getDeviceProcAddr)
        return kErrArgument;
    if (info->renderWidth == 0 || info->renderHeight == 0 || info->renderWidth > 0xFFFFu || info->renderHeight > 0xFFFFu)
        return kErrArgument;
    if (info->denoiser > 1)
        return kErrArgument;

    // A create on top of a live instance is the resize path: release the old one.
    caustica_nrd_destroy();

    Instance* inst = new (std::nothrow) Instance();
    if (!inst)
        return kErrInternal;

    inst->device = reinterpret_cast<VkDevice>(info->device);
    inst->physicalDevice = reinterpret_cast<VkPhysicalDevice>(info->physicalDevice);
    inst->getDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(info->getDeviceProcAddr);
    inst->width = info->renderWidth;
    inst->height = info->renderHeight;
    inst->denoiser = (info->denoiser == 0) ? nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR : nrd::Denoiser::RELAX_DIFFUSE_SPECULAR;
    inst->identifier = static_cast<nrd::Identifier>(inst->denoiser);

    int32_t status = CAUSTICA_NRD_OK;
    try {
        if (!CreateAll(inst))
            status = kErrVulkan;
    } catch (const std::exception&) {
        status = kErrInternal;
    } catch (...) {
        status = kErrInternal;
    }

    if (status != CAUSTICA_NRD_OK) {
        DestroyInstance(inst);
        return status;
    }
    g_instance = inst;
    return CAUSTICA_NRD_OK;
}

CAUSTICA_NRD_EXPORT int32_t caustica_nrd_denoise(const CausticaNrdFrameInfo* frame) {
    Instance* inst = g_instance;
    if (!inst || !frame)
        return kErrArgument;
    if (!frame->commandBuffer)
        return kErrArgument;

    try {
        const VkCommandBuffer commandBuffer = reinterpret_cast<VkCommandBuffer>(frame->commandBuffer);
        const CausticaNrdImageView* hostImages[kHostImageNum] = {
            &frame->viewZ, &frame->normalRoughness, &frame->motion, &frame->diffRadianceHitDist,
            &frame->specRadianceHitDist, &frame->outDiffRadianceHitDist, &frame->outSpecRadianceHitDist,
        };
        for (uint32_t i = 0; i < kHostImageNum; i++) {
            if (!hostImages[i]->image || !hostImages[i]->view) {
                std::fprintf(stderr, "[caustica_nrd] frame image %u is null\n", i);
                return kErrArgument;
            }
        }
        if (!(frame->denoisingRange > 0.0f)) {
            LogError("denoisingRange must be > 0");
            return kErrArgument;
        }
        if (AllZero(frame->viewToClip, 16) || AllZero(frame->worldToView, 16)) {
            LogError("viewToClip/worldToView must be filled in");
            return kErrArgument;
        }

        // ---- common settings -------------------------------------------------
        nrd::CommonSettings commonSettings{};
        std::memcpy(commonSettings.viewToClipMatrix, frame->viewToClip, sizeof(commonSettings.viewToClipMatrix));
        std::memcpy(commonSettings.worldToViewMatrix, frame->worldToView, sizeof(commonSettings.worldToViewMatrix));
        if (!AllZero(frame->worldPrevToWorld, 16)) {
            // The host supplies it (camera-relative worlds move the origin, so the
            // matrix cannot be recovered from the two view matrices).
            std::memcpy(commonSettings.worldPrevToWorldMatrix, frame->worldPrevToWorld,
                        sizeof(commonSettings.worldPrevToWorldMatrix));
        } else if (inst->hasPrev) {
            // Fallback. The ABI carries no previous-frame matrices, so previous
            // world -> current world is derived: viewToWorld(current) *
            // worldToView(previous).
            MultiplyColumnMajor(commonSettings.worldPrevToWorldMatrix, frame->viewToWorld, inst->prevWorldToView);
        }
        if (inst->hasPrev) {
            std::memcpy(commonSettings.viewToClipMatrixPrev, inst->prevViewToClip,
                        sizeof(commonSettings.viewToClipMatrixPrev));
            // NRD requires the previous world->view matrix to carry the camera delta when the world
            // coordinate system moves with the camera, which is the case for this renderer. The host
            // therefore supplies it directly; the cached previous matrix is only a fallback for hosts
            // whose world space is stable.
            if (!AllZero(frame->worldToViewPrev, 16)) {
                std::memcpy(commonSettings.worldToViewMatrixPrev, frame->worldToViewPrev,
                            sizeof(commonSettings.worldToViewMatrixPrev));
            } else {
                std::memcpy(commonSettings.worldToViewMatrixPrev, inst->prevWorldToView,
                            sizeof(commonSettings.worldToViewMatrixPrev));
            }
            commonSettings.cameraJitterPrev[0] = inst->prevJitter[0];
            commonSettings.cameraJitterPrev[1] = inst->prevJitter[1];
        }
        commonSettings.cameraJitter[0] = frame->jitterX;
        commonSettings.cameraJitter[1] = frame->jitterY;
        // The ABI's motion texture is in render pixels ("previous - current").
        commonSettings.motionVectorScale[0] = 1.0f / static_cast<float>(inst->width);
        commonSettings.motionVectorScale[1] = 1.0f / static_cast<float>(inst->height);
        commonSettings.motionVectorScale[2] = 0.0f;
        commonSettings.resourceSize[0] = static_cast<uint16_t>(inst->width);
        commonSettings.resourceSize[1] = static_cast<uint16_t>(inst->height);
        commonSettings.resourceSizePrev[0] = commonSettings.resourceSize[0];
        commonSettings.resourceSizePrev[1] = commonSettings.resourceSize[1];
        commonSettings.rectSize[0] = commonSettings.resourceSize[0];
        commonSettings.rectSize[1] = commonSettings.resourceSize[1];
        commonSettings.rectSizePrev[0] = commonSettings.rectSize[0];
        commonSettings.rectSizePrev[1] = commonSettings.rectSize[1];
        commonSettings.viewZScale = 1.0f; // IN_VIEWZ holds linear view-space depth
        commonSettings.denoisingRange = frame->denoisingRange;
        commonSettings.frameIndex = frame->frameIndex;
        // frameIndex == 0 restarts the sequence: reset history and drop the
        // previous-frame data the shim tracks (NRD forces the same mode on the
        // first SetCommonSettings after CreateInstance).
        commonSettings.accumulationMode = (frame->frameIndex == 0) ? nrd::AccumulationMode::CLEAR_AND_RESTART
                                                                   : nrd::AccumulationMode::CONTINUE;

        nrd::Result result = nrd::SetCommonSettings(*inst->nrd, commonSettings);
        if (result != nrd::Result::SUCCESS) {
            std::fprintf(stderr, "[caustica_nrd] nrd::SetCommonSettings failed: %s\n", ResultName(result));
            return kErrNrd;
        }

        const nrd::DispatchDesc* dispatchDescs = nullptr;
        uint32_t dispatchDescsNum = 0;
        result = nrd::GetComputeDispatches(*inst->nrd, &inst->identifier, 1, dispatchDescs, dispatchDescsNum);
        if (result != nrd::Result::SUCCESS) {
            std::fprintf(stderr, "[caustica_nrd] nrd::GetComputeDispatches failed: %s\n", ResultName(result));
            return kErrNrd;
        }
        if (dispatchDescsNum == 0) {
            LogError("nrd::GetComputeDispatches produced no dispatches");
            return kErrNrd;
        }

        // ---- frame state -----------------------------------------------------
        // NRD bound host textures are contractually in VK_IMAGE_LAYOUT_GENERAL on
        // entry (they are produced by storage-image writes) and are restored to it
        // before this function returns.
        ImageState hostStates[kHostImageNum];
        for (uint32_t i = 0; i < kHostImageNum; i++) {
            ImageState state;
            state.image = reinterpret_cast<VkImage>(hostImages[i]->image);
            state.layout = VK_IMAGE_LAYOUT_GENERAL;
            state.access = kReadWriteAccess;
            state.stage = kComputeStage;
            hostStates[i] = state;
        }

        const uint32_t frameSlot = frame->frameIndex % kQueuedFrameNum;
        VkDescriptorPool framePool = inst->framePools[frameSlot];
        if (inst->api.vkResetDescriptorPool(inst->device, framePool, 0) != VK_SUCCESS) {
            LogError("vkResetDescriptorPool failed");
            return kErrVulkan;
        }

        inst->barriers.clear();
        inst->dispatches.clear();

        // ---- pass 1: descriptors, barriers (as data), constants -------------
        uint32_t uboCursor = 0;
        uint32_t lastSlice = 0;
        bool lastSliceValid = false;
        bool anyConstantsUpdate = false;

        for (uint32_t dispatchIndex = 0; dispatchIndex < dispatchDescsNum; dispatchIndex++) {
            const nrd::DispatchDesc& dispatch = dispatchDescs[dispatchIndex];
            if (dispatch.pipelineIndex >= inst->pipelineDescs.size()) {
                LogError("dispatch references an unknown pipeline");
                return kErrInternal;
            }
            const nrd::PipelineDesc& pipeline = inst->pipelineDescs[dispatch.pipelineIndex];

            uint32_t descriptorsNum = 0;
            for (uint32_t r = 0; r < pipeline.resourceRangesNum; r++)
                descriptorsNum += pipeline.resourceRanges[r].descriptorsNum;
            if (descriptorsNum > inst->writes.size() || descriptorsNum != dispatch.resourcesNum) {
                LogError("dispatch resource count does not match its pipeline's ranges");
                return kErrInternal;
            }

            VkDescriptorSetAllocateInfo allocateInfo{};
            allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocateInfo.descriptorPool = framePool;
            allocateInfo.descriptorSetCount = 1;
            allocateInfo.pSetLayouts = &inst->resourcesSetLayout;
            VkDescriptorSet set = VK_NULL_HANDLE;
            const VkResult allocateResult = inst->api.vkAllocateDescriptorSets(inst->device, &allocateInfo, &set);
            if (allocateResult != VK_SUCCESS) {
                std::fprintf(stderr,
                             "[caustica_nrd] vkAllocateDescriptorSets failed (%d) on dispatch %u; the pool holds "
                             "%u sets\n",
                             static_cast<int>(allocateResult), dispatchIndex, inst->setsMaxNum);
                return kErrVulkan;
            }

            // Aggregate the wanted layout per image: a storage use wins, because
            // VK_IMAGE_LAYOUT_GENERAL serves sampled and storage descriptors alike.
            uint32_t pendingNum = 0;
            uint32_t resourceIndex = 0;
            for (uint32_t r = 0; r < pipeline.resourceRangesNum; r++) {
                const nrd::ResourceRangeDesc& range = pipeline.resourceRanges[r];
                const bool isStorage = range.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE;

                for (uint32_t k = 0; k < range.descriptorsNum; k++, resourceIndex++) {
                    const nrd::ResourceDesc& resource = dispatch.resources[resourceIndex];

                    VkImage image = VK_NULL_HANDLE;
                    VkImageView view = VK_NULL_HANDLE;
                    if (resource.type == nrd::ResourceType::PERMANENT_POOL) {
                        if (resource.indexInPool >= inst->permanentPoolSize) {
                            LogError("permanent pool index out of range");
                            return kErrInternal;
                        }
                        image = inst->poolImages[resource.indexInPool];
                        view = inst->poolImageViews[resource.indexInPool];
                    } else if (resource.type == nrd::ResourceType::TRANSIENT_POOL) {
                        const uint32_t index = inst->permanentPoolSize + resource.indexInPool;
                        if (index >= inst->poolImages.size()) {
                            LogError("transient pool index out of range");
                            return kErrInternal;
                        }
                        image = inst->poolImages[index];
                        view = inst->poolImageViews[index];
                    } else {
                        const int32_t hostIndex = HostImageIndexOf(resource.type);
                        if (hostIndex < 0) {
                            std::fprintf(stderr,
                                         "[caustica_nrd] NRD requested resource type %u ('%s'), which this ABI does "
                                         "not bind\n",
                                         static_cast<uint32_t>(resource.type),
                                         nrd::GetResourceTypeString(resource.type));
                            return kErrInternal;
                        }
                        image = reinterpret_cast<VkImage>(hostImages[hostIndex]->image);
                        view = reinterpret_cast<VkImageView>(hostImages[hostIndex]->view);
                    }

                    const VkImageLayout wantedLayout =
                        isStorage ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    const VkAccessFlags wantedAccess = isStorage ? kReadWriteAccess : kReadAccess;

                    uint32_t pendingIndex = 0;
                    for (; pendingIndex < pendingNum; pendingIndex++) {
                        if (inst->pendingImages[pendingIndex].image == image)
                            break;
                    }
                    if (pendingIndex == pendingNum) {
                        inst->pendingImages[pendingNum].image = image;
                        inst->pendingImages[pendingNum].layout = wantedLayout;
                        inst->pendingImages[pendingNum].access = wantedAccess;
                        pendingNum++;
                    } else if (isStorage) {
                        inst->pendingImages[pendingIndex].layout = VK_IMAGE_LAYOUT_GENERAL;
                        inst->pendingImages[pendingIndex].access = kReadWriteAccess;
                    }

                    VkWriteDescriptorSet& write = inst->writes[resourceIndex];
                    write = VkWriteDescriptorSet{};
                    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    write.dstSet = set;
                    write.dstBinding = (isStorage ? inst->storageTextureOffset : inst->textureOffset) +
                                       inst->resourcesBaseRegisterIndex + k;
                    write.descriptorCount = 1;
                    write.descriptorType = isStorage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                     : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    write.pImageInfo = &inst->imageInfos[resourceIndex];

                    inst->imageInfos[resourceIndex] = VkDescriptorImageInfo{};
                    inst->imageInfos[resourceIndex].sampler = VK_NULL_HANDLE;
                    inst->imageInfos[resourceIndex].imageView = view;
                    inst->writeImages[resourceIndex] = image;
                }
            }

            // Descriptor layouts must match the layout each image is in when the
            // dispatch executes, i.e. the aggregated one.
            for (uint32_t i = 0; i < descriptorsNum; i++) {
                for (uint32_t p = 0; p < pendingNum; p++) {
                    if (inst->pendingImages[p].image == inst->writeImages[i]) {
                        inst->imageInfos[i].imageLayout = inst->pendingImages[p].layout;
                        break;
                    }
                }
            }

            inst->api.vkUpdateDescriptorSets(inst->device, descriptorsNum, inst->writes.data(), 0, nullptr);

            // Record the barriers this dispatch needs, in dispatch order.
            PendingDispatch pendingDispatch{};
            pendingDispatch.pipelineIndex = dispatch.pipelineIndex;
            pendingDispatch.gridWidth = dispatch.gridWidth;
            pendingDispatch.gridHeight = dispatch.gridHeight;
            pendingDispatch.set = set;
            pendingDispatch.barrierFirst = static_cast<uint32_t>(inst->barriers.size());

            for (uint32_t p = 0; p < pendingNum; p++) {
                const PendingImage& wanted = inst->pendingImages[p];
                ImageState* state = FindImageState(inst, hostStates, wanted.image);
                if (!state) {
                    LogError("resource is not tracked by the shim");
                    return kErrInternal;
                }
                // A UAV reuse always needs a barrier, even when the layout does not
                // change: NRD reads and writes the same history textures across
                // consecutive passes.
                const bool needsBarrier =
                    state->layout != wanted.layout || wanted.layout == VK_IMAGE_LAYOUT_GENERAL;
                if (needsBarrier) {
                    AppendBarrier(inst, wanted.image, state->layout, state->access, wanted.layout, wanted.access);
                    state->layout = wanted.layout;
                    state->access = wanted.access;
                    state->stage = kComputeStage;
                }
            }
            pendingDispatch.barrierCount = static_cast<uint32_t>(inst->barriers.size()) - pendingDispatch.barrierFirst;

            // Constants slice for this dispatch.
            if (dispatch.constantBufferDataSize > 0 && dispatch.constantBufferData) {
                const bool reuse = dispatch.constantBufferDataMatchesPreviousDispatch && lastSliceValid;
                if (reuse) {
                    pendingDispatch.uboSlice = lastSlice;
                } else {
                    if (uboCursor >= inst->constantsSlicesPerFrame) {
                        LogError("constants ring overflow");
                        return kErrInternal;
                    }
                    const uint32_t slice = frameSlot * inst->constantsSlicesPerFrame + uboCursor++;
                    uint32_t size = dispatch.constantBufferDataSize;
                    if ((size % 4) != 0) {
                        // vkCmdUpdateBuffer requires a multiple of 4; the padding is
                        // never read by the shader (the block is smaller than the
                        // descriptor range).
                        size = (size + 3u) & ~3u;
                    }
                    if (size > 65536u || size > inst->constantBufferSliceSize) {
                        LogError("constants block does not fit the uniform buffer");
                        return kErrInternal;
                    }
                    inst->api.vkCmdUpdateBuffer(commandBuffer, inst->constantBuffer,
                                                static_cast<VkDeviceSize>(slice) * inst->constantBufferSliceSize, size,
                                                dispatch.constantBufferData);
                    pendingDispatch.uboSlice = slice;
                    lastSlice = slice;
                    lastSliceValid = true;
                    anyConstantsUpdate = true;
                }
            } else {
                pendingDispatch.uboSlice = lastSliceValid ? lastSlice : frameSlot * inst->constantsSlicesPerFrame;
            }

            inst->dispatches.push_back(pendingDispatch);
        }

        // ---- pass 2: record --------------------------------------------------
        if (anyConstantsUpdate) {
            VkBufferMemoryBarrier toTransfer{};
            toTransfer.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            toTransfer.srcAccessMask = kReadAccess;
            toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransfer.buffer = inst->constantBuffer;
            toTransfer.offset = 0;
            toTransfer.size = VK_WHOLE_SIZE;
            inst->api.vkCmdPipelineBarrier(commandBuffer, kComputeStage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                                           1, &toTransfer, 0, nullptr);

            // The uploads were recorded above; make them visible to the dispatches.
            VkBufferMemoryBarrier toCompute = toTransfer;
            toCompute.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toCompute.dstAccessMask = kReadAccess;
            inst->api.vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, kComputeStage, 0, 0, nullptr,
                                           1, &toCompute, 0, nullptr);
        }

        for (const PendingDispatch& pendingDispatch : inst->dispatches) {
            const VkDescriptorSet sets[2] = {pendingDispatch.set, inst->globalSet};
            const uint32_t dynamicOffset = pendingDispatch.uboSlice * static_cast<uint32_t>(inst->constantBufferSliceSize);

            inst->api.vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        inst->pipelines[pendingDispatch.pipelineIndex]);
            inst->api.vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, inst->pipelineLayout,
                                              inst->resourcesSpaceIndex, 2, sets, 1, &dynamicOffset);

            if (pendingDispatch.barrierCount) {
                inst->api.vkCmdPipelineBarrier(commandBuffer, kComputeStage, kComputeStage, 0, 0, nullptr, 0, nullptr,
                                               pendingDispatch.barrierCount,
                                               inst->barriers.data() + pendingDispatch.barrierFirst);
            }

            // NRD already computed the grid; it is never recomputed here.
            inst->api.vkCmdDispatch(commandBuffer, pendingDispatch.gridWidth, pendingDispatch.gridHeight, 1);
        }

        // ---- hand the host textures back in VK_IMAGE_LAYOUT_GENERAL ----------
        inst->barriers.clear();
        for (uint32_t i = 0; i < kHostImageNum; i++) {
            ImageState& state = hostStates[i];
            if (state.layout != VK_IMAGE_LAYOUT_GENERAL) {
                AppendBarrier(inst, state.image, state.layout, state.access, VK_IMAGE_LAYOUT_GENERAL,
                              kReadWriteAccess);
                state.layout = VK_IMAGE_LAYOUT_GENERAL;
                state.access = kReadWriteAccess;
                state.stage = kComputeStage;
            }
        }
        if (!inst->barriers.empty()) {
            inst->api.vkCmdPipelineBarrier(commandBuffer, kComputeStage, kComputeStage, 0, 0, nullptr, 0, nullptr,
                                           static_cast<uint32_t>(inst->barriers.size()), inst->barriers.data());
        }

        // Remember this frame's data: the ABI only carries the current frame, so
        // the shim derives the "prev" values the next call needs.
        std::memcpy(inst->prevViewToClip, commonSettings.viewToClipMatrix, sizeof(inst->prevViewToClip));
        std::memcpy(inst->prevWorldToView, commonSettings.worldToViewMatrix, sizeof(inst->prevWorldToView));
        inst->prevJitter[0] = commonSettings.cameraJitter[0];
        inst->prevJitter[1] = commonSettings.cameraJitter[1];
        inst->hasPrev = true;

        return CAUSTICA_NRD_OK;
    } catch (const std::exception&) {
        LogError("caustica_nrd_denoise threw");
        return kErrInternal;
    } catch (...) {
        LogError("caustica_nrd_denoise threw");
        return kErrInternal;
    }
}

CAUSTICA_NRD_EXPORT void caustica_nrd_destroy(void) {
    if (g_instance) {
        DestroyInstance(g_instance);
        g_instance = nullptr;
    }
}
