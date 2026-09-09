// Caustica rayQuery micro-benchmark host.
//
// Builds a Minecraft-like acceleration structure (one BLAS per 16^3 section,
// two geometries per BLAS: opaque solid + alpha-tested cutout), then measures
// sustained rayQuery throughput for a set of ray mixes with GPU timestamp
// queries. The output is a table of MRays/s plus a verdict that converts the
// measured rate into achievable fps for a full path tracer and for a hybrid
// raster+rayQuery renderer at the same render resolution.
//
// Headless by design: no surface, no swapchain, so it runs over `adb shell` on
// an Android device as well as on a desktop GPU.
//
// Requires: VK_KHR_acceleration_structure + VK_KHR_ray_query +
// VK_KHR_deferred_host_operations (VK_KHR_buffer_device_address on Vulkan 1.1).
// The compute kernel is a separate SPIR-V file, see bench.comp / README.md.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define VK_CHECK(call)                                                                     \
    do {                                                                                   \
        VkResult result_ = (call);                                                         \
        if (result_ != VK_SUCCESS) {                                                       \
            std::fprintf(stderr, "Vulkan error %d at %s:%d: %s\n", (int)result_, __FILE__, \
                         __LINE__, #call);                                                 \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

#define BENCH_INFO(...) std::printf(__VA_ARGS__)

namespace {

constexpr uint32_t kModePrimary = 0;
constexpr uint32_t kModeShadowOpaque = 1;
constexpr uint32_t kModeShadowCandidate = 2;
constexpr uint32_t kModeBounce1 = 3;
constexpr uint32_t kModeBounce2 = 4;

// ---------------------------------------------------------------- options ---

struct Options {
    uint32_t width = 960;
    uint32_t height = 540;
    uint32_t sections = 1024;
    uint32_t trisPerSection = 1024;
    uint32_t frames = 30;
    uint32_t warmup = 5;
    float maxT = 10000.0f;
    bool measureTlasRebuild = false;
    bool incoherent = false;
    std::vector<uint32_t> modes = {0, 1, 2, 3, 4};
    std::string spirvPath;
    bool listOnly = false;
};

void printUsage() {
    std::printf(
        "usage: rayquery-bench [options]\n"
        "  --width N         render width (default 960)\n"
        "  --height N        render height (default 540)\n"
        "  --sections N      number of 16^3 sections / BLAS (default 1024)\n"
        "  --tris N          triangles per section (default 1024)\n"
        "  --frames N        timed frames per config (default 30)\n"
        "  --warmup N        warmup frames (default 5)\n"
        "  --maxT F          ray tmax in blocks (default 10000)\n"
        "  --modes LIST      comma-separated: 0 primary, 1 +shadow-opaque,\n"
        "                    2 +shadow-candidate, 3 +1 bounce, 4 +2 bounces\n"
        "  --spirv PATH      path to bench.comp.spv (default: beside the binary)\n"
        "  --tlas-rebuild    also measure a per-frame TLAS rebuild\n"
        "  --incoherent      also measure uniform random ray directions\n"
        "  --list            print device capabilities and exit\n");
}

Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--width") o.width = (uint32_t)std::atoi(next());
        else if (a == "--height") o.height = (uint32_t)std::atoi(next());
        else if (a == "--sections") o.sections = (uint32_t)std::atoi(next());
        else if (a == "--tris") o.trisPerSection = (uint32_t)std::atoi(next());
        else if (a == "--frames") o.frames = (uint32_t)std::atoi(next());
        else if (a == "--warmup") o.warmup = (uint32_t)std::atoi(next());
        else if (a == "--maxT") o.maxT = (float)std::atof(next());
        else if (a == "--spirv") o.spirvPath = next();
        else if (a == "--tlas-rebuild") o.measureTlasRebuild = true;
        else if (a == "--incoherent") o.incoherent = true;
        else if (a == "--list") o.listOnly = true;
        else if (a == "--modes") {
            o.modes.clear();
            std::string list = next();
            size_t pos = 0;
            while (pos < list.size()) {
                size_t comma = list.find(',', pos);
                if (comma == std::string::npos) comma = list.size();
                o.modes.push_back((uint32_t)std::atoi(list.substr(pos, comma - pos).c_str()));
                pos = comma + 1;
            }
        } else if (a == "--help" || a == "-h") {
            printUsage();
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            printUsage();
            std::exit(1);
        }
    }
    if (o.width == 0 || o.height == 0 || o.frames == 0) {
        std::fprintf(stderr, "width/height/frames must be non-zero\n");
        std::exit(1);
    }
    return o;
}

// ------------------------------------------------------------------- misc ---

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
    bool hostVisible = false;
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

double nowSeconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

uint32_t hashU32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

std::string apiVersionString(uint32_t v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%u.%u.%u", VK_VERSION_MAJOR(v), VK_VERSION_MINOR(v),
                  VK_VERSION_PATCH(v));
    return buf;
}

std::string dirName(const std::string& path) {
    size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

std::vector<char> readFile(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(3);
    }
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> data((size_t)size);
    if (size > 0 && std::fread(data.data(), 1, (size_t)size, f) != (size_t)size) {
        std::fprintf(stderr, "short read on %s\n", path.c_str());
        std::exit(3);
    }
    std::fclose(f);
    return data;
}

// ------------------------------------------------------------- vulkan core ---

struct Device {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    uint32_t timestampValidBits = 0;
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR asProperties{};
    VkCommandPool commandPool = VK_NULL_HANDLE;

    PFN_vkGetBufferDeviceAddress getBufferDeviceAddress = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR getAsBuildSizes = nullptr;
    PFN_vkCreateAccelerationStructureKHR createAccelerationStructure = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR getAsDeviceAddress = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR cmdBuildAccelerationStructures = nullptr;
    PFN_vkResetQueryPool resetQueryPool = nullptr;
};

bool hasExtension(const std::vector<VkExtensionProperties>& list, const char* name) {
    for (const auto& e : list) {
        if (std::strcmp(e.extensionName, name) == 0) return true;
    }
    return false;
}

uint32_t findMemoryType(const VkPhysicalDeviceMemoryProperties& props, uint32_t bits,
                        VkMemoryPropertyFlags required) {
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    return UINT32_MAX;
}

Device createDevice() {
    Device d;

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "caustica-rayquery-bench";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &d.instance));

    uint32_t deviceCount = 0;
    const VkResult enumerateResult = vkEnumeratePhysicalDevices(d.instance, &deviceCount, nullptr);
    if (enumerateResult != VK_SUCCESS) {
        std::fprintf(stderr,
                     "vkEnumeratePhysicalDevices failed (VkResult %d); this environment has no "
                     "usable Vulkan device (headless host, container, or missing GPU access)\n",
                     (int)enumerateResult);
        std::exit(2);
    }
    if (deviceCount == 0) {
        std::fprintf(stderr, "no Vulkan physical device found\n");
        std::exit(2);
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    VK_CHECK(vkEnumeratePhysicalDevices(d.instance, &deviceCount, devices.data()));

    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    for (VkPhysicalDevice candidate : devices) {
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extCount, exts.data());
        if (hasExtension(exts, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
            hasExtension(exts, VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
            hasExtension(exts, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)) {
            chosen = candidate;
            break;
        }
    }
    if (chosen == VK_NULL_HANDLE) {
        std::fprintf(stderr,
                     "no device exposes VK_KHR_acceleration_structure + VK_KHR_ray_query + "
                     "VK_KHR_deferred_host_operations\n");
        std::exit(3);
    }
    d.physical = chosen;

    vkGetPhysicalDeviceProperties(chosen, &d.properties);
    vkGetPhysicalDeviceMemoryProperties(chosen, &d.memoryProperties);
    d.asProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    // Safe floor in case the query below is unavailable.
    d.asProperties.minAccelerationStructureScratchOffsetAlignment = 256;
    // vkGetPhysicalDeviceProperties2 is Vulkan 1.1 core, but the NDK's libvulkan stub
    // for API levels below 29 exports only 1.0 symbols, so resolve it through the
    // loader instead of referencing the symbol directly.
    auto getProperties2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(
            d.instance, "vkGetPhysicalDeviceProperties2");
    if (!getProperties2) {
        getProperties2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(
                d.instance, "vkGetPhysicalDeviceProperties2KHR");
    }
    if (getProperties2) {
        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &d.asProperties;
        getProperties2(chosen, &props2);
    }

    uint32_t queueCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(chosen, &queueCount, nullptr);
    std::vector<VkQueueFamilyProperties> queues(queueCount);
    vkGetPhysicalDeviceQueueFamilyProperties(chosen, &queueCount, queues.data());
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < queueCount; ++i) {
        if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            family = i;
            d.timestampValidBits = queues[i].timestampValidBits;
            break;
        }
    }
    if (family == UINT32_MAX) {
        std::fprintf(stderr, "no compute queue family\n");
        std::exit(3);
    }
    d.queueFamily = family;

    uint32_t chosenExtCount = 0;
    vkEnumerateDeviceExtensionProperties(chosen, nullptr, &chosenExtCount, nullptr);
    std::vector<VkExtensionProperties> chosenExts(chosenExtCount);
    vkEnumerateDeviceExtensionProperties(chosen, nullptr, &chosenExtCount, chosenExts.data());

    const bool core12 = d.properties.apiVersion >= VK_API_VERSION_1_2;
    std::vector<const char*> extensions = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                           VK_KHR_RAY_QUERY_EXTENSION_NAME,
                                           VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
    if (!core12) extensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    // vkCmdResetQueryPool is core only in 1.2; on a 1.1 device the host-side reset
    // needs VK_EXT_host_query_reset, used when the device advertises it.
    const bool useHostQueryResetExt =
            !core12 && hasExtension(chosenExts, VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME);
    if (useHostQueryResetExt) extensions.push_back(VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME);

    VkPhysicalDeviceHostQueryResetFeatures hostQueryResetFeatures{};
    hostQueryResetFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES;
    hostQueryResetFeatures.hostQueryReset = VK_TRUE;

    VkPhysicalDeviceBufferDeviceAddressFeatures bda{};
    bda.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    bda.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asFeatures.accelerationStructure = VK_TRUE;
    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{};
    rayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rayQueryFeatures.rayQuery = VK_TRUE;
    asFeatures.pNext = &rayQueryFeatures;
    if (useHostQueryResetExt) rayQueryFeatures.pNext = &hostQueryResetFeatures;
    bda.pNext = &asFeatures;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = family;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &bda;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &queueInfo;
    dci.enabledExtensionCount = (uint32_t)extensions.size();
    dci.ppEnabledExtensionNames = extensions.data();
    VK_CHECK(vkCreateDevice(chosen, &dci, nullptr, &d.device));
    vkGetDeviceQueue(d.device, family, 0, &d.queue);

    auto load = [&](const char* name) { return vkGetDeviceProcAddr(d.device, name); };
    d.getBufferDeviceAddress = (PFN_vkGetBufferDeviceAddress)load("vkGetBufferDeviceAddress");
    if (!d.getBufferDeviceAddress) {
        d.getBufferDeviceAddress = (PFN_vkGetBufferDeviceAddress)load("vkGetBufferDeviceAddressKHR");
    }
    d.getAsBuildSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)load("vkGetAccelerationStructureBuildSizesKHR");
    d.createAccelerationStructure = (PFN_vkCreateAccelerationStructureKHR)load("vkCreateAccelerationStructureKHR");
    d.getAsDeviceAddress = (PFN_vkGetAccelerationStructureDeviceAddressKHR)load("vkGetAccelerationStructureDeviceAddressKHR");
    d.cmdBuildAccelerationStructures = (PFN_vkCmdBuildAccelerationStructuresKHR)load("vkCmdBuildAccelerationStructuresKHR");
    d.resetQueryPool = (PFN_vkResetQueryPool)load("vkResetQueryPool");
    if (!d.resetQueryPool) {
        d.resetQueryPool = (PFN_vkResetQueryPool)load("vkResetQueryPoolEXT");
    }
    if (!d.getBufferDeviceAddress || !d.getAsBuildSizes || !d.createAccelerationStructure ||
        !d.getAsDeviceAddress || !d.cmdBuildAccelerationStructures) {
        std::fprintf(stderr, "failed to load required acceleration-structure entry points\n");
        std::exit(3);
    }

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = family;
    VK_CHECK(vkCreateCommandPool(d.device, &poolInfo, nullptr, &d.commandPool));
    return d;
}

Buffer createBuffer(Device& d, VkDeviceSize size, VkBufferUsageFlags usage, bool deviceLocal) {
    Buffer b;
    b.size = size;
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(d.device, &info, nullptr, &b.buffer));

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(d.device, b.buffer, &req);
    VkMemoryPropertyFlags flags = deviceLocal
            ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    uint32_t type = findMemoryType(d.memoryProperties, req.memoryTypeBits, flags);
    if (type == UINT32_MAX && deviceLocal) {
        flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        type = findMemoryType(d.memoryProperties, req.memoryTypeBits, flags);
    }
    if (type == UINT32_MAX) {
        std::fprintf(stderr, "no suitable memory type for %llu bytes\n", (unsigned long long)size);
        std::exit(3);
    }
    const bool wantAddress = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
    VkMemoryAllocateFlagsInfo allocFlags{};
    allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = wantAddress ? &allocFlags : nullptr;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    VK_CHECK(vkAllocateMemory(d.device, &alloc, nullptr, &b.memory));
    VK_CHECK(vkBindBufferMemory(d.device, b.buffer, b.memory, 0));

    b.hostVisible = (d.memoryProperties.memoryTypes[type].propertyFlags &
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    if (b.hostVisible) {
        VK_CHECK(vkMapMemory(d.device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
    }
    if (wantAddress) {
        VkBufferDeviceAddressInfo addressInfo{};
        addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addressInfo.buffer = b.buffer;
        b.address = d.getBufferDeviceAddress(d.device, &addressInfo);
    }
    return b;
}

void destroyBuffer(Device& d, Buffer& b) {
    if (b.mapped) vkUnmapMemory(d.device, b.memory);
    if (b.buffer) vkDestroyBuffer(d.device, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(d.device, b.memory, nullptr);
    b = Buffer{};
}

void oneShotSubmit(Device& d, const std::function<void(VkCommandBuffer)>& record) {
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = d.commandPool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(d.device, &alloc, &cmd));
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
    record(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(d.device, &fenceInfo, nullptr, &fence));
    VK_CHECK(vkQueueSubmit(d.queue, 1, &submit, fence));
    VK_CHECK(vkWaitForFences(d.device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(d.device, fence, nullptr);
    vkFreeCommandBuffers(d.device, d.commandPool, 1, &cmd);
}

void upload(Device& d, const Buffer& dst, const void* data, VkDeviceSize size, VkDeviceSize offset = 0) {
    if (dst.hostVisible) {
        std::memcpy((uint8_t*)dst.mapped + offset, data, (size_t)size);
        return;
    }
    Buffer staging = createBuffer(d, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);
    std::memcpy(staging.mapped, data, (size_t)size);
    oneShotSubmit(d, [&](VkCommandBuffer cmd) {
        VkBufferCopy copy{};
        copy.srcOffset = 0;
        copy.dstOffset = offset;
        copy.size = size;
        vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &copy);
    });
    destroyBuffer(d, staging);
}

Image createImage(Device& d, uint32_t width, uint32_t height, VkFormat format,
                  VkImageUsageFlags usage) {
    Image img;
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(d.device, &info, nullptr, &img.image));

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(d.device, img.image, &req);
    uint32_t type = findMemoryType(d.memoryProperties, req.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    VK_CHECK(vkAllocateMemory(d.device, &alloc, nullptr, &img.memory));
    VK_CHECK(vkBindImageMemory(d.device, img.image, img.memory, 0));

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = img.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(d.device, &viewInfo, nullptr, &img.view));
    return img;
}

void transitionImage(Device& d, VkImage image, VkImageLayout from, VkImageLayout to,
                     VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                     VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
    oneShotSubmit(d, [&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = from;
        barrier.newLayout = to;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = srcAccess;
        barrier.dstAccessMask = dstAccess;
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    });
}

// ------------------------------------------------------------------- scene ---

struct Scene {
    uint32_t sections = 0;
    uint32_t trisPerSection = 0;
    uint32_t solidTrisPerSection = 0;
    uint32_t cutoutTrisPerSection = 0;
    uint32_t gridX = 0, gridY = 0, gridZ = 0;
    uint64_t totalTris = 0;
    Buffer vertices;
    Buffer uvs;
    Buffer instances;
    VkDeviceSize vertexBytes = 0;
    VkDeviceSize uvBytes = 0;
};

Scene buildScene(Device& d, const Options& o) {
    Scene s;
    uint32_t tris = std::max(2u, o.trisPerSection) & ~1u;
    s.trisPerSection = tris;
    s.solidTrisPerSection = tris / 2;
    s.cutoutTrisPerSection = tris - s.solidTrisPerSection;

    uint32_t requested = std::max(1u, o.sections);
    s.gridX = std::max(1u, (uint32_t)std::floor(std::cbrt((double)requested)));
    s.gridZ = s.gridX;
    s.gridY = std::max(1u, requested / (s.gridX * s.gridZ));
    while (s.gridX * s.gridY * s.gridZ < requested) ++s.gridX;
    s.sections = s.gridX * s.gridY * s.gridZ;
    s.totalTris = (uint64_t)s.sections * tris;

    const uint32_t cellsPerSide = std::max(1u, (uint32_t)std::sqrt((double)(tris / 2)));
    const float cellSize = 16.0f / (float)cellsPerSide;

    std::vector<float> vertexData((size_t)s.totalTris * 3 * 3);
    std::vector<float> uvData((size_t)s.totalTris * 3 * 2);
    size_t vcursor = 0;
    size_t uvcursor = 0;

    for (uint32_t section = 0; section < s.sections; ++section) {
        if (section % 256 == 0 && section > 0) {
            BENCH_INFO("    geometry: %u/%u sections\n", section, s.sections);
        }
        const uint32_t sx = section % s.gridX;
        const uint32_t sz = (section / s.gridX) % s.gridZ;
        const uint32_t sy = section / (s.gridX * s.gridZ);
        const float originX = (float)sx * 16.0f;
        const float originY = (float)sy * 16.0f;
        const float originZ = (float)sz * 16.0f;

        auto height = [&](uint32_t cx, uint32_t cz) {
            const float fx = (float)cx * cellSize;
            const float fz = (float)cz * cellSize;
            const float h = 6.0f
                    + 2.6f * std::sin(fx * 0.45f + (float)section * 0.37f)
                    + 2.2f * std::cos(fz * 0.41f + (float)section * 0.21f)
                    + 0.7f * std::sin((fx + fz) * 0.9f);
            return std::min(15.5f, std::max(0.5f, h));
        };

        for (uint32_t cz = 0; cz < cellsPerSide; ++cz) {
            for (uint32_t cx = 0; cx < cellsPerSide; ++cx) {
                const float x0 = originX + (float)cx * cellSize;
                const float z0 = originZ + (float)cz * cellSize;
                const float x1 = x0 + cellSize;
                const float z1 = z0 + cellSize;
                const float y00 = originY + height(cx, cz);
                const float y10 = originY + height(cx + 1, cz);
                const float y01 = originY + height(cx, cz + 1);
                const float y11 = originY + height(cx + 1, cz + 1);

                const float u0 = (float)((cx + section) % 8) * 0.125f;
                const float v0 = (float)((cz + section) % 8) * 0.125f;
                const float u1 = u0 + 0.125f;
                const float v1 = v0 + 0.125f;

                const float tri[2][3][3] = {
                        {{x0, y00, z0}, {x1, y10, z0}, {x0, y01, z1}},
                        {{x1, y10, z0}, {x1, y11, z1}, {x0, y01, z1}},
                };
                const float uv[2][3][2] = {
                        {{u0, v0}, {u1, v0}, {u0, v1}},
                        {{u1, v0}, {u1, v1}, {u0, v1}},
                };
                for (int t = 0; t < 2; ++t) {
                    for (int v = 0; v < 3; ++v) {
                        std::memcpy(&vertexData[vcursor], tri[t][v], sizeof(float) * 3);
                        vcursor += 3;
                        std::memcpy(&uvData[uvcursor], uv[t][v], sizeof(float) * 2);
                        uvcursor += 2;
                    }
                }
            }
        }
    }

    s.vertexBytes = (VkDeviceSize)vertexData.size() * sizeof(float);
    s.uvBytes = (VkDeviceSize)uvData.size() * sizeof(float);
    const VkBufferUsageFlags asInput = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    s.vertices = createBuffer(d, s.vertexBytes, asInput, true);
    s.uvs = createBuffer(d, s.uvBytes, asInput, true);
    upload(d, s.vertices, vertexData.data(), s.vertexBytes);
    upload(d, s.uvs, uvData.data(), s.uvBytes);

    std::vector<VkAccelerationStructureInstanceKHR> instanceData(s.sections);
    for (uint32_t i = 0; i < s.sections; ++i) {
        const uint32_t sx = i % s.gridX;
        const uint32_t sz = (i / s.gridX) % s.gridZ;
        const uint32_t sy = i / (s.gridX * s.gridZ);
        VkAccelerationStructureInstanceKHR inst{};
        inst.transform.matrix[0][0] = 1.0f;
        inst.transform.matrix[1][1] = 1.0f;
        inst.transform.matrix[2][2] = 1.0f;
        inst.transform.matrix[0][3] = (float)sx * 16.0f;
        inst.transform.matrix[1][3] = (float)sy * 16.0f;
        inst.transform.matrix[2][3] = (float)sz * 16.0f;
        inst.instanceCustomIndex = i;
        inst.mask = 0xFF;
        inst.instanceShaderBindingTableRecordOffset = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instanceData[i] = inst;
    }
    s.instances = createBuffer(d, sizeof(VkAccelerationStructureInstanceKHR) * s.sections,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    upload(d, s.instances, instanceData.data(),
           sizeof(VkAccelerationStructureInstanceKHR) * s.sections);
    return s;
}

// --------------------------------------------------- acceleration structures ---

struct Accel {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    Buffer storage;
    VkDeviceAddress address = 0;
};

Accel createAccel(Device& d, VkAccelerationStructureTypeKHR type, VkDeviceSize size) {
    Accel a;
    a.storage = createBuffer(d, size,
                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, true);
    VkAccelerationStructureCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    info.type = type;
    info.size = size;
    info.buffer = a.storage.buffer;
    VK_CHECK(d.createAccelerationStructure(d.device, &info, nullptr, &a.handle));
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addressInfo.accelerationStructure = a.handle;
    a.address = d.getAsDeviceAddress(d.device, &addressInfo);
    return a;
}

struct AccelSet {
    std::vector<Accel> blases;
    Accel tlas;
    double blasBuildMs = 0.0;
    double tlasBuildMs = 0.0;
    VkDeviceSize blasBytes = 0;
    VkDeviceSize tlasBytes = 0;
    // Retained so the TLAS can be rebuilt per frame during the measurement. The
    // scratch buffer must outlive the build info: tlasBuild.scratchData.deviceAddress
    // points into it, and --tlas-rebuild submits that address every frame.
    Buffer tlasScratch;
    VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{};
    VkAccelerationStructureBuildRangeInfoKHR tlasRange{};
};

AccelSet buildAccelerationStructures(Device& d, const Scene& scene) {
    AccelSet set;
    set.blases.resize(scene.sections);

    std::vector<VkAccelerationStructureGeometryKHR> geometries(scene.sections * 2);
    std::vector<uint32_t> primitiveCounts(scene.sections * 2);
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(scene.sections);
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(scene.sections * 2);
    std::vector<VkDeviceSize> blasSizes(scene.sections);

    const uint32_t solidTris = scene.solidTrisPerSection;
    const uint32_t cutoutTris = scene.cutoutTrisPerSection;
    const VkDeviceSize sectionVertexStride = (VkDeviceSize)scene.trisPerSection * 3 * 12;

    for (uint32_t i = 0; i < scene.sections; ++i) {
        for (uint32_t g = 0; g < 2; ++g) {
            const uint32_t triCount = (g == 0) ? solidTris : cutoutTris;
            VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
            triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triangles.vertexData.deviceAddress = scene.vertices.address + sectionVertexStride * i
                    + (g == 0 ? 0 : (VkDeviceSize)solidTris * 3 * 12);
            triangles.vertexStride = 12;
            triangles.maxVertex = triCount * 3 - 1;
            triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
            triangles.indexData.deviceAddress = 0;
            triangles.transformData.deviceAddress = 0;

            VkAccelerationStructureGeometryKHR geometry{};
            geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            geometry.flags = (g == 0) ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
            geometry.geometry.triangles = triangles;
            geometries[i * 2 + g] = geometry;
            primitiveCounts[i * 2 + g] = triCount;
            ranges[i * 2 + g].primitiveCount = triCount;
        }

        VkAccelerationStructureBuildGeometryInfoKHR build{};
        build.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build.geometryCount = 2;
        build.pGeometries = &geometries[i * 2];
        buildInfos[i] = build;

        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        d.getAsBuildSizes(d.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                          &buildInfos[i], &primitiveCounts[i * 2], &sizes);
        blasSizes[i] = sizes.accelerationStructureSize;
        set.blasBytes += sizes.accelerationStructureSize;
    }

    const VkDeviceSize scratchAlignment = std::max<VkDeviceSize>(
            1, d.asProperties.minAccelerationStructureScratchOffsetAlignment);
    std::vector<VkDeviceSize> scratchOffsets(scene.sections);
    VkDeviceSize scratchTotal = 0;
    for (uint32_t i = 0; i < scene.sections; ++i) {
        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        d.getAsBuildSizes(d.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                          &buildInfos[i], &primitiveCounts[i * 2], &sizes);
        scratchTotal = (scratchTotal + scratchAlignment - 1) / scratchAlignment * scratchAlignment;
        scratchOffsets[i] = scratchTotal;
        scratchTotal += sizes.buildScratchSize;
    }
    Buffer scratch = createBuffer(d, scratchTotal,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, true);

    for (uint32_t i = 0; i < scene.sections; ++i) {
        set.blases[i] = createAccel(d, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, blasSizes[i]);
        buildInfos[i].dstAccelerationStructure = set.blases[i].handle;
        buildInfos[i].scratchData.deviceAddress = scratch.address + scratchOffsets[i];
    }

    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePointers(scene.sections);
    for (uint32_t i = 0; i < scene.sections; ++i) rangePointers[i] = &ranges[i * 2];

    const double blasStart = nowSeconds();
    BENCH_INFO("  BLAS build submitted (scratch %.1f MB); waiting on the GPU...\n",
               (double)scratchTotal / (1024.0 * 1024.0));
    oneShotSubmit(d, [&](VkCommandBuffer cmd) {
        d.cmdBuildAccelerationStructures(cmd, scene.sections, buildInfos.data(),
                                         rangePointers.data());
    });
    set.blasBuildMs = (nowSeconds() - blasStart) * 1000.0;
    BENCH_INFO("  BLAS build done in %.1f ms\n", set.blasBuildMs);

    // --- TLAS ---
    VkAccelerationStructureGeometryKHR tlasGeometry{};
    tlasGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlasGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeometry.geometry.instances.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeometry.geometry.instances.data.deviceAddress = scene.instances.address;

    VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{};
    tlasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    tlasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tlasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tlasBuild.geometryCount = 1;
    tlasBuild.pGeometries = &tlasGeometry;

    VkAccelerationStructureBuildSizesInfoKHR tlasSizes{};
    tlasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    d.getAsBuildSizes(d.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuild,
                      &scene.sections, &tlasSizes);
    set.tlasBytes = tlasSizes.accelerationStructureSize;

    // The instance buffer needs the BLAS device addresses, which only exist now.
    std::vector<VkAccelerationStructureInstanceKHR> instanceData(scene.sections);
    {
        Buffer readback = createBuffer(d, sizeof(VkAccelerationStructureInstanceKHR) * scene.sections,
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
        oneShotSubmit(d, [&](VkCommandBuffer cmd) {
            VkBufferCopy copy{};
            copy.size = readback.size;
            vkCmdCopyBuffer(cmd, scene.instances.buffer, readback.buffer, 1, &copy);
        });
        std::memcpy(instanceData.data(), readback.mapped, (size_t)readback.size);
        destroyBuffer(d, readback);
    }
    for (uint32_t i = 0; i < scene.sections; ++i) {
        instanceData[i].accelerationStructureReference = set.blases[i].address;
    }
    upload(d, scene.instances, instanceData.data(),
           sizeof(VkAccelerationStructureInstanceKHR) * scene.sections);

    set.tlas = createAccel(d, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                           tlasSizes.accelerationStructureSize);
    tlasBuild.dstAccelerationStructure = set.tlas.handle;
    // Kept alive for the whole run: --tlas-rebuild reuses this address every frame.
    set.tlasScratch = createBuffer(d, tlasSizes.buildScratchSize,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, true);
    tlasBuild.scratchData.deviceAddress = set.tlasScratch.address;
    set.tlasBuild = tlasBuild;
    set.tlasRange = {scene.sections, 0, 0, 0};

    const double tlasStart = nowSeconds();
    oneShotSubmit(d, [&](VkCommandBuffer cmd) {
        const VkAccelerationStructureBuildRangeInfoKHR* rangePtr = &set.tlasRange;
        d.cmdBuildAccelerationStructures(cmd, 1, &set.tlasBuild, &rangePtr);
    });
    set.tlasBuildMs = (nowSeconds() - tlasStart) * 1000.0;
    BENCH_INFO("  TLAS build done in %.2f ms\n", set.tlasBuildMs);
    destroyBuffer(d, scratch);
    return set;
}

// ------------------------------------------------------------------ shader ---

struct Push {
    float camPosTanX[4];
    float camRight[4];
    float camUp[4];
    float camFwdTanY[4];
    float params[4];   // spp, maxBounces, maxT, coherence
    float sun[4];      // xyz direction, w intensity
    uint32_t dims[4];  // x, y, mode, frameIndex
    uint32_t extra[4]; // trisPerSection, cutoutTriOffset, seedSalt, reserved
};
static_assert(sizeof(Push) == 128, "push constant block must be 128 bytes");

struct Pipeline {
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

Pipeline createPipeline(Device& d, const std::string& spirvPath, uint32_t bindlessUnused) {
    (void)bindlessUnused;
    Pipeline p;

    VkDescriptorSetLayoutBinding bindings[5]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 5;
    layoutInfo.pBindings = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(d.device, &layoutInfo, nullptr, &p.setLayout));

    VkDescriptorPoolSize sizes[4]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    sizes[0].descriptorCount = 1;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[1].descriptorCount = 1;
    sizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[2].descriptorCount = 1;
    sizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[3].descriptorCount = 2;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 4;
    poolInfo.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(d.device, &poolInfo, nullptr, &p.pool));

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = p.pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &p.setLayout;
    VK_CHECK(vkAllocateDescriptorSets(d.device, &allocInfo, &p.set));

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.offset = 0;
    range.size = sizeof(Push);
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &p.setLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(d.device, &pipelineLayoutInfo, nullptr, &p.layout));

    const std::vector<char> code = readFile(spirvPath);
    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = code.size();
    moduleInfo.pCode = (const uint32_t*)code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(d.device, &moduleInfo, nullptr, &module));

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = module;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = p.layout;
    VK_CHECK(vkCreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                      &p.pipeline));
    vkDestroyShaderModule(d.device, module, nullptr);
    return p;
}

// ------------------------------------------------------------------ results ---

struct RunResult {
    std::string name;
    uint32_t mode = 0;
    uint32_t bounces = 0;
    bool incoherent = false;
    double gpuMsPerFrame = 0.0;
    double wallMsPerFrame = 0.0;
    double raysPerFrame = 0.0;
    double raysPerPixel = 0.0;
    double mraysPerSecond = 0.0;
    bool timestamps = false;
};

}  // namespace

int main(int argc, char** argv) {
    // Unbuffered stdout: over `adb shell "cmd"` stdout is a pipe, not a TTY, so the C
    // runtime would block-buffer progress until 4 KiB accumulated or the process exited,
    // which makes a slow on-device run look like a hang.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const Options options = parseArgs(argc, argv);
    Device d = createDevice();

    const VkPhysicalDeviceLimits& limits = d.properties.limits;
    BENCH_INFO("=== device ===\n");
    BENCH_INFO("  name                     %s\n", d.properties.deviceName);
    BENCH_INFO("  apiVersion               %s\n", apiVersionString(d.properties.apiVersion).c_str());
    BENCH_INFO("  driverVersion            0x%08x\n", d.properties.driverVersion);
    BENCH_INFO("  deviceType               %d\n", (int)d.properties.deviceType);
    BENCH_INFO("  timestampPeriod          %.4f ns\n", limits.timestampPeriod);
    BENCH_INFO("  timestampValidBits       %u\n", d.timestampValidBits);
    BENCH_INFO("  timestampComputeAndGraphics %d\n", (int)limits.timestampComputeAndGraphics);
    BENCH_INFO("  maxPushConstantsSize     %u\n", limits.maxPushConstantsSize);
    BENCH_INFO("  maxComputeWorkGroupSize  %u x %u x %u\n", limits.maxComputeWorkGroupSize[0],
               limits.maxComputeWorkGroupSize[1], limits.maxComputeWorkGroupSize[2]);
    BENCH_INFO("  maxComputeInvocations    %u\n", limits.maxComputeWorkGroupInvocations);
    BENCH_INFO("  maxStorageBufferRange    %u\n", limits.maxStorageBufferRange);
    BENCH_INFO("  asScratchAlignment       %u\n",
               d.asProperties.minAccelerationStructureScratchOffsetAlignment);
    BENCH_INFO("  maxInstanceCount         %llu\n",
               (unsigned long long)d.asProperties.maxInstanceCount);

    if (options.listOnly) return 0;
    if (limits.maxPushConstantsSize < sizeof(Push)) {
        std::fprintf(stderr, "device push constant limit %u < %zu\n", limits.maxPushConstantsSize,
                     sizeof(Push));
        return 3;
    }

    VkFormatProperties storageFormat{};
    vkGetPhysicalDeviceFormatProperties(d.physical, VK_FORMAT_R16G16B16A16_SFLOAT,
                                        &storageFormat);
    if (!(storageFormat.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
        std::fprintf(stderr, "R16G16B16A16_SFLOAT lacks storage-image support\n");
        return 3;
    }

    std::string spirvPath = options.spirvPath;
    if (spirvPath.empty()) spirvPath = dirName(argv[0]) + "/bench.comp.spv";

    // --- scene ---
    BENCH_INFO("\n=== building scene ===\n");
    BENCH_INFO("  target: %u sections x %u triangles, render %ux%u\n", options.sections,
               options.trisPerSection, options.width, options.height);
    BENCH_INFO("  generating geometry and uploading (this is CPU work)...\n");
    const double sceneStart = nowSeconds();
    Scene scene = buildScene(d, options);
    BENCH_INFO("  geometry ready in %.1f ms\n", (nowSeconds() - sceneStart) * 1000.0);
    BENCH_INFO("  building %u BLASes (%.2f M triangles) - the slow phase on mobile...\n",
               scene.sections, (double)scene.totalTris / 1e6);
    AccelSet accel = buildAccelerationStructures(d, scene);
    const double sceneMs = (nowSeconds() - sceneStart) * 1000.0;
    BENCH_INFO("  sections                 %u (%u x %u x %u, 16^3 each)\n", scene.sections,
               scene.gridX, scene.gridY, scene.gridZ);
    BENCH_INFO("  triangles                %llu (%.0f per section)\n",
               (unsigned long long)scene.totalTris, (double)scene.trisPerSection);
    BENCH_INFO("  vertex + uv memory       %.1f MB\n",
               (double)(scene.vertexBytes + scene.uvBytes) / (1024.0 * 1024.0));
    BENCH_INFO("  BLAS memory              %.1f MB\n",
               (double)accel.blasBytes / (1024.0 * 1024.0));
    BENCH_INFO("  TLAS memory              %.2f MB (%u instances)\n",
               (double)accel.tlasBytes / (1024.0 * 1024.0), scene.sections);
    BENCH_INFO("  BLAS build (all sections) %.1f ms\n", accel.blasBuildMs);
    BENCH_INFO("  TLAS build               %.2f ms\n", accel.tlasBuildMs);
    BENCH_INFO("  total setup              %.1f ms\n", sceneMs);

    // --- alpha texture ---
    const uint32_t texSize = 256;
    std::vector<uint8_t> texels(texSize * texSize * 4);
    for (uint32_t y = 0; y < texSize; ++y) {
        for (uint32_t x = 0; x < texSize; ++x) {
            const uint32_t h = hashU32(x * 1973u ^ y * 9277u);
            const uint8_t alpha = ((h & 255u) < 102u) ? 0u : 255u;  // ~40% transparent
            uint8_t* t = &texels[(y * texSize + x) * 4];
            t[0] = (uint8_t)(h & 255u);
            t[1] = (uint8_t)((h >> 8) & 255u);
            t[2] = (uint8_t)((h >> 16) & 255u);
            t[3] = alpha;
        }
    }
    Image alphaImage = createImage(d, texSize, texSize, VK_FORMAT_R8G8B8A8_UNORM,
                                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    transitionImage(d, alphaImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    {
        Buffer staging = createBuffer(d, texels.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);
        std::memcpy(staging.mapped, texels.data(), texels.size());
        oneShotSubmit(d, [&](VkCommandBuffer cmd) {
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {texSize, texSize, 1};
            vkCmdCopyBufferToImage(cmd, staging.buffer, alphaImage.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        });
        destroyBuffer(d, staging);
    }
    transitionImage(d, alphaImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkSampler sampler = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSampler(d.device, &samplerInfo, nullptr, &sampler));

    Image output = createImage(d, options.width, options.height, VK_FORMAT_R16G16B16A16_SFLOAT,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    transitionImage(d, output.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    Buffer rayCounter = createBuffer(d, sizeof(uint32_t),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    Buffer rayReadback = createBuffer(d, sizeof(uint32_t) * options.frames,
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);

    Pipeline pipeline = createPipeline(d, spirvPath, 0);

    // --- descriptor writes ---
    VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
    asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures = &accel.tlas.handle;

    VkDescriptorImageInfo storageImageInfo{};
    storageImageInfo.imageView = output.view;
    storageImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo samplerImageInfo{};
    samplerImageInfo.sampler = sampler;
    samplerImageInfo.imageView = alphaImage.view;
    samplerImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo uvInfo{};
    uvInfo.buffer = scene.uvs.buffer;
    uvInfo.offset = 0;
    uvInfo.range = scene.uvBytes;

    VkDescriptorBufferInfo counterInfo{};
    counterInfo.buffer = rayCounter.buffer;
    counterInfo.offset = 0;
    counterInfo.range = sizeof(uint32_t);

    VkWriteDescriptorSet writes[5]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = &asWrite;
    writes[0].dstSet = pipeline.set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = pipeline.set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &storageImageInfo;
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = pipeline.set;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &samplerImageInfo;
    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = pipeline.set;
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].pBufferInfo = &uvInfo;
    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet = pipeline.set;
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[4].pBufferInfo = &counterInfo;
    vkUpdateDescriptorSets(d.device, 5, writes, 0, nullptr);

    // --- run list ---
    struct Run {
        std::string name;
        uint32_t mode;
        uint32_t bounces;
        bool incoherent;
    };
    std::vector<Run> runs;
    auto addRun = [&](const char* name, uint32_t mode, uint32_t bounces, bool incoherent) {
        for (uint32_t m : options.modes) {
            if (m == mode) {
                runs.push_back({name, mode, bounces, incoherent});
                return;
            }
        }
    };
    addRun("primary-only", kModePrimary, 0, false);
    addRun("primary+shadow-opaque", kModeShadowOpaque, 0, false);
    addRun("primary+shadow-candidate", kModeShadowCandidate, 0, false);
    addRun("path-1bounce", kModeBounce1, 1, false);
    addRun("path-2bounce", kModeBounce2, 2, false);
    if (options.incoherent) {
        runs.push_back({"primary-random", kModePrimary, 0, true});
        runs.push_back({"path-2bounce-random", kModeBounce2, 2, true});
    }

    // Camera inside the section volume, looking across it.
    const float camX = (float)scene.gridX * 8.0f;
    const float camY = (float)scene.gridY * 8.0f + 12.0f;
    const float camZ = (float)scene.gridZ * 8.0f;
    const float forward[3] = {0.25f, -0.15f, 1.0f};
    const float forwardLength = std::sqrt(forward[0] * forward[0] + forward[1] * forward[1] +
                                          forward[2] * forward[2]);
    const float fwd[3] = {forward[0] / forwardLength, forward[1] / forwardLength,
                          forward[2] / forwardLength};
    const float worldUp[3] = {0.0f, 1.0f, 0.0f};
    float right[3] = {fwd[1] * worldUp[2] - fwd[2] * worldUp[1],
                      fwd[2] * worldUp[0] - fwd[0] * worldUp[2],
                      fwd[0] * worldUp[1] - fwd[1] * worldUp[0]};
    const float rightLength = std::sqrt(right[0] * right[0] + right[1] * right[1] +
                                        right[2] * right[2]);
    for (float& v : right) v /= rightLength;
    const float up[3] = {right[1] * fwd[2] - right[2] * fwd[1],
                         right[2] * fwd[0] - right[0] * fwd[2],
                         right[0] * fwd[1] - right[1] * fwd[0]};
    const float tanHalfFovY = std::tan(30.0f * 3.14159265358979f / 180.0f);
    const float tanHalfFovX = tanHalfFovY * (float)options.width / (float)options.height;

    // --- timing resources ---
    const uint32_t queryCount = options.frames * 2;
    VkQueryPoolCreateInfo queryPoolInfo{};
    queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolInfo.queryCount = queryCount;
    VkQueryPool queryPool = VK_NULL_HANDLE;
    const bool useTimestamps = d.timestampValidBits > 0 && limits.timestampPeriod > 0.0f &&
            d.resetQueryPool != nullptr;
    if (useTimestamps) {
        VK_CHECK(vkCreateQueryPool(d.device, &queryPoolInfo, nullptr, &queryPool));
    } else {
        std::fprintf(stderr,
                     "note: GPU timestamps unavailable (validBits=%u, period=%.2f); "
                     "falling back to wall-clock timing\n",
                     d.timestampValidBits, limits.timestampPeriod);
    }

    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = d.commandPool;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(d.device, &cmdAlloc, &cmd));
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(d.device, &fenceInfo, nullptr, &fence));

    const uint32_t groupCountX = (options.width + 7) / 8;
    const uint32_t groupCountY = (options.height + 7) / 8;
    std::vector<RunResult> results;

    for (const Run& run : runs) {
        BENCH_INFO("  running %-26s (%u warmup + %u timed frames)...\n", run.name.c_str(),
                   options.warmup, options.frames);
        Push push{};
        push.camPosTanX[0] = camX;
        push.camPosTanX[1] = camY;
        push.camPosTanX[2] = camZ;
        push.camPosTanX[3] = tanHalfFovX;
        std::memcpy(push.camRight, right, sizeof(right));
        std::memcpy(push.camUp, up, sizeof(up));
        push.camFwdTanY[0] = fwd[0];
        push.camFwdTanY[1] = fwd[1];
        push.camFwdTanY[2] = fwd[2];
        push.camFwdTanY[3] = tanHalfFovY;
        push.params[0] = 1.0f;                    // spp (the host loop owns sampling)
        push.params[1] = (float)run.bounces;
        push.params[2] = options.maxT;
        push.params[3] = run.incoherent ? 1.0f : 0.0f;
        const float sunDir[3] = {0.4f, 0.8f, 0.45f};
        const float sunLength = std::sqrt(sunDir[0] * sunDir[0] + sunDir[1] * sunDir[1] +
                                          sunDir[2] * sunDir[2]);
        push.sun[0] = sunDir[0] / sunLength;
        push.sun[1] = sunDir[1] / sunLength;
        push.sun[2] = sunDir[2] / sunLength;
        push.sun[3] = 3.0f;
        push.dims[0] = options.width;
        push.dims[1] = options.height;
        push.dims[2] = run.mode;
        push.dims[3] = 0;
        push.extra[0] = scene.trisPerSection;
        push.extra[1] = scene.solidTrisPerSection;
        push.extra[2] = 1u;
        push.extra[3] = 0u;

        VK_CHECK(vkResetCommandBuffer(cmd, 0));
        if (useTimestamps && d.resetQueryPool) {
            // Host-side reset: valid on 1.2 core and on 1.1 + VK_EXT_host_query_reset.
            d.resetQueryPool(d.device, queryPool, 0, queryCount);
        }
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1,
                                &pipeline.set, 0, nullptr);

        const uint32_t totalFrames = options.warmup + options.frames;
        for (uint32_t frame = 0; frame < totalFrames; ++frame) {
            Push framePush = push;
            framePush.dims[3] = frame;
            vkCmdPushConstants(cmd, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push),
                               &framePush);
            if (frame >= options.warmup) {
                vkCmdFillBuffer(cmd, rayCounter.buffer, 0, sizeof(uint32_t), 0);
                VkMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                                     nullptr, 0, nullptr);
                if (useTimestamps) {
                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool,
                                        (frame - options.warmup) * 2);
                }
                if (options.measureTlasRebuild) {
                    const VkAccelerationStructureBuildRangeInfoKHR* rangePtr = &accel.tlasRange;
                    d.cmdBuildAccelerationStructures(cmd, 1, &accel.tlasBuild, &rangePtr);
                }
            }
            vkCmdDispatch(cmd, groupCountX, groupCountY, 1);
            if (frame >= options.warmup) {
                if (useTimestamps) {
                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool,
                                        (frame - options.warmup) * 2 + 1);
                }
                VkBufferCopy counterCopy{};
                counterCopy.size = sizeof(uint32_t);
                counterCopy.dstOffset = sizeof(uint32_t) * (frame - options.warmup);
                vkCmdCopyBuffer(cmd, rayCounter.buffer, rayReadback.buffer, 1, &counterCopy);
            }
        }
        VK_CHECK(vkEndCommandBuffer(cmd));

        const double wallStart = nowSeconds();
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        VK_CHECK(vkResetFences(d.device, 1, &fence));
        VK_CHECK(vkQueueSubmit(d.queue, 1, &submit, fence));
        VK_CHECK(vkWaitForFences(d.device, 1, &fence, VK_TRUE, UINT64_MAX));
        const double wallMs = (nowSeconds() - wallStart) * 1000.0;

        RunResult r;
        r.name = run.name;
        r.mode = run.mode;
        r.bounces = run.bounces;
        r.incoherent = run.incoherent;
        r.wallMsPerFrame = wallMs / (double)options.frames;
        r.timestamps = useTimestamps;

        uint64_t totalRays = 0;
        const uint32_t* counts = (const uint32_t*)rayReadback.mapped;
        for (uint32_t i = 0; i < options.frames; ++i) totalRays += counts[i];
        r.raysPerFrame = (double)totalRays / (double)options.frames;
        r.raysPerPixel = r.raysPerFrame / ((double)options.width * (double)options.height);

        if (useTimestamps) {
            std::vector<uint64_t> stamps(queryCount);
            VK_CHECK(vkGetQueryPoolResults(d.device, queryPool, 0, queryCount,
                                           stamps.size() * sizeof(uint64_t), stamps.data(),
                                           sizeof(uint64_t),
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            uint64_t ticks = 0;
            for (uint32_t i = 0; i < options.frames; ++i) {
                ticks += stamps[i * 2 + 1] - stamps[i * 2];
            }
            const double gpuMs = (double)ticks * (double)limits.timestampPeriod / 1e6;
            r.gpuMsPerFrame = gpuMs / (double)options.frames;
            // rays / ms * 1000 = rays / s; the / 1e6 to MRays/s is folded in here.
            r.mraysPerSecond = r.raysPerFrame / (r.gpuMsPerFrame * 1e3);
        } else {
            r.gpuMsPerFrame = r.wallMsPerFrame;
            r.mraysPerSecond = r.raysPerFrame / (r.wallMsPerFrame * 1e3);
        }
        results.push_back(r);

        BENCH_INFO("  %-26s %8.0f rays/frame  %7.3f ms  %8.1f MRays/s\n", r.name.c_str(),
                   r.raysPerFrame, r.gpuMsPerFrame, r.mraysPerSecond);
    }

    // --- summary ---
    BENCH_INFO("\n=== throughput (render %ux%u, %u sections, %.2f M triangles) ===\n",
               options.width, options.height, scene.sections, (double)scene.totalTris / 1e6);
    BENCH_INFO("  %-26s %8s %10s %12s %8s\n", "config", "rays/px", "ms/frame", "MRays/s", "source");
    for (const RunResult& r : results) {
        BENCH_INFO("  %-26s %8.2f %10.3f %12.1f %8s\n", r.name.c_str(), r.raysPerPixel,
                   r.gpuMsPerFrame, r.mraysPerSecond, r.timestamps ? "gpu-ts" : "wall");
    }

    // --- verdict ---
    const double pixels = (double)options.width * (double)options.height;
    const RunResult* fullPt = nullptr;
    const RunResult* hybrid = nullptr;
    for (const RunResult& r : results) {
        if (r.name == "path-2bounce") fullPt = &r;
        if (r.name == "primary+shadow-candidate") hybrid = &r;
    }
    if (fullPt && hybrid) {
        BENCH_INFO("\n=== verdict at %ux%u ===\n", options.width, options.height);
        BENCH_INFO("  full path tracer  (path-2bounce, %.2f rays/px, %.1f MRays/s)\n",
                   fullPt->raysPerPixel, fullPt->mraysPerSecond);
        BENCH_INFO("    -> %.1f fps at this render resolution\n",
                   fullPt->mraysPerSecond * 1e6 / (pixels * fullPt->raysPerPixel));
        BENCH_INFO("  hybrid raster+rayQuery (primary+shadow-candidate, %.2f rays/px, %.1f MRays/s)\n",
                   hybrid->raysPerPixel, hybrid->mraysPerSecond);
        BENCH_INFO("    -> %.1f fps at this render resolution\n",
                   hybrid->mraysPerSecond * 1e6 / (pixels * hybrid->raysPerPixel));
        const double fullPtNeeded30 = pixels * fullPt->raysPerPixel * 30.0 / 1e6;
        const double fullPtNeeded60 = pixels * fullPt->raysPerPixel * 60.0 / 1e6;
        BENCH_INFO("  required for full PT: %.0f MRays/s @30fps, %.0f MRays/s @60fps\n",
                   fullPtNeeded30, fullPtNeeded60);
    } else {
        BENCH_INFO("\n(verdict needs both mode 2 and mode 4; rerun without --modes filtering)\n");
    }

    vkDeviceWaitIdle(d.device);
    return 0;
}
