package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.gen.MotionBlurPushData;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkSamplerCreateInfo;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.comfyfluffy.caustica.rt.RtContext.check;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;

/**
 * Display-resolution motion blur: integrates the reconstructed scene along each pixel's motion vector over the
 * shutter interval, into its own scene-linear image.
 *
 * <p>It runs after reconstruction and before the display transform, so the streak is integrated in
 * scene-linear radiance and the ACES output transform then sees the blurred image — the physical order. The
 * denoisers and DLSS are untouched by it: they consume the render-resolution trace and its guides, while this
 * pass only reads the reconstruction's display-resolution result plus the guides' motion and depth.
 */
public final class RtMotionBlurPipeline {
    private static final String SHADER_DIR = "/caustica/shaders/pipelines/motion_blur/";
    private static final int THREADS = 16;

    private final RtContext ctx;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long descriptorSet;
    private final long pipelineLayout;
    private final long pipeline;
    private final long sampler;
    private long boundSceneView = -1;
    private long boundMotionView = -1;
    private long boundDepthView = -1;
    private long boundOutputView = -1;
    private boolean destroyed;

    private RtMotionBlurPipeline(RtContext ctx, long dsl, long pool, long set, long layout, long pipeline,
                                 long sampler) {
        this.ctx = ctx;
        this.descriptorSetLayout = dsl;
        this.descriptorPool = pool;
        this.descriptorSet = set;
        this.pipelineLayout = layout;
        this.pipeline = pipeline;
        this.sampler = sampler;
    }

    public static RtMotionBlurPipeline create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            // CLAMP_TO_EDGE: taps that walk off the frame are dropped by the shader's own bounds test, so the
            // sampler's behaviour there only has to be well defined, not border-coloured.
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR)
                    .minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f)
                    .maxLod(0.0f);
            LongBuffer pSampler = stack.mallocLong(1);
            check(VK10.vkCreateSampler(vk, samplerInfo, null, pSampler), "vkCreateSampler(motion blur)");
            long sampler = pSampler.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, "motion blur linear sampler");

            VkDescriptorSetLayoutBinding.Buffer binds = VkDescriptorSetLayoutBinding.calloc(MOTION_BLUR_BINDING_COUNT, stack);
            for (int binding : new int[] {MOTION_BLUR_SOURCE, MOTION_BLUR_MOTION, MOTION_BLUR_DEPTH}) {
                binds.get(binding).binding(binding).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            binds.get(MOTION_BLUR_OUTPUT).binding(MOTION_BLUR_OUTPUT)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);

            VkDescriptorSetLayoutCreateInfo dslci = VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(binds);
            LongBuffer p = stack.mallocLong(1);
            check(VK10.vkCreateDescriptorSetLayout(vk, dslci, null, p), "vkCreateDescriptorSetLayout(motion blur)");
            long dsl = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, dsl, "motion blur descriptor set layout");

            // Three sampled guides plus one storage image: the counts have to match the set layout exactly.
            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(3);
            VkDescriptorPoolCreateInfo dpci = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default().maxSets(1).pPoolSizes(poolSizes);
            check(VK10.vkCreateDescriptorPool(vk, dpci, null, p), "vkCreateDescriptorPool(motion blur)");
            long pool = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, pool, "motion blur descriptor pool");

            VkDescriptorSetAllocateInfo dsai = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                    .descriptorPool(pool).pSetLayouts(stack.longs(dsl));
            LongBuffer pSet = stack.mallocLong(1);
            check(VK10.vkAllocateDescriptorSets(vk, dsai, pSet), "vkAllocateDescriptorSets(motion blur)");
            long set = pSet.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, set, "motion blur descriptor set");

            VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
            pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(MotionBlurPushData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo plci = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                    .pSetLayouts(stack.longs(dsl)).pPushConstantRanges(pushRange);
            check(VK10.vkCreatePipelineLayout(vk, plci, null, p), "vkCreatePipelineLayout(motion blur)");
            long layout = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, "motion blur pipeline layout");

            long module = loadModule(vk, stack, "main.comp.spv");
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, module, "motion blur shader module");
            VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack).sType$Default()
                    .stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT).module(module).pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer cpci = VkComputePipelineCreateInfo.calloc(1, stack);
            cpci.get(0).sType$Default().stage(stage).layout(layout);
            LongBuffer pPipeline = stack.mallocLong(1);
            check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, cpci, null, pPipeline),
                    "vkCreateComputePipelines(motion blur)");
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, pPipeline.get(0), "motion blur compute pipeline");
            VK10.vkDestroyShaderModule(vk, module, null);

            return new RtMotionBlurPipeline(ctx, dsl, pool, set, layout, pPipeline.get(0), sampler);
        }
    }

    public long sampler() {
        return sampler;
    }

    /**
     * Point the pass at this frame's images: the display-resolution scene to integrate, the render-resolution
     * motion and depth guides, and the display-resolution destination. Views are cached, so the steady state
     * writes nothing.
     */
    public void setImages(long sceneView, long motionView, long depthView, long outputView) {
        if (boundSceneView == sceneView && boundMotionView == motionView && boundDepthView == depthView
                && boundOutputView == outputView) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer infos = VkDescriptorImageInfo.calloc(4, stack);
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(4, stack);
            int index = 0;
            for (int binding : new int[] {MOTION_BLUR_SOURCE, MOTION_BLUR_MOTION, MOTION_BLUR_DEPTH}) {
                long view = binding == MOTION_BLUR_SOURCE ? sceneView : binding == MOTION_BLUR_MOTION ? motionView : depthView;
                infos.get(index).imageView(view).sampler(sampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                writes.get(index).sType$Default().dstSet(descriptorSet).dstBinding(binding)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .pImageInfo(VkDescriptorImageInfo.create(infos.address(index), 1));
                index++;
            }
            infos.get(index).imageView(outputView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(index).sType$Default().dstSet(descriptorSet).dstBinding(MOTION_BLUR_OUTPUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .pImageInfo(VkDescriptorImageInfo.create(infos.address(index), 1));
            VK10.vkUpdateDescriptorSets(ctx.vk(), VkWriteDescriptorSet.create(writes.address(), 4), null);
        }
        boundSceneView = sceneView;
        boundMotionView = motionView;
        boundDepthView = depthView;
        boundOutputView = outputView;
    }

    /**
     * Records the pass. {@code velocityScaleX/Y} convert the guides' render-pixel motion into display-pixel
     * offsets; the projection terms turn the depth guide into view distances for the tap depth test.
     */
    public void dispatch(VkCommandBuffer cmd, int displayWidth, int displayHeight, float velocityScaleX,
                         float velocityScaleY, float projM22, float projM32, float shutterFraction,
                         int samples, float maxRadiusPixels, float depthTolerance) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "motion blur")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
                    stack.longs(descriptorSet), null);
            ByteBuffer push = stack.malloc(MotionBlurPushData.BYTE_SIZE);
            new MotionBlurPushData(displayWidth, displayHeight, samples, velocityScaleX, velocityScaleY,
                    projM22, projM32, shutterFraction, maxRadiusPixels, depthTolerance).write(push);
            VK10.vkCmdPushConstants(cmd, pipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (displayWidth + THREADS - 1) / THREADS,
                    (displayHeight + THREADS - 1) / THREADS, 1);
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, pipeline, null);
        VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
        VK10.vkDestroySampler(vk, sampler, null);
        destroyed = true;
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String name) {
        byte[] bytes;
        try (InputStream in = RtMotionBlurPipeline.class.getResourceAsStream(SHADER_DIR + name)) {
            if (in == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + SHADER_DIR + name);
            }
            bytes = in.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V resource: " + SHADER_DIR + name, e);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length).put(bytes);
        code.flip();
        try {
            VkShaderModuleCreateInfo smci = VkShaderModuleCreateInfo.calloc(stack).sType$Default().pCode(code);
            LongBuffer pModule = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(vk, smci, null, pModule), "vkCreateShaderModule(" + name + ")");
            return pModule.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }
}
