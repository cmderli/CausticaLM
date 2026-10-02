package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.gen.DepthOfFieldPushData;
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
 * Half-display-resolution depth-of-field gather over the reconstructed image.
 *
 * <p>The trace blurs with the camera aperture, but one primary ray per pixel samples one point of the pupil, so
 * on its own it resolves only a small disc of confusion: a pixel whose disc is dozens of pixels wide gets a
 * single sample of it, which reads as speckle and as an in/out flicker along every silhouette, and resolving it
 * with more samples costs the whole ray budget multiplied by the sample count. This pass supplies the rest of
 * the disc by gathering, so the traced part stays a physically correct near-field bokeh and the gather expands
 * it to the aperture the camera was set to without a single extra ray.
 *
 * <p>Half resolution is what makes a kernel of dozens of pixels affordable, and the accumulating history is what
 * makes a small tap count sufficient: the tap pattern rotates with the frame, so a wide disc is covered by the
 * union of several frames. See the shader for the occlusion and overlap rules that keep a wide gather from
 * dragging a foreground across the background behind it.
 */
public final class RtDepthOfFieldPipeline {
    private static final String SHADER_DIR = "/caustica/shaders/pipelines/depth_of_field/";
    private static final int THREADS = 8;
    /** The upscale stage's workgroup, which differs because it runs at display resolution. */
    private static final int UPSCALE_THREADS = 16;

    private final RtContext ctx;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long descriptorSet;
    private final long pipelineLayout;
    private final long pipeline;
    private final long upscalePipeline;
    private final long blurPipeline;
    private final long sampler;
    private long boundGatheredSamplerView = -1;
    private long boundUpsampleView = -1;
    private long boundSharpSourceView = -1;
    private long boundSceneView = -1;
    private long boundDepthView = -1;
    private long boundMotionView = -1;
    private long boundGatheredView = -1;
    private long boundHistoryView = -1;
    private boolean destroyed;

    private RtDepthOfFieldPipeline(RtContext ctx, long dsl, long pool, long set, long layout, long pipeline,
                                   long upscalePipeline, long blurPipeline, long sampler) {
        this.ctx = ctx;
        this.descriptorSetLayout = dsl;
        this.descriptorPool = pool;
        this.descriptorSet = set;
        this.pipelineLayout = layout;
        this.pipeline = pipeline;
        this.upscalePipeline = upscalePipeline;
        this.blurPipeline = blurPipeline;
        this.sampler = sampler;
    }

    public static RtDepthOfFieldPipeline create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            // LINEAR: the gather's taps and its history reprojection both want a filtered read. CLAMP_TO_EDGE
            // only has to be well defined, because the shader drops out-of-frame taps itself.
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
            check(VK10.vkCreateSampler(vk, samplerInfo, null, pSampler), "vkCreateSampler(depth of field)");
            long sampler = pSampler.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, "depth of field linear sampler");

            VkDescriptorSetLayoutBinding.Buffer binds =
                    VkDescriptorSetLayoutBinding.calloc(DEPTH_OF_FIELD_BINDING_COUNT, stack);
            for (int binding : new int[] {DEPTH_OF_FIELD_SCENE, DEPTH_OF_FIELD_DEPTH,
                    DEPTH_OF_FIELD_MOTION, DEPTH_OF_FIELD_HISTORY,
                    DEPTH_OF_FIELD_GATHERED_SAMPLER, DEPTH_OF_FIELD_SHARP_SOURCE}) {
                binds.get(binding).binding(binding).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            for (int binding : new int[] {DEPTH_OF_FIELD_GATHERED, DEPTH_OF_FIELD_UPSAMPLE}) {
                binds.get(binding).binding(binding).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }

            VkDescriptorSetLayoutCreateInfo dslci = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(binds);
            LongBuffer p = stack.mallocLong(1);
            check(VK10.vkCreateDescriptorSetLayout(vk, dslci, null, p),
                    "vkCreateDescriptorSetLayout(depth of field)");
            long dsl = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, dsl,
                    "depth of field descriptor set layout");

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(2);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(5);
            VkDescriptorPoolCreateInfo dpci = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default()
                    .maxSets(1).pPoolSizes(poolSizes);
            check(VK10.vkCreateDescriptorPool(vk, dpci, null, p), "vkCreateDescriptorPool(depth of field)");
            long pool = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, pool, "depth of field descriptor pool");

            VkDescriptorSetAllocateInfo dsai = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                    .descriptorPool(pool).pSetLayouts(stack.longs(dsl));
            LongBuffer pSet = stack.mallocLong(1);
            check(VK10.vkAllocateDescriptorSets(vk, dsai, pSet), "vkAllocateDescriptorSets(depth of field)");
            long set = pSet.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, set, "depth of field descriptor set");

            VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
            pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0)
                    .size(DepthOfFieldPushData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo plci = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                    .pSetLayouts(stack.longs(dsl)).pPushConstantRanges(pushRange);
            check(VK10.vkCreatePipelineLayout(vk, plci, null, p), "vkCreatePipelineLayout(depth of field)");
            long layout = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, "depth of field pipeline layout");

            long gather = createPipeline(ctx, vk, stack, layout, "main.comp.spv", "main", "gather");
            long upscale = createPipeline(ctx, vk, stack, layout, "upscale.comp.spv", "main", "upscale");
            long blur = createPipeline(ctx, vk, stack, layout, "main.comp.spv", "blur", "blur");

            return new RtDepthOfFieldPipeline(ctx, dsl, pool, set, layout, gather, upscale, blur, sampler);
        }
    }

    /**
     * Point the pass at this frame's images: the reconstructed scene to gather from, the render-resolution depth
     * and motion guides, the half-resolution destination, and the previous frame's result as history. Views are
     * cached, so the steady state writes nothing.
     */
    /**
     * Point the pass at this frame's images.
     *
     * @param historyView the previous frame's gather result and this frame's history
     * @param destinationView this frame's gather target; it must not be the same image as {@code historyView},
     *                    because one is sampled and the other is written and a descriptor set cannot bind one
     *                    image both ways
     * @param upscaleView the display-resolution target of the upscale stage
     */
    public void setImages(long sceneView, long depthView, long motionView, long historyView, long destinationView,
                          long upscaleView, long sharpSourceView) {
        if (boundSceneView == sceneView && boundDepthView == depthView && boundMotionView == motionView
                && boundGatheredView == destinationView && boundHistoryView == historyView
                && boundUpsampleView == upscaleView && boundSharpSourceView == sharpSourceView) {
            return;
        }
        if (historyView == destinationView) {
            // The gather samples its history and writes its destination, so one descriptor set cannot hold both:
            // an image bound as a sampled image and as a storage image at once is invalid, and a gather reading
            // the image it is writing would be reading a half-written frame.
            throw new IllegalArgumentException("depth of field history and destination must be different images");
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            // One info and one write per binding: six sampled (scene, depth, motion, history, the gathered
            // image's sampling view, and the display-resolution scene the upscale hands back untouched) plus two
            // storage (the gathered image the gather writes, and the display-resolution image the upscale
            // writes).
            VkDescriptorImageInfo.Buffer infos = VkDescriptorImageInfo.calloc(8, stack);
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(8, stack);
            // The gather writes `gatheredView` while the upscale reads the OTHER scratch image, so the same
            // binding is a storage image for one stage and a sampled image for the other. Both are declared, and
            // each dispatch binds the one it needs: the gather never reads the gathered image, and the upscale
            // never writes it.
            // One info and one write per binding. Sampled: scene, depth, motion, history, the pyramid's three
            // extra levels, and the gathered image the upscale reads. Storage: the image the gather writes and
            // the display-resolution image the upscale writes. The gathered image appears at two bindings because
            // the stages use it differently -- the gather writes the frame it is producing, the upscale reads the
            // one the gather just finished -- and the two dispatches pass DIFFERENT images there, since a
            // descriptor set may not bind one image both ways.
            // Binding 3 is the gather's OWN target and is a storage image, so the image the upscale reads is
            // bound as a sampled image at binding 5 instead. A binding carries one descriptor type: giving the
            // gathered image both roles at binding 3 is invalid, and it would also be wrong, because the two
            // stages must be looking at different images -- the one being produced and the one just finished.
            int[] sampled = {DEPTH_OF_FIELD_SCENE, DEPTH_OF_FIELD_DEPTH, DEPTH_OF_FIELD_MOTION,
                    DEPTH_OF_FIELD_HISTORY, DEPTH_OF_FIELD_GATHERED_SAMPLER, DEPTH_OF_FIELD_SHARP_SOURCE};
            long[] views = {sceneView, depthView, motionView, historyView, historyView, sharpSourceView};
            for (int index = 0; index < sampled.length; index++) {
                infos.get(index).imageView(views[index]).sampler(sampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                writes.get(index).sType$Default().dstSet(descriptorSet).dstBinding(sampled[index])
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .pImageInfo(VkDescriptorImageInfo.create(infos.address(index), 1));
            }
            int[] storage = {DEPTH_OF_FIELD_GATHERED, DEPTH_OF_FIELD_UPSAMPLE};
            long[] storageViews = {destinationView, upscaleView};
            for (int index = 0; index < storage.length; index++) {
                int slot = sampled.length + index;
                infos.get(slot).imageView(storageViews[index]).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                writes.get(slot).sType$Default().dstSet(descriptorSet).dstBinding(storage[index])
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(VkDescriptorImageInfo.create(infos.address(slot), 1));
            }
            VK10.vkUpdateDescriptorSets(ctx.vk(), VkWriteDescriptorSet.create(writes.address(), 8), null);
        }
        boundSceneView = sceneView;
        boundDepthView = depthView;
        boundMotionView = motionView;
        boundGatheredView = destinationView;
        boundHistoryView = historyView;
        boundUpsampleView = upscaleView;
        boundSharpSourceView = sharpSourceView;
    }

    /** The optical parameters of one frame's gather. See the shader for what each one bounds. */
    /**
     * One frame's optics.
     *
     * @param targetRadiusPixels the circle of confusion, in display pixels
     * @param tracedRadiusPixels the part of the disc the primary trace already covered, in display pixels
     * @param focusDistance the focal plane, in blocks
     * @param apertureRadiusPixels the aperture's radius in display pixels: the disc the lens draws for a point
     *         at infinity, before the falloff
     * @param focalLength the focal length in blocks, which is what turns a surface's distance into its image
     *         distance
     */
    public record Params(float targetRadiusPixels, float tracedRadiusPixels, float focusDistance,
                         float apertureRadiusPixels, float focalLength, float foregroundTolerance,
                         float depthTolerance, float historyWeight, int resetToken, int frameIndex,
                         int tapCount) {
    }

    /**
     * Records the pass. {@code velocityScaleX/Y} convert the guides' render-pixel motion into gather-pixel
     * offsets; the projection terms turn the depth guide into view distances for the occluded-tap test.
     */
    public void dispatch(VkCommandBuffer cmd, int gatherWidth, int gatherHeight, int displayWidth,
                         int displayHeight, float velocityScaleX,
                         float velocityScaleY, float projM22, float projM32, Params params,
                         float gatherMinRadiusPixels, int blurRadius, int firstStage) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "depth of field")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
                    stack.longs(descriptorSet), null);
            ByteBuffer push = stack.malloc(DepthOfFieldPushData.BYTE_SIZE);
            writePush(push, gatherWidth, gatherHeight, displayWidth, params, gatherMinRadiusPixels, velocityScaleX,
                    velocityScaleY, projM22, projM32, firstStage, 0);
            VK10.vkCmdPushConstants(cmd, pipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (gatherWidth + THREADS - 1) / THREADS,
                    (gatherHeight + THREADS - 1) / THREADS, 1);
            VulkanCommandEncoder.memoryBarrier(cmd, stack);

            // The gather's residual smoothing, across and then down. Each pass reads the image the previous one
            // wrote and writes the other, which is what the two ping-pong images and the descriptor set's own
            // source/destination pair are already arranged for -- and it is why a call to setImages between them
            // would silently undo the separation.
            for (int stage = 1; stage <= 2 && blurRadius > 0; stage++) {
                VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, blurPipeline);
                ByteBuffer blurPush = stack.malloc(DepthOfFieldPushData.BYTE_SIZE);
                writePush(blurPush, gatherWidth, gatherHeight, displayWidth, params, gatherMinRadiusPixels, velocityScaleX,
                        velocityScaleY, projM22, projM32, stage, blurRadius);
                VK10.vkCmdPushConstants(cmd, pipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, blurPush);
                VK10.vkCmdDispatch(cmd, (gatherWidth + THREADS - 1) / THREADS,
                        (gatherHeight + THREADS - 1) / THREADS, 1);
                VulkanCommandEncoder.memoryBarrier(cmd, stack);
            }

            // The upscale reads what the gather just wrote, so it needs a second dispatch of the same set with
            // the other entry point. Its push is identical apart from the resolution it covers.
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, upscalePipeline);
            ByteBuffer upscalePush = stack.malloc(DepthOfFieldPushData.BYTE_SIZE);
            writePush(upscalePush, gatherWidth, gatherHeight, displayWidth, params, gatherMinRadiusPixels, velocityScaleX,
                    velocityScaleY, projM22, projM32, 0, 0);
            VK10.vkCmdPushConstants(cmd, pipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, upscalePush);
            VK10.vkCmdDispatch(cmd, (displayWidth + UPSCALE_THREADS - 1) / UPSCALE_THREADS,
                    (displayHeight + UPSCALE_THREADS - 1) / UPSCALE_THREADS, 1);
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
    }

    private static long createPipeline(RtContext ctx, VkDevice vk, MemoryStack stack, long layout, String file,
                                       String entry, String label) {
        long module = loadModule(vk, stack, file);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, module, "depth of field " + label + " module");
        VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack).sType$Default()
                .stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT).module(module).pName(stack.UTF8(entry));
        VkComputePipelineCreateInfo.Buffer cpci = VkComputePipelineCreateInfo.calloc(1, stack);
        cpci.get(0).sType$Default().stage(stage).layout(layout);
        LongBuffer pPipeline = stack.mallocLong(1);
        check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, cpci, null, pPipeline),
                "vkCreateComputePipelines(depth of field " + label + ")");
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, pPipeline.get(0), "depth of field " + label);
        VK10.vkDestroyShaderModule(vk, module, null);
        return pPipeline.get(0);
    }

    private static void writePush(ByteBuffer push, int gatherWidth, int gatherHeight, int displayWidth,
                                  Params params,
                                  float gatherMinRadiusPixels, float velocityScaleX, float velocityScaleY,
                                  float projM22, float projM32, int stage, int blurRadius) {
        float displayToGather = (float) gatherWidth / Math.max((float) displayWidth, 1.0f);
        new DepthOfFieldPushData(gatherWidth, gatherHeight, params.targetRadiusPixels(),
                params.tracedRadiusPixels(), params.focusDistance(), params.apertureRadiusPixels(),
                params.focalLength(), displayToGather, 0.0f, gatherMinRadiusPixels,
                params.foregroundTolerance(), params.depthTolerance(),
                params.historyWeight(), params.resetToken(), params.resetToken(), params.frameIndex(),
                params.tapCount(), velocityScaleX, velocityScaleY, projM22, projM32, stage,
                blurRadius).write(push);
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, pipeline, null);
        VK10.vkDestroyPipeline(vk, upscalePipeline, null);
        VK10.vkDestroyPipeline(vk, blurPipeline, null);
        VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
        VK10.vkDestroySampler(vk, sampler, null);
        destroyed = true;
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String name) {
        byte[] bytes;
        try (InputStream in = RtDepthOfFieldPipeline.class.getResourceAsStream(SHADER_DIR + name)) {
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
