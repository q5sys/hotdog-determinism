#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

iteration=0

while true; do
    echo "Starting iteration $iteration..."
    
    python -m ltx_pipelines.ti2vid_two_stages \
        --checkpoint-path /zstor/ai/ltx2/LTX-2/ltx-2-19b-dev.safetensors \
        --distilled-lora /zstor/ai/ltx2/LTX-2/ltx-2-19b-distilled-lora-384.safetensors \
        --spatial-upsampler-path /zstor/ai/ltx2/LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors \
        --gemma-root /zstor/ai/ltx2/ComfyUI/models/text_encoders/gemma_3_12B/ \
        --prompt "a hotdog walking down the street" \
        --output-path "hotdog-gpu0-${iteration}.mp4" \
        --height 320 \
        --width 640 \
        --frame-rate 24 \
        --num-frames 240 \
        --num-inference-steps 20 \
        --cfg-guidance-scale 4 \
        --seed "$iteration" \
        --enable-fp8

    python -m ltx_pipelines.ti2vid_two_stages \
        --checkpoint-path /zstor/ai/ltx2/LTX-2/ltx-2-19b-dev.safetensors \
        --distilled-lora /zstor/ai/ltx2/LTX-2/ltx-2-19b-distilled-lora-384.safetensors \
        --spatial-upsampler-path /zstor/ai/ltx2/LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors \
        --gemma-root /zstor/ai/ltx2/ComfyUI/models/text_encoders/gemma_3_12B/ \
        --prompt "a hotdog walking down the street" \
        --output-path "hotdog-gpu0-${iteration}-2.mp4" \
        --height 320 \
        --width 640 \
        --frame-rate 24 \
        --num-frames 240 \
        --num-inference-steps 20 \
        --cfg-guidance-scale 4 \
        --seed "$iteration" \
        --enable-fp8

    echo "Finished iteration $iteration"
    iteration=$((iteration + 1))
done
