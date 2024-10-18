#!/bin/bash
for i in {0..2800..200}
do
    # config_file="model_configs/model_configs_4/llama1e-8_${i}.json"
    # config_file="model_configs/model_configs_4/llamashuf_${i}.json"
    # config_file="model_configs/model_configs_4/llamafix1e-7_${i}.json"
    config_file="model_configs/custom_vllm.json"
    # CUDA_VISIBLE_DEVICES=4,5,6,7 \
    SEARCH_ALG="basic_prompting" \
    python eval.py \
        --model-config-path ${config_file} \
        --model-model-name /mnt/efs/evanwang/model_weights/dpo/me/work_reward1/checkpoint-${i} \
        --model-tensor-parallel-size 4 \
        --output test_results/ch_workreward1_${i} \
        --dataset codegenning/F_livecodebench_lite_v2 \
        --completion-limit 1 \
        --temperature 0 \
        --top-p 0.9 \
        --num-shots 1 \
        --cache-file caches/work_cache.json \
        --split test \
        --testbank codegenning/B_livecodebench_lite_v2
done
