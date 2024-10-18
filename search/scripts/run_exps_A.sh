
# idea_algorithms=("idea_filter" "simple_idea")
# simple_algorithms=("basic_prompting" "simple_filter")


# for SEARCH_ALG in "${idea_algorithms[@]}"
# do
#     python eval.py \
#         --idea-model-config-path model_configs/gpt-4o-mini.json \
#         --code-model-config-path model_configs/gpt-4o-mini.json \
#         --output final_results/${SEARCH_ALG}64_4omi_temp0.8 \
#         --experiment-directory other_logs/final_logs/${SEARCH_ALG}64_4omi_temp0.8 \
#         --dataset codegenning/F_livecodebench_lite_v2 \
#         --completion-limit 10 \
#         --gen-batch-size 4 \
#         --num-batches-to-try 16 \
#         --code-temperature 0.8 \
#         --idea-temperature 0.8 \
#         --top-p 0.95 \
#         --split test \
#         --testbank codegenning/B_livecodebench_lite_v2 \
#         --cache-file caches/4omi_cache.json
# done

for dataset in mbpp_plus human_eval_plus livecodebench_lite_v3
do
    # for model in gpt-4o-mini gpt-4o deepseek-coder
    for model in sonnet-3-5
    do
        SEARCH_ALG="basic_prompting" \
            python eval.py \
            --model-config-path model_configs/${model}.json \
            --output final_results/${dataset}/default_${model}_temp0.8 \
            --experiment-directory other_logs/final_logs/${dataset}/default_${model}_temp0.8 \
            --dataset codegenning/F_${dataset} \
            --completion-limit 50 \
            --temperature 0.8 \
            --top-p 0.95 \
            --split test \
            --testbank codegenning/B_${dataset} \
            --cache-file caches/${model}_cache.json

        SEARCH_ALG="simple_idea" \
            python eval.py \
            --idea-model-config-path model_configs/${model}.json \
            --code-model-config-path model_configs/${model}.json \
            --output final_results/${dataset}/simple_idea_${model}_temp0.8 \
            --experiment-directory other_logs/final_logs/${dataset}/simple_idea_${model}_temp0.8 \
            --dataset codegenning/F_${dataset} \
            --completion-limit 50 \
            --idea-temperature 0.8 \
            --code-temperature 0.8 \
            --top-p 0.95 \
            --split test \
            --testbank codegenning/B_${dataset} \
            --cache-file caches/${model}_cache.json
        # SEARCH_ALG="simple_filter" \
        #     python eval.py \
        #     --model-config-path model_configs/${model}.json \
        #     --output final_results/${dataset}/simple_filter100_${model}_temp0.8 \
        #     --experiment-directory other_logs/final_logs/${dataset}/simple_filter100_${model}_temp0.8 \
        #     --dataset codegenning/F_${dataset} \
        #     --completion-limit 25 \
        #     --temperature 0.8 \
        #     --gen-batch-size 10 \
        #     --num-batches-to-try 10 \
        #     --top-p 0.95 \
        #     --split test \
        #     --testbank codegenning/B_${dataset} \
        #     --cache-file caches/${model}_cache.json

        # SEARCH_ALG="idea_filter" \
        #     python eval.py \
        #     --idea-model-config-path model_configs/${model}.json \
        #     --code-model-config-path model_configs/${model}.json \
        #     --output final_results/${dataset}/idea_filter100_${model}_temp0.8 \
        #     --experiment-directory other_logs/final_logs/${dataset}/idea_filter100_${model}_temp0.8 \
        #     --dataset codegenning/F_${dataset} \
        #     --completion-limit 25 \
        #     --idea-temperature 0.8 \
        #     --code-temperature 0.8 \
        #     --gen-batch-size 10 \
        #     --num-batches-to-try 10 \
        #     --top-p 0.95 \
        #     --split test \
        #     --testbank codegenning/B_${dataset} \
        #     --cache-file caches/${model}_cache.json
    done
done
