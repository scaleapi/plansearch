# for model in gpt-4o-mini gpt-4o deepseek-coder sonnet-3-5
# for model in baby-deepseek-b_sgl
# for model in llama3170bi_sgl
for model in llama3170b_sgl
do
    for dataset in mbpp_plus human_eval_plus livecodebench_lite_v3
    do
        if [[ "$dataset" == "mbpp_plus" || "$dataset" == "human_eval_plus" ]]; then
            testbank="codegenning/B_${dataset}_v2"
        else
            testbank="codegenning/B_${dataset}"
        fi

        SEARCH_ALG="basic_prompting" \
            python eval.py \
            --model-config-path model_configs/${model}.json \
            --output final_results/base_v_instruct/${dataset}/basic_prompting225_${model}_temp0.9 \
            --experiment-directory other_logs/final_logs/base_v_instruct/${dataset}/basic_prompting225_${model}_temp0.9 \
            --dataset codegenning/F_${dataset} \
            --completion-limit 225 \
            --temperature 0.9 \
            --top-p 0.95 \
            --split test \
            --testbank $testbank \
            --cache-file caches/${model}_${dataset}_cache.json \
            --exec-type both \
            --exec-batch-size 325 \
            --global-batch-size 12288
            

        # SEARCH_ALG="simple_idea" \
        #     python eval.py \
        #     --idea-model-config-path model_configs/${model}.json \
        #     --code-model-config-path model_configs/${model}.json \
        #     --output final_results/${dataset}/simple_idea225_${model}_temp0.9 \
        #     --experiment-directory other_logs/final_logs/${dataset}/simple_idea225_${model}_temp0.9 \
        #     --dataset codegenning/F_${dataset} \
        #     --completion-limit 225 \
        #     --idea-temperature 0.9 \
        #     --code-temperature 0.9 \
        #     --top-p 0.95 \
        #     --split test \
        #     --testbank $testbank \
        #     --cache-file caches/${model}_${dataset}_cache.json \
        #     --exec-type both \
        #     --exec-batch-size 275
    done
done
