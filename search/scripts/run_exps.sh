for model in gpt-4o-mini gpt-4o deepseek-coder sonnet-3-5
do
    for dataset in livecodebench_lite_v3 mbpp_plus human_eval_plus 
    do
        if [[ "$dataset" == "mbpp_plus" || "$dataset" == "human_eval_plus" ]]; then
            testbank="codegenning/B_${dataset}_v2"
        else
            testbank="codegenning/B_${dataset}"
        fi

        SEARCH_ALG="combo_observation" \
            python eval.py \
            --idea-model-config-path model_configs/${model}.json \
            --code-model-config-path model_configs/${model}.json \
            --output final_results/${dataset}/combo_observation_no_${model}_temp0.9 \
            --experiment-directory other_logs/final_logs/${dataset}/combo_observation_no_${model}_temp0.9 \
            --dataset codegenning/F_${dataset} \
            --completion-limit -1 \
            --idea-temperature 0.9 \
            --code-temperature 0.9 \
            --top-p 0.95 \
            --split test \
            --testbank $testbank \
            --cache-file caches/${model}_${dataset}_combo_observation_cache.json \
            --exec-type both \
            --exec-batch-size 18000 \
            --exec-num-processes 180 \
            --completions-from-model

        SEARCH_ALG="basic_prompting" \
            python eval.py \
            --model-config-path model_configs/${model}.json \
            --output final_results/${dataset}/basic_prompting225_${model}_temp0.9 \
            --experiment-directory other_logs/final_logs/${dataset}/basic_prompting225_${model}_temp0.9 \
            --dataset codegenning/F_${dataset} \
            --completion-limit 225 \
            --temperature 0.9 \
            --top-p 0.95 \
            --split test \
            --testbank $testbank \
            --cache-file caches/${model}_${dataset}_cache.json \
            --exec-type both \
            --exec-batch-size 18000 \
            --exec-num-processes 180 

        SEARCH_ALG="simple_idea" \
            python eval.py \
            --idea-model-config-path model_configs/${model}.json \
            --code-model-config-path model_configs/${model}.json \
            --output final_results/${dataset}/simple_idea225_${model}_temp0.9 \
            --experiment-directory other_logs/final_logs/${dataset}/simple_idea225_${model}_temp0.9 \
            --dataset codegenning/F_${dataset} \
            --completion-limit 225 \
            --idea-temperature 0.9 \
            --code-temperature 0.9 \
            --top-p 0.95 \
            --split test \
            --testbank $testbank \
            --cache-file caches/${model}_${dataset}_cache.json \
            --exec-type both \
            --exec-batch-size 18000 \
            --exec-num-processes 180 
    done
done
