for dataset in mbpp_plus human_eval_plus livecodebench_lite_v3
do
    for model in gpt-4o-mini gpt-4o deepseek-coder sonnet-3-5
    # for model in baby-deepseek-i_sgl baby-deepseek-b_sgl
    # for model in llama318b_sgl
    do
        # python scripts/check_similar.py \
        #     --model-config-path model_configs/gpt-4o-mini.json \
        #     --results-file final_results/${dataset}/basic_prompting225_${model}_temp0.9 \
        #     --log-directory other_logs/similar_logs/right_wrong_logs/${dataset}/basic_prompting225_${model}_temp0.9 \
        #     --cache-file caches/${dataset}_${model}_basic_prompting_right_wrong_check_similar_cache.json \
        #     --comparison-type right-wrong \
        #     --max-to-consider 40
        python scripts/check_similar.py \
            --model-config-path model_configs/gpt-4o-mini.json \
            --results-file final_results/${dataset}/basic_prompting225_${model}_temp0.9 \
            --log-directory other_logs/similar_logs/right_logs/${dataset}/basic_prompting225_${model}_temp0.9 \
            --cache-file caches/${dataset}_${model}_basic_prompting_right_wrong_check_similar_cache.json \
            --comparison-type right \
            --max-to-consider 40
        python scripts/check_similar.py \
            --model-config-path model_configs/gpt-4o-mini.json \
            --results-file final_results/${dataset}/basic_prompting225_${model}_temp0.9 \
            --log-directory other_logs/similar_logs/wrong_logs/${dataset}/basic_prompting225_${model}_temp0.9 \
            --cache-file caches/${dataset}_${model}_basic_prompting_right_wrong_check_similar_cache.json \
            --comparison-type wrong \
            --max-to-consider 40


        # python scripts/check_similar.py \
        #     --model-config-path model_configs/gpt-4o-mini.json \
        #     --results-file final_results/${dataset}/simple_idea225_${model}_temp0.9 \
        #     --log-directory other_logs/similar_logs/final_logs/${dataset}/simple_idea225_${model}_temp0.9 \
        #     --cache-file caches/check_similar_cache.json \
        #     --max-to-consider 40

        # python scripts/check_similar.py \
        #     --model-config-path model_configs/gpt-4o-mini.json \
        #     --results-file final_results/${dataset}/combo_observation_no_${model}_temp0.9 \
        #     --log-directory other_logs/similar_logs/final_logs/${dataset}/combo_observation_no_${model}_temp0.9 \
        #     --cache-file caches/${dataset}_${model}_combo_observation_no_check_similar_cache.json \
        #     --max-to-consider 40

        # python scripts/check_similar.py \
        #     --model-config-path model_configs/gpt-4o-mini.json \
        #     --results-file final_results/base_v_instruct/${dataset}/basic_prompting225_${model}_temp0.9 \
        #     --log-directory other_logs/similar_logs/final_logs/base_v_instruct/${dataset}/basic_prompting225_${model}_temp0.9 \
        #     --cache-file caches/${dataset}_${model}_basic_prompting_check_similar_cache.json \
        #     --max-to-consider 40
    done
done
