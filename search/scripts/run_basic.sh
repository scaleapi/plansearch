# Run in models/research/evan/search: bash scripts/run_basic.sh
SEARCH_ALG="basic_prompting" python scale_lcb_eval.py --model gpt-4-turbo --output test_results/test_small --dataset codegenning/livecodebench_lite_v2_lite35 --completion-limit 2
# Change ~/src/CodeRM/coderm/eval/metrics.py to the path of the metrics.py file
python ~/src/CodeRM/coderm/eval/metrics.py test_results/test_small
# Output should be roughly 35-40%-ish pass@1
