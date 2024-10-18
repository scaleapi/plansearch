# Run in models/research/evan/search: bash scripts/run_filter.sh
SEARCH_ALG="simple_filter" python scale_lcb_eval.py --model gpt-4-turbo --output test_results/test_filter_small --dataset codegenning/livecodebench_lite_v2_lite35 --completion-limit 1 --gen-batch-size 2 --num-batches-to-try 2 --cache-file cache.json --temperature 0.3
# Change ~/src/CodeRM/coderm/eval/metrics.py to the path of the metrics.py file
python ~/src/CodeRM/coderm/eval/metrics.py test_results/test_filter_small
# Output should be roughly 40-50%-ish pass@1
