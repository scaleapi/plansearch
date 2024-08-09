trap 'kill $SERVER_PID $NEXT_SERVER_PID; pkill -f "sglang.launch_server.* --port $port_even" -u evanwang; pkill -f "sglang.launch_server.* --port $port_odd" -u evanwang; exit' INT

model_path_base="/mnt/efs/evanwang/model_weights/dpo/checks/checkpoint-"
odd_even_counter=0

port_even=30000
port_odd=30011

iteration_limit=9
increment_value=3
i=3

echo "Starting initial server launch..."
if [ $((odd_even_counter % 2)) -eq 0 ]; then
    echo "Launching server on port $port_even with GPUs 0,1,2,3"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server --model-path ${model_path_base}$i --port $port_even --tp 4 --mem-fraction-static 0.5 &
else
    echo "Launching server on port $port_odd with GPUs 4,5,6,7"
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server --model-path ${model_path_base}$i --port $port_odd --tp 4 --mem-fraction-static 0.5 &
fi
SERVER_PID=$!
echo "Initial server launched with PID: $SERVER_PID"

while [ $i -le $iteration_limit ]
do
  echo "Processing iteration with i=$i"
  if [ $i -lt $iteration_limit ]; then
    if [ $(((odd_even_counter + 1) % 2)) -eq 0 ]; then
      echo "Launching next server on port $port_even with GPUs 0,1,2,3"
      (sleep 30 && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server --model-path ${model_path_base}$((i+increment_value)) --port $port_even --tp 4 --mem-fraction-static 0.5 & echo $! > next_server_pid.txt) &
    else
      echo "Launching next server on port $port_odd with GPUs 4,5,6,7"
      (sleep 30 && CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server --model-path ${model_path_base}$((i+increment_value)) --port $port_odd --tp 4 --mem-fraction-static 0.5 & echo $! > next_server_pid.txt) &
    fi
    sleep 1  # Give some time for the PID to be written
    NEXT_SERVER_PID=$(cat next_server_pid.txt)
    echo "Next server launched with PID: $NEXT_SERVER_PID"
  fi

  echo "Running evaluation script for i=$i"
  SEARCH_ALG="basic_prompting" python eval.py --model-config-path model_configs/model_configs/hsg_$((odd_even_counter % 2)).json --output test_results/test_$i --dataset codegenning/F_livecodebench_lite_v2_lite35 --completion-limit 1 --max-tokens 4096 --temperature 0 --top-p 0.9 --num-shots 1 --split test --testbank codegenning/B_livecodebench_lite_v2
  
  echo "Killing server with PID: $SERVER_PID"
  kill $SERVER_PID

  if [ $((odd_even_counter % 2)) -eq 0 ]; then
    pkill -f "sglang.launch_server.* --port $port_even" -u evanwang
  else
    pkill -f "sglang.launch_server.* --port $port_odd" -u evanwang
  fi
  
  if [ $i -lt $iteration_limit ]; then
    SERVER_PID=$NEXT_SERVER_PID
    echo "Switching to next server with PID: $SERVER_PID"
  fi

  odd_even_counter=$((odd_even_counter + 1))
  echo "Incremented odd_even_counter to $odd_even_counter"
  i=$((i + increment_value))
done
