python two_tower.py \
   --data_root ./ \
  --epochs 5 --batch_size 512 --max_features 5000 \
  --hid 256 --temperature 0.07 --topk 10 --device cuda:0 --eval_chunk 8192 \
  --use_tool_emb 1 \
  --use_agent_id_emb 1 \
  --agent_id_weight 0.5 \
