python BGE_two_tower.py \
  --data_root ./ \
  --epochs 10 --batch_size 512 \
  --eval_chunk 8192 --device cuda:0 \
  --embed_url http://127.0.0.1:8502/get_embedding \
  --embed_batch 64 \
  --use_tool_emb 1 \
