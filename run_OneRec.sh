export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# python OneRec.py \
#     --data_root ./ \
#     --epochs 10 \
#     --mode gen --topk 10 --device cuda:0 \
#     --use_sessions 1 --session_len 4 \
#     --train_mask 0 --candidate_size 200



python OneRec.py \
  --data_root ./ \
  --lr 1e-5 \
  --mode dpo --topk 10 --device cuda:0 \
  --dpo_steps 1200 --dpo_batch 64 \
  --max_features 5000 \
  --use_sessions 1 --session_len 4 \
  --amp 1 --enc_chunk 512 \
  --train_mask 0 --candidate_size 200
