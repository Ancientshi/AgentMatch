
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python KALM.py \
  --data_root ./ \
  --save_dir  ./KALM \
  --model_name KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5 \
  --epochs 1 --batch_size 16 --lr 2e-4 --warmup_ratio 0.05 \
  --pos_topk 5 --pairs_per_q 5 --valid_ratio 0.2 \
  --tune_mode lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --use_amp 1 --grad_accum_steps 2 --eval_steps 2000 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
