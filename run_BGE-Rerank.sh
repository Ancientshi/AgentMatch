
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python BGE-Rerank.py \
  --data_root ./ \
  --save_dir ./BGE-Rerank \
  --triples_cache ./BGE-Rerank/triples_cache.npy.gz \
  --model_name BAAI/bge-reranker-base \
  --epochs 1 \
  --batch_size 8 \
  --accum_steps 2 \
  --lr 2e-5 \
  --max_len 192 \
  --hard_neg_per_pos 2 \
  --rand_neg_per_pos 1 \
  --tfidf_max_features 20000 \
  --valid_ratio 0.2 \
  --device cuda:0 \
  --eval_cand_size 1000 \
  --recall_topk 200 \
  --rerank_batch 256 \
  --use_lora 1 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --grad_ckpt 1 \