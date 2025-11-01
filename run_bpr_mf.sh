python bpr_mf.py \
  --data_root ./ \
  --epochs 5 --batch_size 4096 --factors 128 \
  --neg_per_pos 1 --topk 10 --device cuda:0 \
  --eval_cand_size 1000 --knn_N 8 
