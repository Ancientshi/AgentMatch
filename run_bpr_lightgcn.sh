python lightgcn.py \
  --data_root ./ \
  --epochs 10 --batch_size 4096 --emb_dim 128 --layers 3 \
  --neg_per_pos 1 --reg 1e-4 --topk 10 --device cuda:0 \
  --init_from_tfidf 0 --tfidf_fit_scope train --graph_scope train \
  --knn_N 8 \
