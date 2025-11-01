python SimGCL.py \
  --data_root ./ \
  --epochs 10 --batch_size 4096 --emb_dim 128 --layers 3 \
  --neg_per_pos 1 --reg 1e-4 --topk 10 --device cuda:0 \
  --ssl_lambda 0.1 --ssl_temp 0.2 --ssl_noise 0.1 \
  --ssl_users 4096 --ssl_items 4096 \
  --graph_scope train \
  --init_from_tfidf 0 --tfidf_fit_scope train \
  --eval_cand_size 1000 --knn_N 8 \
  --rebuild_cache 0 --rebuild_training_cache 0
