python bpr_tfidf_bert.py \
  --data_root ./ \
  --epochs 3 --batch_size 256 --max_features 5000 \
  --pretrained_model distilbert-base-uncased --max_len 128 \
  --neg_per_pos 1 --topk 10 --device cuda:0

