python two_tower_tfidf_bert.py \
  --data_root ./ \
  --epochs 20 --batch_size 512 --max_features 5000 \
  --hid 256 --topk 10 --device cuda:0 --eval_chunk 8192 \
  --pretrained_model distilbert-base-uncased --max_len 128 \
  --use_tool_emb 1 \
