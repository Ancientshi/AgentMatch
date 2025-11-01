
# 训练并在若干样例问题上打印 Top-K 推荐
python bpr_tfidf.py \
  --data_root ./ \
  --epochs 5 \
  --batch_size 256 \
  --max_features 5000 \
  --neg_per_pos 1 \
  --topk 10 \
  --device cuda:0 \


