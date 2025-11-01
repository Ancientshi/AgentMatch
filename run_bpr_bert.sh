# # Basic training
# python bpr_bert.py \
#   --data_root ./ \
#   --epochs 10 \
#   --batch_size 256 \
#   --pretrained_model distilbert-base-uncased \
#   --max_len 128 \
#   --text_hidden 256 \
#   --id_dim 64 \
#   --neg_per_pos 1 \
#   --topk 10 \
#   --pooling cls --device cuda:0


# last n layers unfrozen
# python bpr_bert.py \
#   --data_root ./ \
#   --tune_mode full --unfreeze_last_n 2 --unfreeze_emb 1 \
#   --encoder_lr 5e-5 \
#   --epochs 3 \
#   --batch_size 64 \
#   --pretrained_model distilbert-base-uncased \
#   --max_len 128 \
#   --text_hidden 256 \
#   --id_dim 64 \
#   --neg_per_pos 1 \
#   --topk 10 \
#   --pooling cls --device cuda:0

# LoRA 
python bpr_bert.py \
  --data_root ./ \
  --tune_mode lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 \
  --lora_targets query,key,value,dense --encoder_lr 5e-5 \
  --epochs 3 --batch_size 256 --device cuda:1
