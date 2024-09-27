# transformer
python3 ./chameleon_train.py \
--model_fn ./models/chameleon.transformer.pth \
--train_fn ./data/chameleon.train.tok.pickle \
--valid_fn ./data/chameleon.valid.tok.pickle \
--src_tgt koen \
--with_text 1 \
--gpu_id 0 \
--batch_size 64 \
--n_epochs 5 \
--max_length 512 \
--dropout .2 \
--hidden_size 768 \
--n_layers 4 \
--max_grad_norm 1e+8 \
--iteration_per_update 2 \
--lr 1e-3 \
--lr_step 0 \
--use_adam \
--use_transformer \
--use_mps \
--rl_n_epochs 0

# # seq2seq
# python3 ./item_converter_train.py \
# --model_fn ./models/item.converter.seq2seq.pth \
# --train_fn ./data/item.converter.train.tsv \
# --valid_fn ./data/item.converter.valid.tsv \
# --with_text 0 \
# --gpu_id 0 \
# --batch_size 256 \
# --n_epochs 10 \
# --max_length 256 \
# --dropout .2 \
# --word_vec_size 512 \
# --hidden_size 768 \
# --n_layers 4 \
# --max_grad_norm 1e+8 \
# --iteration_per_update 2 \
# --lr 1e-3 \
# --lr_step 0 \
# --use_adam \
# --rl_n_epochs 0