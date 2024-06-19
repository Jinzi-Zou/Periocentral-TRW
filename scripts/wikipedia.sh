cd ../
python -u main.py \
--data wikipedia \
--pos_dim 108 \
--n_degree 64 1 \
--n_layer 2 \
--bias 1e-7 \
--gama 10 \
--pos_enc lp \
--seed 0 \
--n_epoch 1 \
--lr 0.0001 \
--link_out_hidden 256 \
--tree_h_size 256 \
--period_window 900  1 \
--use_period \
--j 5 \
--mode i