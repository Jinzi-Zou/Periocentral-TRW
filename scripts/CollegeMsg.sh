cd ../
python -u main.py \
--data CollegeMsg \
--pos_dim 100 \
--n_degree 64 1 \
--n_layer 2 \
--bias 1e-6 \
--gama 15 \
--pos_enc lp \
--seed 0 \
--n_epoch 1 \
--lr 0.0001 \
--drop_out 0.5 \
--link_dropout 0.5 \
--link_out_hidden 256 \
--tree_h_size 256 \
--bs 32 \
--period_window 1800 1 \
--use_period \
--j 5 \
--mode i
