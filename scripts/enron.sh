cd ../
python -u main.py \
--data enron \
--pos_dim 108 \
--n_degree 64 1 \
--n_layer 3 \
--mode i \
--bias 1e-6 \
--gama 10 \
--pos_enc lp \
--seed 0 \
--n_epoch 50 \
--lr 0.0001 \
--period_window 1800 1 \
--use_period \
--j 10 \
--mode t