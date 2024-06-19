# Temporal Random Walks with the Awareness of Node Centrality and Link Periodicity for Dynamic Link Prediction

## Abstract



## Requirements

- python==3.8
- matplotlib==3.7.4
- networkx==3.1
- numba==0.58.1
- scikit-learn==1.3.2
- torch==1.12.1
- pandas==2.0.3
- numpy==1.24.4

To install all dependencies:

```setup
pip install -r requirements.txt
```

## Training

To train the model in the paper, run this command （here we provide three examples）：

```train
python main.py --data CollegeMsg --pos_dim 100 --n_degree 64 1 --n_layer 2 --bias 1e-6 --gama 15 --pos_enc lp --seed 0 --n_epoch 1 --lr 0.0001 --drop_out 0.5 --link_dropout 0.5 --link_out_hidden 256 --tree_h_size 256 --bs 32 --period_window 1800 1 --use_period --j 5 --mode i
```

```
python main.py --data enron --pos_dim 108 --n_degree 64 1 --n_layer 3 --mode i 
--bias 1e-6 --gama 10 --pos_enc lp --seed 0 --n_epoch 50 --lr 0.0001 --period_window 1800 1 --use_period --j 10 --mode i
```

```
python main.py --data wikipedia --pos_dim 108 --n_degree 64 1 --n_layer 2 --bias 1e-7 --gama 10 --pos_enc lp --seed 0 --n_epoch 1 --lr 0.0001 --link_out_hidden 256 --tree_h_size 256 --period_window 900  1 --use_period --j 5 --mode i
```
