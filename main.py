import pandas as pd
import torch

from log import *
from eval import *
from utils import *
from train import *
#import numba
from module import CAWN, LinkPredictor
from graph import NeighborFinder
import resource
from tree_lstm import TreeLSTM
import warnings
warnings.filterwarnings("ignore")
args, sys_argv = get_args()
save_path = f'./log/{args.data}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
ATTN_NUM_HEADS = args.attn_n_head
DROP_OUT = args.drop_out
GPU = args.gpu
USE_TIME = args.time
ATTN_AGG_METHOD = args.attn_agg_method
ATTN_MODE = args.attn_mode
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
POS_ENC = args.pos_enc
POS_DIM = args.pos_dim
WALK_POOL = args.walk_pool
WALK_N_HEAD = args.walk_n_head
WALK_MUTUAL = args.walk_mutual if WALK_POOL == 'attn' else False
TOLERANCE = args.tolerance
CPU_CORES = args.cpu_cores
NGH_CACHE = args.ngh_cache
VERBOSITY = args.verbosity
AGG = args.agg
SEED = args.seed
N_RUNS = args.n_runs
TIME_WINDOWS = int(args.time_window[0])*int(args.time_window[1])
PERIOD_WINDOWS = int(args.period_window[0])*int(args.period_window[1])
assert(CPU_CORES >= -1)
set_random_seed(SEED)
logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv)

# Load data and sanity check
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
g_df.sort_values(by="ts" , inplace=True, ascending=True)
sorted_index = list(g_df.index)
if args.data_usage < 1:
    g_df = g_df.iloc[:int(args.data_usage*g_df.shape[0])]
    logger.info('use partial data, ratio: {}'.format(args.data_usage))
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))[sorted_index]
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values
max_idx = max(src_l.max(), dst_l.max())


if src_l.min()==0 or dst_l.min()==0:
    src_l += 1
    dst_l += 1
max_idx = max(src_l.max(), dst_l.max())

if DATA in ['enron', 'socialevolve', 'uci', 'lastfm', 'copenhagen','UNvote','Flights','mooc','Contacts','CanParl','USLegis','UNtrade']:
    node_zero_padding = np.zeros((n_feat.shape[0], 172 - n_feat.shape[1]))
    n_feat = np.concatenate([n_feat, node_zero_padding], axis=1)
    edge_zero_padding = np.zeros((e_feat.shape[0], 172 - e_feat.shape[1]))
    e_feat = np.concatenate([e_feat, edge_zero_padding], axis=1)

val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
if args.mode == 't':
    logger.info('Transductive training...')
    valid_train_flag = (ts_l <= val_time)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time

else:
    random.seed(2024)
    assert(args.mode == 'i')
    logger.info('Inductive training...')
    # pick some nodes to mask (i.e. reserved for testing) for inductive setting
    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    num_total_unique_nodes = len(total_node_set)
    mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes))) #timestamp属于验证集和测试集，的所有节点里，随机采样10%
    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_mask_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
    valid_train_flag = (ts_l <= val_time) * (none_mask_node_flag > 0.5)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time) * (none_mask_node_flag > 0.5)  # both train and val edges can not contain any masked nodes 训练集和验证集没有被mask掉的节点（没有新节点）
    valid_test_flag = (ts_l > test_time) * (none_mask_node_flag < 0.5)  # test edges must contain at least one masked node 测试集至少有一个新节点
    valid_test_new_new_flag = (ts_l > test_time) * mask_src_flag * mask_dst_flag
    valid_test_new_old_flag = (valid_test_flag.astype(int) - valid_test_new_new_flag.astype(int)).astype(bool)
    logger.info('Sampled {} nodes (10 %) which are masked in training and reserved for testing...'.format(len(mask_node_set)))

# split data according to the mask
train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = src_l[valid_train_flag], dst_l[valid_train_flag], ts_l[valid_train_flag], e_idx_l[valid_train_flag], label_l[valid_train_flag]
truncate = 28000
train_src_l = train_src_l[:truncate]
train_dst_l = train_dst_l[:truncate]
train_ts_l =train_ts_l[:truncate]
train_e_idx_l=train_e_idx_l[:truncate]
train_label_l=train_label_l[:truncate]
#计算当前周期窗口下，有多少个batch
diff = np.diff(train_ts_l)
count = 0
index = 0

batch_index = [0]
for d in diff:
    count += d
    index += 1
    if index == diff.shape[0]:
        batch_index.append(index + 2)
    elif count > PERIOD_WINDOWS:
        batch_index.append(index)
        count = 0

num_batch = len(batch_index) - 1
val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = src_l[valid_val_flag], dst_l[valid_val_flag], ts_l[valid_val_flag], e_idx_l[valid_val_flag], label_l[valid_val_flag]
test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = src_l[valid_test_flag], dst_l[valid_test_flag], ts_l[valid_test_flag], e_idx_l[valid_test_flag], label_l[valid_test_flag]
if args.mode == 'i':
    test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_e_idx_new_new_l, test_label_new_new_l = src_l[valid_test_new_new_flag], dst_l[valid_test_new_new_flag], ts_l[valid_test_new_new_flag], e_idx_l[valid_test_new_new_flag], label_l[valid_test_new_new_flag]
    test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_e_idx_new_old_l, test_label_new_old_l = src_l[valid_test_new_old_flag], dst_l[valid_test_new_old_flag], ts_l[valid_test_new_old_flag], e_idx_l[valid_test_new_old_flag], label_l[valid_test_new_old_flag]
train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l
val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l
train_val_data = (train_data, val_data)


full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(DATA,TIME_WINDOWS,full_adj_list, bias=args.bias,beta =args.beta,gama = args.gama, use_cache=NGH_CACHE, sample_method=args.pos_sample,full=True) #所有的链路
partial_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
for src, dst, eidx, ts in zip(val_src_l, val_dst_l, val_e_idx_l, val_ts_l):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
partial_ngh_finder = NeighborFinder(DATA,TIME_WINDOWS,partial_adj_list, bias=args.bias,beta =args.beta,gama = args.gama, use_cache=NGH_CACHE, sample_method=args.pos_sample,full=False)  #训练集和验证集的链路
ngh_finders = partial_ngh_finder, full_ngh_finder


# create random samplers to generate train/val/test instances
train_rand_sampler = RandEdgeSampler((train_src_l, ), (train_dst_l, ))
val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_dst_l, val_dst_l))
test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))
rand_samplers = train_rand_sampler, val_rand_sampler


# model initialization
device = torch.device('cuda:{}'.format(GPU))
# device = torch.device("mps")
metrics_transductive = {f'run{idx}': {} for idx in range(N_RUNS)}
metrics_new_new = {f'run{idx}': {} for idx in range(N_RUNS)}
metrics_new_old = {f'run{idx}': {} for idx in range(N_RUNS)}
for i in range(N_RUNS):
    start_time_run = time.time()
    logger.info("************************************")
    logger.info("********** Run {} starts. **********".format(i))
    cawn = CAWN(args,device,n_feat, e_feat,TIME_WINDOWS, agg=AGG,
                num_layers=NUM_LAYER, use_time=USE_TIME, attn_agg_method=ATTN_AGG_METHOD, attn_mode=ATTN_MODE,
                n_head=ATTN_NUM_HEADS, drop_out=DROP_OUT, pos_dim=POS_DIM, pos_enc=POS_ENC,
                num_neighbors=NUM_NEIGHBORS, walk_n_head=WALK_N_HEAD, walk_mutual=WALK_MUTUAL, walk_linear_out=args.walk_linear_out, walk_pool=args.walk_pool,
                cpu_cores=CPU_CORES, verbosity=VERBOSITY, get_checkpoint_path=get_checkpoint_path)
    cawn.to(device)
    optimizer = torch.optim.Adam(cawn.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

    start = time.time()
    # start train and val phases
    train_val(args,PERIOD_WINDOWS,train_val_data, cawn, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, ngh_finders, rand_samplers, logger)
    end = time.time()
    # final testing
    cawn.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finderargs, model,predictor
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for {} nodes'.format(args.mode),args, cawn, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l)
    logger.info('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_acc, test_auc, test_ap))
    test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1]*6
    if args.mode == 'i':
        test_new_new_acc, test_new_new_ap, test_new_new_f1, test_new_new_auc = eval_one_epoch('test for {} nodes'.format(args.mode),args, cawn, test_rand_sampler, test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_label_new_new_l, test_e_idx_new_new_l)
        logger.info('Test statistics: {} new-new nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_new_acc, test_new_new_auc,test_new_new_ap ))
        test_new_old_acc, test_new_old_ap, test_new_old_f1, test_new_old_auc = eval_one_epoch('test for {} nodes'.format(args.mode), args, cawn,test_rand_sampler, test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_label_new_old_l, test_e_idx_new_old_l)
        logger.info('Test statistics: {} new-old nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_old_acc, test_new_old_auc, test_new_old_ap))
    metrics_transductive[f'run{i}']['accuracy'] = test_acc
    metrics_transductive[f'run{i}']['auc'] = test_auc
    metrics_transductive[f'run{i}']['ap'] = test_ap
    metrics_new_new[f'run{i}']['accuracy'] = test_new_new_acc
    metrics_new_new[f'run{i}']['auc'] = test_new_new_auc
    metrics_new_new[f'run{i}']['ap'] = test_new_new_ap
    metrics_new_old[f'run{i}']['accuracy'] = test_new_old_acc
    metrics_new_old[f'run{i}']['auc'] = test_new_old_auc
    metrics_new_old[f'run{i}']['ap'] = test_new_old_ap
    # save model
    logger.info('Saving CAWN model ...')
    torch.save(cawn.state_dict(), best_model_path)
    logger.info('CAWN model saved')

    # save one line result
    save_oneline_result('log/', args, [test_acc, test_auc, test_ap, test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc])

accuracies = [result['accuracy'] for result in metrics_transductive.values()]
aucs = [result['auc'] for result in metrics_transductive.values()]
aps = [result['ap'] for result in metrics_transductive.values()]

# 计算平均值
mean_accuracy = np.round(np.mean(accuracies),4)
mean_auc = np.round(np.mean(aucs),4)
mean_ap = np.round(np.mean(aps),4)

# 计算标准差
std_accuracy = np.round(np.std(accuracies),3)
std_auc = np.round(np.std(aucs),3)
std_ap = np.round(np.std(aps),3)

logger.info(f"All Accuracy:{mean_accuracy}±{std_accuracy}")
logger.info(f"ALL AUC:{mean_auc}±{std_auc}")
logger.info(f"ALL AP:{mean_ap}±{std_ap}")

accuracies = [result['accuracy'] for result in metrics_new_new.values()]
aucs = [result['auc'] for result in metrics_new_new.values()]
aps = [result['ap'] for result in metrics_new_new.values()]

# 计算平均值
mean_accuracy = np.round(np.mean(accuracies),4)
mean_auc = np.round(np.mean(aucs),4)
mean_ap = np.round(np.mean(aps),4)

# 计算标准差
std_accuracy = np.round(np.std(accuracies),3)
std_auc = np.round(np.std(aucs),3)
std_ap = np.round(np.std(aps),3)

logger.info(f"New_New_Accuracy:{mean_accuracy}±{std_accuracy}")
logger.info(f"New_New_AUC:{mean_auc}±{std_auc}")
logger.info(f"New_New_AP:{mean_ap}±{std_ap}")

accuracies = [result['accuracy'] for result in metrics_new_old.values()]
aucs = [result['auc'] for result in metrics_new_old.values()]
aps = [result['ap'] for result in metrics_new_old.values()]

# 计算平均值
mean_accuracy = np.round(np.mean(accuracies),4)
mean_auc = np.round(np.mean(aucs),4)
mean_ap = np.round(np.mean(aps),4)

# 计算标准差
std_accuracy = np.round(np.std(accuracies),3)
std_auc = np.round(np.std(aucs),3)
std_ap = np.round(np.std(aps),3)

logger.info(f"New_Old_Accuracy:{mean_accuracy}±{std_accuracy}")
logger.info(f"New_Old_AUC:{mean_auc}±{std_auc}")
logger.info(f"New_Old_AP:{mean_ap}±{std_ap}")