import torch
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from eval import *
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True


def train_val(args,PEIROD_WINDOWS,train_val_data, model, mode, bs, epochs, criterion, optimizer, early_stopper, ngh_finders, rand_samplers, logger):
    # unpack the data, prepare for the training
    train_data, val_data = train_val_data
    train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = train_data
    val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = val_data
    train_rand_sampler, val_rand_sampler = rand_samplers
    partial_ngh_finder, full_ngh_finder = ngh_finders
    if mode == 't':  # transductive
        model.update_ngh_finder(full_ngh_finder)
    elif mode == 'i':  # inductive
        model.update_ngh_finder(partial_ngh_finder)
    else:
        raise ValueError('training mode {} not found.'.format(mode))
    device = model.n_feat_th.data.device
    num_instance = len(train_src_l)
    logger.info('num of training instances: {}'.format(num_instance))
    idx_list = np.arange(num_instance)

    if args.use_batch:
        diff = np.diff(train_ts_l)
        count = 0
        index = 0
        batch_s = 0
        batch_index = [0]
        for d in diff:
            count += d
            index += 1
            batch_s+=1
            if index == diff.shape[0]:
                batch_index.append(index + 2)
            elif count > PEIROD_WINDOWS or batch_s==bs:
            # elif count > PEIROD_WINDOWS:
                batch_index.append(index)
                count = 0
                batch_s=0
    else:
        diff = np.diff(train_ts_l)
        count = 0
        index = 0
        batch_index = [0]
        for d in diff:
            count += d
            index += 1
            if index == diff.shape[0]:
                batch_index.append(index + 2)
            elif count > PEIROD_WINDOWS:
                batch_index.append(index)
                count = 0


    num_batch = len(batch_index) - 1
    for epoch in range(epochs):
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        logger.info('start {} epoch'.format(epoch))
        total_sample_time=0
        for k in tqdm(range(num_batch)):
            pos_embed_dic = {}
            neg_embed_dic = {}
            batch_idx = idx_list[batch_index[k]:batch_index[k + 1]]
            src_l_cut, dst_l_cut = train_src_l[batch_idx], train_dst_l[batch_idx]
            ts_l_cut = train_ts_l[batch_idx]
            e_l_cut = train_e_idx_l[batch_idx]
            label_l_cut = train_label_l[batch_idx]  # currently useless since we are not predicting edge labels
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)

            # feed in the data and learn from error
            optimizer.zero_grad()
            model.train()
            model.linkpredictor.train()

            pos_src_embed,pos_tgt_embed, neg_src_emb,neg_tgt_embed,sample_time = model.contrast(args,src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut)   # the core training code
            total_sample_time+=sample_time
            pos_embed_single = pos_src_embed*pos_tgt_embed
            neg_embed_single = neg_src_emb*neg_tgt_embed
            if args.use_period:
                count = 0
                src_dst=np.hstack((src_l_cut, dst_l_cut)).reshape(-1, 2)
                tuple_src_dst = [tuple(row) for row in src_dst.tolist()]
                for i in tuple_src_dst:
                    if i not in pos_embed_dic:
                        pos_embed_dic[i] = pos_embed_single[count].view(1,-1)
                    else:
                        pos_embed_dic[i]=torch.cat((pos_embed_dic[i],pos_embed_single[count].view(1,-1)),0)
                    count+=1
                count = 0
                src_fake = np.hstack((src_l_cut, dst_l_fake)).reshape(-1, 2)
                tuple_src_fake = [tuple(row) for row in src_fake.tolist()]
                for i in tuple_src_fake:
                    if i not in neg_embed_dic:
                        neg_embed_dic[i] = neg_embed_single[count].view(1, -1)
                    else:
                        neg_embed_dic[i] = torch.cat((neg_embed_dic[i], neg_embed_single[count].view(1, -1)), 0)
                    count += 1
                for key,pos_embe in pos_embed_dic.items():
                    lated_embe = pos_embe[-1].view(-1,1)
                    peri_weight = torch.matmul(pos_embe[:-1,:],lated_embe)
                    if peri_weight.size(0)>1:
                        peri_weight = (peri_weight-peri_weight.min())/(peri_weight.max()-peri_weight.min())
                        values, indices = torch.sort(peri_weight, dim=0,descending=True)
                        indices = indices.view(-1)
                        if indices.size(0)>args.j:
                            indices = indices[:args.j]
                        peri_embed = torch.add(pos_embe[-1].view(1,-1),torch.sum(torch.mul(pos_embe[indices,:],peri_weight[indices,:]),dim=0).view(1,-1))
                    else:
                        peri_embed = pos_embe
                    pos_embed_dic[key]=peri_embed

                for key, neg_embe in neg_embed_dic.items():
                    lated_embe = neg_embe[-1].view(-1, 1)
                    peri_weight = torch.matmul(neg_embe[:-1, :], lated_embe)
                    if peri_weight.size(0) > 1:
                        peri_weight = (peri_weight - peri_weight.min()) / (peri_weight.max() - peri_weight.min())
                        values, indices = torch.sort(peri_weight, dim=0, descending=True)
                        indices = indices.view(-1)
                        if indices.size(0) > args.j:
                            indices = indices[:args.j]
                        peri_embed = torch.add(neg_embe[-1].view(1, -1),
                                               torch.sum(torch.mul(neg_embe[indices, :], peri_weight[indices,:]), dim=0).view(1,
                                                                                                                   -1))
                    else:
                        peri_embed = neg_embe
                    neg_embed_dic[key] = peri_embed

                pos_embe = torch.zeros((0,peri_embed.size(1))).to(device)
                neg_embe = torch.zeros((0, peri_embed.size(1))).to(device)
                for embe in pos_embed_dic.values():
                    pos_embe = torch.cat((pos_embe,embe),dim=0)
                for embe in neg_embed_dic.values():
                    neg_embe = torch.cat((neg_embe,embe),dim=0)
                pos_embe = torch.cat((pos_embed_single,pos_embe),dim=0)
                neg_embe = torch.cat((neg_embed_single, neg_embe), dim=0)
                pos_score = model.linkpredictor(pos_embe)
                neg_score = model.linkpredictor(neg_embe)
            else:
                pos_score = model.linkpredictor(pos_embed_single)
                neg_score = model.linkpredictor(neg_embed_single)


            pos_label = torch.ones(pos_score.size(0), dtype=torch.float, device=device, requires_grad=False)
            neg_label = torch.zeros(neg_score.size(0), dtype=torch.float, device=device, requires_grad=False)
            loss = criterion(pos_score.flatten(), pos_label) + criterion(neg_score.flatten(), neg_label)
            loss.backward()
            optimizer.step()
            
            # collect training results
            with torch.no_grad():
                model.eval()
                model.linkpredictor.eval()
                pos_prob = torch.sigmoid(pos_score.flatten())
                neg_prob = torch.sigmoid(neg_score.flatten())
                pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(pos_prob.size(0)), np.zeros(neg_prob.size(0))])
                accuracy = (pred_label == true_label).mean()
                acc.append(accuracy)
                aprecision = average_precision_score(true_label, pred_score)
                ap.append(aprecision)
                m_loss.append(loss.item())
                auc_score = roc_auc_score(true_label, pred_score)
                auc.append(auc_score)

        # validation phase use all information
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for {} nodes'.format(mode),args,model, val_rand_sampler, val_src_l,
                                                          val_dst_l, val_ts_l, val_label_l, val_e_idx_l)
        logger.info('epoch: {}:'.format(epoch))
        logger.info('epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('train acc: {}, val acc: {}'.format(np.mean(acc), val_acc))
        logger.info('train auc: {}, val auc: {}'.format(np.mean(auc), val_auc))
        logger.info('train ap: {}, val ap: {}'.format(np.mean(ap), val_ap))
        if epoch == 0:
            # save things for data anaysis
            checkpoint_dir = '/'.join(model.get_checkpoint_path(0).split('/')[:-1])
            model.ngh_finder.save_ngh_stats(checkpoint_dir)  # for data analysis
            model.save_common_node_percentages(checkpoint_dir)

        # early stop check and checkpoint saving
        if early_stopper.early_stop_check(val_ap):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_checkpoint_path = model.get_checkpoint_path(early_stopper.best_epoch)
            model.load_state_dict(torch.load(best_checkpoint_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
            print(model.get_checkpoint_path(epoch))
            torch.save(model.state_dict(), model.get_checkpoint_path(epoch))

    print("walk采样时间",total_sample_time)