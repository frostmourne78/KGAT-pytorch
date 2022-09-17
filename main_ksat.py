import heapq
import os
import sys
import random
from time import time

import pandas as pd
from prettytable import PrettyTable
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model.KSAT import KSAT
from test_parser.parser_ksat import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_ksat import load_data

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_dict(train_entity_pairs, n_items, start, end, train_user_set):
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs, train_user_set)).to(device)
    return feed_dict


def evaluate(model, n_items, test_batch_size, user_dict, Ks, device):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}

    batch_result_mid=list()
    test_batch_size_e = test_batch_size
    model.eval()
    train_user_dict = user_dict['train_user_set']
    test_user_dict = user_dict['test_user_set']

    u_batch_size = test_batch_size_e
    i_batch_size = test_batch_size_e

    test_users = list(test_user_dict.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    entity_gcn_emb, user_gcn_emb = model.generate()

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        batch_result_mid = list()
        user_list_batch = test_users[start: end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = user_gcn_emb[user_batch]

        if True:
            # batch-item test
            n_item_batchs = n_items // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_items))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end - i_start).to(device)
                i_g_embddings = entity_gcn_emb[item_batch]

                i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == n_items

        user_batch_rating_uid = list(zip(rate_batch, user_list_batch))

        for i_user in range(0, len(user_batch_rating_uid)):
            mid = user_batch_rating_uid[i_user]
            rating = mid[0]
            u = mid[1]

            try:
                training_items = train_user_dict[u]
            except Exception:
                training_items = []
                # user u's items in the test set
            user_pos_test = test_user_dict[u]

            all_items = set(range(0, n_items))
            test_items = list(all_items - set(training_items))
            if True:
                r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)

            batch_result = get_performance(user_pos_test, r, auc, Ks,batch_result_mid)

        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users

    assert count == n_test_users
    return result


def get_performance(user_pos_test, r, auc, Ks,bath_result):
    precision, recall, ndcg, hit_ratio = [], [], [], []
    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))
    bath_result.append({'recall': np.array(recall), 'precision': np.array(precision), 'ndcg': np.array(ndcg),
                        'hit_ratio': np.array(hit_ratio), 'auc': auc})
    return bath_result


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc


def train(args):
    global device
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    """cf data"""  # 此数据是用户和项目的交互信息 如用户1的项目的交互历史 每一行都是一个用户，她/他与项目的积极互动：（userID和a list of itemID
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))
    # construct model & optimizer
    """define model"""
    model = KSAT(args, n_users, n_entities, n_relations, mean_mat_list[0], graph)

    model.to(device)
    logging.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1
    best_recall = 0

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}

    # train model
    print("start training ...")
    for epoch in range(1, args.epoch):
        # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]

        """training"""
        loss, s = 0, 0
        train_s_t = time()
        while s + args.cf_batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_pairs, n_items,
                                  s, s + args.cf_batch_size,
                                  user_dict['train_user_set'])
            batch_loss = model(batch, mode='train_cf')
            batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.cf_batch_size

            if (epoch % args.cf_print_every) == 0:
                logging.info(
                    'Training: Epoch {:04d}| Iter Loss {:.4f}'.format(epoch, batch_loss.item()))
        logging.info('Training: Epoch {:04d} | Iter Mean Loss {:.4f}'.format(epoch, batch_loss / args.cf_batch_size))

        train_e_t = time()

        if (epoch % args.evaluate_every) == 0 or epoch == args.epoch - 1:
            ret = evaluate(model, n_items, args.test_batch_size, user_dict, Ks, device)
            print(ret)
            epoch_list.append(epoch)

    # save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg']:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    logging.info(
        'Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
            int(best_metrics['epoch_idx']), best_metrics['precision@{}'.format(k_min)],
            best_metrics['precision@{}'.format(k_max)], best_metrics['recall@{}'.format(k_min)],
            best_metrics['recall@{}'.format(k_max)], best_metrics['ndcg@{}'.format(k_min)],
            best_metrics['ndcg@{}'.format(k_max)]))


def predict(args):
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    _, user_dict = load_data(args)

    # load model
    model = KSAT(args, n_users, n_entities, n_relations)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # predict
    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    cf_scores, metrics_dict = evaluate(model, args.test_batch_size, user_dict, Ks, device)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    print('CF Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'],
        metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))


if __name__ == '__main__':
    args = parse_ksat_args()
    train(args)
    # predict(args)
