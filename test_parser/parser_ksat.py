import argparse


def parse_ksat_args():
    parser = argparse.ArgumentParser(description="Run KSAT.")


    #dataset
    parser.add_argument("--dataset",nargs='?',default="last-fm",help="Choose a dataset:[last-fm,amazon-book,yelp2018]")
    parser.add_argument('--data_dir', nargs='?', default='datasets/',help='Input data path.')

    #use_pretrain
    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth',
                        help='Path of stored model.')

    #train
    parser.add_argument('--seed', type=int, default=2022,help='Random seed.')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--cf_batch_size', type=int, default=1024,help='CF batch size.')
    parser.add_argument('--kg_batch_size', type=int, default=1024,help='KG batch size.')
    parser.add_argument('--test_batch_size', type=int, default=1024,help='Test batch size (the user number to test every batch).')
    parser.add_argument('--embed_dim', type=int, default=64 ,help='User / entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=64,help='Relation Embedding size.')
    parser.add_argument('--n_factors', type=int, default=4,help='number of latent factor for user favour')
    parser.add_argument("--ind", type=str, default='distance', help="Independence modeling: mi, distance, cosine")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',help='Calculate metric@K when evaluating.')
    parser.add_argument('--evaluate_every', type=int, default=10,help='Epoch interval of evaluating CF.')
    parser.add_argument('--cf_print_every', type=int, default=1,help='Iter interval of printing CF loss.')
    parser.add_argument('--kg_print_every', type=int, default=1,help='Iter interval of printing KG loss.')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--sim_regularity', type=float, default=1e-4, help='regularization weight for latent factor')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5,help='Lambda when calculating CF l2 loss.')
    parser.add_argument('--stopping_steps', type=int, default=10,help='Number of epoch for early stopping')
    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')

    args = parser.parse_args()


    # ===== save model ===== #
    save_dir = 'trained_model/KSAT/'
    args.save_dir = save_dir

    return args


