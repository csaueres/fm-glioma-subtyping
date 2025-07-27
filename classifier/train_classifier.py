import argparse
import os

# internal imports
from clam_funcs.utils.file_utils import save_pkl, load_pkl
from clam_funcs.utils.utils import *
from model.model_utils_common import train,get_embed_dim
from clam_funcs.dataset_generic import Generic_MIL_Dataset

import pandas as pd
import numpy as np

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_val_auc = []
    all_val_acc = []
    all_val_f1 = []
    all_val_mcc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        #in this case the val dataset and test dataset are the same.
        train_dataset, val_dataset, test_dataset = dataset.return_splits(csv_path='{}/split_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        train_dataset.load_from_h5(True)
        val_dataset.load_from_h5(True)
        test_dataset.load_from_h5(True)
        results, val_acc, val_auc, val_f1, val_mcc  = train(datasets, i, args)
        all_val_f1.append(val_f1)
        all_val_auc.append(val_auc)
        all_val_mcc.append(val_mcc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'val_acc': all_val_acc, 
        'val_auc': all_val_auc, 'val_f1': all_val_f1, 'val_mcc' : all_val_mcc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--embedder', type=str, default='uni_v1')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--n_block', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--csv', type=str, default=None)
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--model_type', type=str, choices=['clam_sb','clam_mb','mamba','linear'], default='mamba', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--task', type=str)
# parser.add_argument('--use_h5', action="store_true",default=False)
parser.add_argument('--patch_frac',type=float,default=1.0,help='what fraction of random patches to use for each image for training')
parser.add_argument('--augments', nargs='+', help='which augmentations to use (for none use og)', required=True)
args = parser.parse_args()
args.return_attn=False
args.data_dir+=os.sep+args.embedder
args.embed_dim = get_embed_dim(args.embedder)

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'seed': args.seed,
            'n_block': args.n_block,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

print('\nLoad Dataset')


if args.task == 'idh_1p19q_class':
    args.n_class=3
    dataset = Generic_MIL_Dataset(csv_path = args.csv,
                            data_dir= args.data_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'gbm':0, 'astro':1, 'oligo':2},
                            patient_strat=False,
                            ignore=[],
                            patch_frac=args.patch_frac,
                            augments=args.augments) 
elif args.task == 'idh_class':
    args.n_class=2
    dataset = Generic_MIL_Dataset(csv_path = args.csv,
                            data_dir= args.data_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'gbm':0, 'astro':1, 'oligo':1},
                            patient_strat=False,
                            ignore=[])
elif args.task == '1p19q_class':
    args.n_class=2
    dataset = Generic_MIL_Dataset(csv_path = args.csv,
                            data_dir= args.data_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'gbm':99, 'astro':0, 'oligo':1},
                            patient_strat=False,
                            ignore=[99])
elif args.task == 'dataset_id':
    args.n_class=3
    dataset = Generic_MIL_Dataset(csv_path = args.csv,
                            data_dir= args.data_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'tcga':0, 'tum':1, 'ebrains':2},
                            patient_strat=False,
                            ignore=[],
                            patch_frac=args.patch_frac,
                            augments=args.augments) 
else:
    raise NotImplementedError
    
args.results_dir = os.path.join(args.results_dir, str(args.exp_code))
os.makedirs(args.results_dir,exist_ok=True)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()


print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


