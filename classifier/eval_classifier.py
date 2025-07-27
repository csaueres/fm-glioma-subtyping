import numpy as np
import torch
import torch.nn.functional as F

import os
import argparse
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, matthews_corrcoef, f1_score,balanced_accuracy_score
from sklearn.metrics import auc as calc_auc

from clam_funcs.utils.file_utils import save_pkl, load_pkl
from clam_funcs.utils.utils import *
from clam_funcs.dataset_generic import Generic_MIL_Dataset
from model.eval_utils import Accuracy_Logger
from model.model_utils_common import get_classifier, get_embed_dim


def get_results_for_fold(model, eval_data, f, args):
    
    ckpt_path = os.path.join(args.checkpoint_dir, "s_{}_checkpoint.pt".format(f))
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)
    #print_network(model)
    if(args.split_dir):
        #if a splits dir is being passed assume we want to evaluate on all folds
        #get correct validation fold
        _, eval_data, _ = eval_data.return_splits(csv_path=f'{args.split_dir}/split_{f}.csv')
    eval_data.load_from_h5(True)
    eval_loader = get_split_loader(eval_data,training=False)
    results_dict,acc_logger ,(val_error, val_auc, val_f1, val_mcc) = summary(model, eval_loader, args.n_class,args.case_level)
    #filename = f'output/slide_level_preds/{args.exp_code}/split_{f}_results.pkl'
    #save_pkl(filename, results_dict)

    return acc_logger, 1-val_error, val_auc, val_f1, val_mcc



def summary(model, loader, n_class,case_level):
    acc_logger = Accuracy_Logger(n_classes=n_class)
    model.eval()

    slide_ids = loader.dataset.slide_data['slide_id']
    case_ids = loader.dataset.slide_data['case_id']
    slide_results = {}
    case_results={}
    for batch_idx, (data, label) in enumerate(loader):
        if(label==-1):
            continue
        data = data.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        case_id = case_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, _ = model(data)
            logits=logits.cpu()
            probs = F.softmax(logits, dim=1).squeeze(0)
        probs = probs.cpu().numpy()
        if case_id in case_results:
            case_results[case_id][0].append(probs)
        else:
            case_results[case_id]=([probs],label.item())
        slide_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
    #Case level
    if(case_level):
        print("Case Level Preds")
        all_labels = np.array([v[1] for v in case_results.values()])
        all_probs = np.array([np.mean(v[0],axis=0) for v in case_results.values()])
    else:
        print("Slide Level Preds")
        all_labels = np.array([v['label'] for v in slide_results.values()])
        all_probs = np.array([v['prob'] for v in slide_results.values()])

    all_preds = np.argmax(all_probs,axis=-1)
    acc_logger.log_batch(all_preds,all_labels)
    test_error = 1. - (np.sum(all_preds==all_labels)/len(all_labels))

    if n_class == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_class)])
        for class_idx in range(n_class):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))
        ba=balanced_accuracy_score(all_labels,all_preds)
    mcc=matthews_corrcoef(all_labels,all_preds,sample_weight=None)


    return slide_results, acc_logger, (test_error, auc, ba, mcc)

parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--eval_data_csv', type=str, default=None, 
                    help='csv of items to be evaluated on')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='splits')
parser.add_argument('--embedder', type=str, default='uni_v1')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--checkpoint_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--case_level',default=False,action='store_true')
parser.add_argument('--model_type', type=str,default="TODO")
parser.add_argument('--augments', nargs='+', help='which augmentations to use (for none use og)', required=True)
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.data_dir+=os.sep+args.embedder
args.embed_dim = get_embed_dim(args.embedder)


if __name__ == "__main__":
    args.n_class=3
    args.n_block=2
    args.drop_out=0.0
    args.return_attn=False
    dataset = Generic_MIL_Dataset(csv_path = args.eval_data_csv,
                            data_dir= args.data_dir,
                            shuffle = False, 
                            seed = 7, 
                            print_info = True,
                            label_dict = {'gbm':0, 'astro':1, 'oligo':2},
                            patient_strat=False,
                            patient_voting='max',
                            ignore=[],
                            augments=args.augments,
                            )
    model, _ = get_classifier(args)
    fold_results=[]
    for f in range(args.k):
        acc_logger, acc, auc, f1, mcc = get_results_for_fold(model,dataset,f,args)
        class_accs=[]
        for i in range(args.n_class):
            c_acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, c_acc, correct, count))
            class_accs.append(c_acc)
        fold_results.append((f,acc,*class_accs,auc,f1,mcc))
    results_df = pd.DataFrame(fold_results,columns=["Fold","Accuracy","Class 0 Acc.", "Class 1 Acc.", "Class 2 Acc.", "AUC","BA","MCC"])
    averages = results_df.mean()
    stds = results_df.std(ddof=0)
    results_df=pd.concat([results_df,averages.to_frame().T,stds.to_frame().T], ignore_index=True)
    results_df['Fold']=pd.Series(["1","2","3","4","5","Avg","StdDev"])
    print(f"\n{args.exp_code}")
    print(results_df.to_string())




