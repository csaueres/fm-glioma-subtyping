import numpy as np
import torch
import os

from model.modules.MambaMIL import MambaMIL
from model.modules.LinearNet import LinearModel
from custom_losses import SoftMCCLoss, SoftMCCLossMulti, WeightedCombinedLosses

from clam_funcs.utils.utils import *
from model.eval_utils import *
from model.train_utils_clam import init_clam_model_train, clam_bag_loss

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


#sets up the model and a loss function. Ignore the loss function at eval time. For eval the checkpoint needs to be loaded after!!
def get_classifier(args):
    if(args.n_class==2):
        losses = [torch.nn.BCEWithLogitsLoss(),
                    SoftMCCLoss(),]
    else:
        losses = [torch.nn.CrossEntropyLoss(reduction='mean'),
                    SoftMCCLossMulti(),]
    weights = [1.0, 1.0]
    loss_fn = WeightedCombinedLosses(losses, weights)
    if(args.model_type=='mamba'):
        model = MambaMIL(in_dim = args.embed_dim, n_classes=args.n_class, dropout=args.drop_out, act='gelu', n_layer = args.n_block,return_attn=args.return_attn)
    elif(args.model_type=='linear'):
        model=LinearModel(args.embed_dim,args.n_class,args.drop_out)
    elif(args.model_type in ['clam_sb','clam_mb']):
        model = init_clam_model_train(args,device)
        if clam_bag_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            loss_fn = SmoothTop1SVM(n_classes = args.n_class)
            loss_fn = loss_fn.to(device)
    else:
        raise NotImplementedError("No such model type :", args.model_type)
    return model, loss_fn

def get_embed_dim(embedder):
    if(embedder=='uni_v1'):
        embed_dim=1024
    elif(embedder=='gigapath'):
        embed_dim=1536
    elif(embedder=='virchow'):
        embed_dim=2560
    elif(embedder=='musk'):
        embed_dim=1024
    elif(embedder=='retccl'):
        embed_dim=2048
    elif(embedder=='h0-mini'):
        embed_dim=1536
    else:
        raise NotImplementedError("No such embedder :", embedder)
    return embed_dim

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))


    print('\nInit Model...', end=' ')
    model, loss_fn = get_classifier(args)
    
    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split)
    test_loader = get_split_loader(test_split)
    print('Done!')

    early_stopping = EarlyStopping(patience = 10, stop_epoch=15, verbose = True)

    if(args.model_type in ['mamba','linear']):
        from model.train_utils_mamba import train_loop, validate
    elif(args.model_type in ['clam_sb','clam_mb']):
        from model.train_utils_clam import train_loop, validate

    for epoch in range(args.max_epochs):    
        train_loop(epoch, model, train_loader, optimizer, args.n_class, writer, loss_fn)
        stop = validate(cur, epoch, model, val_loader, args.n_class, 
            early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    # else:
    #     torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _,_ ,(val_error, val_auc, val_f1,val_mcc) = summary(model, val_loader, args.n_class)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, acc_logger, (test_error, test_auc, _, _) = summary(model, test_loader, args.n_class)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_class):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))


    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.close()
    return results_dict, 1-val_error, val_auc, val_f1, val_mcc
