
import numpy as np
import torch
import torch.nn.functional as F
import os

from clam_funcs.utils.utils import *
from model.modules.model_clam import CLAM_SB, CLAM_MB
from model.eval_utils import Accuracy_Logger,custom_auc
B = 8
inst_loss = 'svm' #svm, ce
clam_bag_loss = 'ce' #svm
clam_bag_weight=0.7

def init_clam_model_train(args,dev):
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_class, 
                  "embed_dim": args.embed_dim,
                  "size_arg": 'small', #'big'/'small'
                  "subtyping":True,
                  "gate":True
                  }
    
    if B > 0:
        model_dict.update({'k_sample': B})
    
    if inst_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        instance_loss_fn = SmoothTop1SVM(n_classes = 2)
        if dev.type == 'cuda':
            instance_loss_fn = instance_loss_fn.cuda()
    else:
        instance_loss_fn = torch.nn.CrossEntropyLoss()
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
    elif args.model_type == 'clam_mb':
        model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
    else:
        raise NotImplementedError
    
    return model

#Might not need this, since we load checkpoint and set to eval later manually

# def init_clam_model_eval(args, ckpt_path, device='cuda'):
#     print('Init Model')    
#     model_dict = {"dropout": args.drop_out, 'n_classes': args.n_class, "embed_dim": args.embed_dim}
    
#     if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
#         model_dict.update({"size_arg": args.model_size})
    
#     if args.model_type =='clam_sb':
#         model = CLAM_SB(**model_dict)
#     elif args.model_type =='clam_mb':
#         model = CLAM_MB(**model_dict)

#     print_network(model)

#     ckpt = torch.load(ckpt_path)
#     ckpt_clean = {}
#     for key in ckpt.keys():
#         if 'instance_loss_fn' in key:
#             continue
#         ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
#     model.load_state_dict(ckpt_clean, strict=True)

#     _ = model.to(device)
#     _ = model.eval()
#     return model


def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, extra_info = model(data, label=label, instance_eval=True)
        instance_dict = extra_info['instance_dict']

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        #Y_prob = F.softmax(logits, dim = 1)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = clam_bag_weight * loss + (1-clam_bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        #TODO: check if the values being generated here actually make sense
        # if (batch_idx + 1) % 20 == 0:
        #     print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
        #         'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)


def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, extra_info = model(data, label=label, instance_eval=True)
            instance_dict = extra_info['instance_dict']
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            Y_prob = F.softmax(logits, dim = 1)
            
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    auc = custom_auc(labels,prob,n_classes)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False