import numpy as np
import torch
import torch.nn.functional as F
import os

from clam_funcs.utils.utils import *
from model.eval_utils import Accuracy_Logger, custom_auc

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_loop(epoch, model, loader, optimizer, n_class, writer = None, loss_fn = None):   
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_class)
    train_loss = 0.
    train_error = 0.
    agg_logits = []
    agg_labels=[]
    minibatch_size = 16
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        #print("Input shape", data.shape)
        logits, _ = model(data)
        # print(logits.shape)
        #print(batch_idx)
        #print(logits)

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        # print(Y_hat)
        # print(label)
        
        acc_logger.log(Y_hat, label)
        error = calculate_error(Y_hat, label)
        train_error += error
        #loss = loss_fn(logits, label)
        #loss_value = loss.item()
        #train_loss += loss_value
        agg_logits.append(logits)
        agg_labels.append(label)
        if (batch_idx + 1) % minibatch_size == 0:
            #MCC Loss:
            minibatch_logits = torch.cat(agg_logits,dim=0)
            minibatch_labels = torch.nn.functional.one_hot(torch.cat(agg_labels,dim=0),n_class).float()
            # print(minibatch_labels.shape)
            # print(minibatch_logits.shape)
            loss = loss_fn(minibatch_logits, minibatch_labels)
            #print("Loss", loss.item())
            train_loss += loss.item()
        
            # backward pass
            loss.backward()
            # step
            optimizer.step()
            optimizer.zero_grad()
            agg_logits = []
            agg_labels=[]

    train_error /= len(loader)
    train_loss /= len(loader)/minibatch_size

    print('\nEpoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_class):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


   
def validate(cur, epoch, model, loader, n_class, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_class)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, _ = model(data)

            Y_hat = torch.topk(logits, 1, dim=1)[1]

            acc_logger.log(Y_hat, label)
            
            #loss = loss_fn(logits, label)

            all_logits.append(logits.cpu().numpy().squeeze())
            all_labels.append(label.item())
            
            #val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
    all_logits = np.asarray(all_logits); all_labels = np.asarray(all_labels)
    val_loss = loss_fn(torch.FloatTensor(all_logits),torch.nn.functional.one_hot(torch.LongTensor(all_labels),n_class).float()).item()
    val_error /= len(loader)

    #all_probs = F.softmax(all_logits,dim=1)
    all_probs = np.exp(all_logits)/np.sum(np.exp(all_logits),axis=1,keepdims=True)
    auc = custom_auc(all_labels,all_probs,n_class)
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_class):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

