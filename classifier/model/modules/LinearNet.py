import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self, in_dim, n_classes, dropout):
        super().__init__()

        self.fc = nn.Linear(in_dim,n_classes)

        
    def forward(self, x):
        logits = self.majority_vote(x)
        return logits, {}
    
    def majority_vote(self,x):
        patch_logits = self.fc(x)
        avg_logits = torch.mean(patch_logits,dim=0,keepdims=True)
        return avg_logits

    def class_average_patch(self,x):
        avg_x = torch.mean(x,dim=0,keepdims=True)
        out = self.fc(avg_x)
        return out
    