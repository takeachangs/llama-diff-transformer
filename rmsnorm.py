import torch 
import torch.nn as nn 

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x):
        
        x_norm = x * torch.rsqrt(
                x.pow(2).mean(dim = -1, keepdim = True) + 
                self.eps
                )
        return (x_norm * self.weight).to(dtype=x.dtype)
    