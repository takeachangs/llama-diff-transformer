import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__
        self.f1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype = cfg["dtype"], bias = False),
        self.f2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype = cfg["dtype"], bias = False),
        self.f3 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype = cfg["dtype"], bias = False),
        self.silu = SiLU()
    
    def forward(self, x):
        x_f1 = self.f1(x)
        x_f2 = self.f2(x)
        x = self.silu(x_f1) * x_f2 # SwiGLU

        return self.f3(x)
class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)