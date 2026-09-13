"""PieceTransformer Q18+V and matched MLP baseline for the 54-label puzzle."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    """Pre-norm transformer block using PyTorch scaled-dot-product attention."""
    def __init__(self,width,heads):
        super().__init__()
        self.heads=heads
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width);self.out=nn.Linear(width,width)
        self.ff=nn.Sequential(nn.Linear(width,4*width),nn.SiLU(),nn.Linear(4*width,width))

    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        x=x+self.out(F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,t,d))
        return x+self.ff(self.norm2(x))


class PieceTransformer(nn.Module):
    """26 physical-piece tokens plus CLS; folded slot embeddings and shared Q/V heads."""
    def __init__(self,pieces,width=256,layers=4,heads=8):
        super().__init__()
        indices=torch.tensor([p+[0]*(3-len(p)) for p in pieces])
        mask=torch.tensor([[1]*len(p)+[0]*(3-len(p)) for p in pieces])
        self.register_buffer('indices',indices);self.register_buffer('mask',mask)
        self.table=nn.Parameter(torch.randn(3,54,width)*width**-.5)
        self.position=nn.Parameter(torch.randn(26,width)*.02)
        self.cls=nn.Parameter(torch.randn(1,1,width)*.02)
        self.input_norm=nn.LayerNorm(width)
        self.blocks=nn.Sequential(*(Block(width,heads) for _ in range(layers)))
        self.norm=nn.LayerNorm(width)
        self.q=nn.Linear(width,18);self.v=nn.Linear(width,1)

    def forward(self,state):
        b=state.shape[0]
        values=state[:,self.indices].long()
        tokens=self.position.unsqueeze(0).expand(b,-1,-1)
        for slot in range(3):
            encoded=F.one_hot(values[:,:,slot],54).to(self.table.dtype)
            tokens=tokens+encoded@self.table[slot]*self.mask[:,slot,None]
        hidden=torch.cat((self.cls.expand(b,-1,-1),tokens),dim=1)
        pooled=self.norm(self.blocks(self.input_norm(hidden))[:,0])
        return self.q(pooled),self.v(pooled).squeeze(-1)


class ResidualMLP(nn.Module):
    """One-hot 54-position MLP baseline with the same Q18 and scalar V objective."""
    def __init__(self,width=512):
        super().__init__()
        self.input=nn.Linear(54*54,width)
        self.blocks=nn.ModuleList(nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width)) for _ in range(4))
        self.norm=nn.LayerNorm(width);self.q=nn.Linear(width,18);self.v=nn.Linear(width,1)

    def forward(self,state):
        x=F.silu(self.input(F.one_hot(state.long(),54).flatten(1).float()))
        for block in self.blocks:x=x+block(x)
        x=self.norm(x)
        return self.q(x),self.v(x).squeeze(-1)


def make_model(geometry,config):
    """Input geometry/config; output the selected Q/V scorer without hidden defaults."""
    if config.get('model','transformer')=='mlp': return ResidualMLP(config.get('width',512))
    return PieceTransformer(geometry['pieces'],config.get('width',256),config.get('layers',4),8)
