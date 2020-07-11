import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def position_encoding(emb):
    ans = []
    dim = len(emb[0])
    for idx in range(1, len(emb)+1):
        tmp = [idx / (10000**((i//2)*2/dim)) for i in range(1, dim+1)]
        tmp[0::2] = np.sin(tmp[0::2])
        tmp[1::2] = np.cos(tmp[1::2])
        ans.append(tmp)
    return torch.LongTensor(ans)

def get_mask(seq): # TODO
    batch_size, seq_len = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

class ScaledDotProduct(nn.Module):
    def __init__(self):
        super(ScaledDotProduct, self).__init__()

    def forward(self, q, k, v, mask=None):
        dk = k.size()[0]
        k = k.transpose(2,3)
        w = torch.div(torch.matmul(q, k), dk**0.5)
        if (mask is not None):
            w = w.masked_fill(mask==0, -1e9)
        w = F.softmax(w, dim= -1)
        return torch.matmul(w, v)

class MultiHead(nn.Module):
    def __init__(self, model_dim):
        super(MultiHead, self).__init__()
        self.h = 8
        self.attn = ScaledDotProduct()
        self.w_qs = nn.Linear(model_dim, model_dim, bias=False)
        self.w_ks = nn.Linear(model_dim, model_dim, bias=False)
        self.w_vs = nn.Linear(model_dim, model_dim, bias=False)
        self.model_dim = model_dim

        self.fc = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, q, k, v, mask=None):
        result = []
        batch_size, len_q, len_k, len_v = q.size()[0], q.size()[1], k.size()[1], v.size()[1]

        q = self.w_qs(q).view(batch_size, len_q, self.h, -1).transpose(1, 2)
        k = self.w_ks(k).view(batch_size, len_k, self.h, -1).transpose(1, 2)
        v = self.w_vs(v).view(batch_size, len_v, self.h, -1).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        heads = self.attn(q, k, v, mask)
        heads = heads.transpose(1, 2).contiguous()
        heads = heads.view(batch_size, -1, self.model_dim)
        return self.fc(heads)


class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 512)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Encoder(nn.Module):
    def __init__(self, N, dim, pad_idx, vocab_size, device):
        super(Encoder, self).__init__()
        self.N = N
        self.dim = dim
        self.pad_idx = pad_idx
        self.device = device
        self.mhs = nn.ModuleList([MultiHead(self.dim) for i in range(self.N)])
        self.ffns = nn.ModuleList([FFN() for i in range(self.N)])
        self.ans = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for i in range(self.N*2)])
        self.emb = nn.Embedding(vocab_size, dim, pad_idx)

    def forward(self, x):
        batch_size = x.size()[0]
        seq_mask = get_pad_mask(x, self.pad_idx)
        x = self.emb(x)
        positions = position_encoding(x.view(-1, self.dim))
        positions = positions.view(batch_size, -1, self.dim).to(self.device)
        x = F.dropout(x + positions, p=0.1)
        for i in range(self.N):
            x = x+F.dropout(self.mhs[i](x, x, x, mask=seq_mask), p=0.1)
            x = self.ans[2*i](x)
            residuals = x
            x = F.dropout(self.ffns[i](x), p=0.1)
            x += residuals
            x = self.ans[2*i+1](x)
        return x, seq_mask

class Decoder(nn.Module):
    def __init__(self, N, dim, pad_idx, vocab_size, device):
        super(Decoder, self).__init__()
        self.N = N
        self.dim = dim # dimension of model
        self.device = device
        self.pad_idx = pad_idx
        self.ans = nn.ModuleList([nn.LayerNorm(dim, eps=1e-06) for i in range(self.N*3)])
        self.emb = nn.Embedding(vocab_size, dim, pad_idx)
        self.mhs = nn.ModuleList([MultiHead(self.dim) for i in range(self.N*2)])
        self.ffns = nn.ModuleList([FFN() for i in range(self.N)])
        self.fc = nn.Linear(dim, vocab_size)

    def forward(self, x, enc_output, seq_mask):
        batch_size = x.size()[0]
        masks = get_mask(x) & get_pad_mask(x, self.pad_idx)
        x = self.emb(x)
        positions = position_encoding(x.view(-1, self.dim))
        positions = positions.view(batch_size, -1, self.dim).to(self.device)
        x = F.dropout(x + positions, p=0.1)

        for i in range(self.N):
            x = x+ F.dropout(self.mhs[2*i](x, x, x, mask=masks), p=0.1)
            x = self.ans[3*i](x)
            x = x+F.dropout(self.mhs[2*i+1](x, enc_output, enc_output, mask=seq_mask), p=0.1)
            x = self.ans[3*i+1](x)
            x = x + F.dropout(self.ffns[i](x), p=0.1)
            x = self.ans[3*i+2](x)
        x = self.fc(x)*(self.dim**-0.5)
        return x
