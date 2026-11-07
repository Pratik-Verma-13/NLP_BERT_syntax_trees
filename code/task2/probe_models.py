import torch
import torch.nn as nn

class StructuralDistanceProbe(nn.Module):
    def __init__(self, bert_dim=768, rank=64):
        super().__init__()
        # Linear projection matrix B
        self.B = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(rank, bert_dim)))

    def forward(self, h):
        """
        h: Tensor of shape (seq_len, 768)
        Returns: Tensor of predicted squared distances of shape (seq_len, seq_len)
        """
        proj = torch.matmul(h, self.B.t())  # (seq_len, rank)
        diff = proj.unsqueeze(1) - proj.unsqueeze(0)  # (seq_len, seq_len, rank)
        return torch.sum(diff ** 2, dim=-1)           # (seq_len, seq_len)

class StructuralDepthProbe(nn.Module):
    def __init__(self, bert_dim=768, rank=64):
        super().__init__()
        self.B = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(rank, bert_dim)))

    def forward(self, h):
        """
        h: Tensor of shape (seq_len, 768)
        Returns: Tensor of predicted squared depths of shape (seq_len,)
        """
        proj = torch.matmul(h, self.B.t())  # (seq_len, rank)
        return torch.sum(proj ** 2, dim=-1)           # (seq_len,)