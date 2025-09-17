from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_norm(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute symmetric normalized edge weights for LightGCN.

    L_hat = D^{-1/2} A D^{-1/2}, where A is adjacency (including both directions).
    Returns edge_weight aligned with edge_index columns.
    """
    row, col = edge_index
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    return deg_inv_sqrt[row] * deg_inv_sqrt[col]


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        edge_index: torch.Tensor,
        embed_dim: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.n_layers = n_layers
        self.edge_index = edge_index
        self.register_buffer("edge_weight", compute_norm(edge_index, self.num_nodes))

        self.emb = nn.Embedding(self.num_nodes, embed_dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self, x: torch.Tensor) -> torch.Tensor:
        row, col = self.edge_index
        # message passing: x_{k+1} = L_hat @ x_k using sparse-like scatter
        out = torch.zeros_like(x)
        msg = x[col] * self.edge_weight.unsqueeze(-1)
        out.index_add_(0, row, msg)
        return out

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # layer-wise propagation + average as LightGCN
        x0 = self.emb.weight
        out = x0
        xk = x0
        for _ in range(self.n_layers):
            xk = self.propagate(xk)
            out = out + xk
        out = out / (self.n_layers + 1)

        user_emb = out[: self.num_users]
        item_emb = out[self.num_users :]
        return user_emb, item_emb

    @staticmethod
    def bpr_loss(u: torch.Tensor, i: torch.Tensor, j: torch.Tensor, l2_reg: float = 1e-4) -> torch.Tensor:
        # Bayesian Personalized Ranking loss
        x_ui = (u * i).sum(dim=1)
        x_uj = (u * j).sum(dim=1)
        loss = -F.logsigmoid(x_ui - x_uj).mean()
        # L2 regularization on embeddings
        loss = loss + l2_reg * (u.norm(2).pow(2) + i.norm(2).pow(2) + j.norm(2).pow(2)) / u.size(0)
        return loss

