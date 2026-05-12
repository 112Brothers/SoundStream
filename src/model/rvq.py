import torch
import torch.nn.functional as F
from torch import nn

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes, d, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.decay = decay
        self.eps = eps
        embed = torch.randn(num_codes, d)
        self.register_buffer("codebook", embed)
        self.register_buffer("cluster_size", torch.ones(num_codes))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("_initialized", torch.zeros(1, dtype=torch.bool))
        self._init_buf: list = []  # accum until n >= num_codes

    @torch.no_grad()
    def _kmeans_init(self, data):
        """k-means (10 iters) on accum frames for codebook."""
        K = self.num_codes
        data = data.float()
        n = data.shape[0]
        idx = torch.randperm(n, device=data.device)[:K]
        centroids = data[idx].clone()
        for _ in range(10):
            dists = (
                data.pow(2).sum(1, keepdim=True)
                - 2 * data @ centroids.t()
                + centroids.pow(2).sum(1)
            )
            assign = dists.argmin(1)
            for k in range(K):
                mask = assign == k
                if mask.any():
                    centroids[k] = data[mask].mean(0)
        self.codebook.copy_(centroids)
        self.embed_avg.copy_(centroids)
        self.cluster_size.fill_(1.0)
        self._initialized.fill_(True)

    def forward(self, x):
        # x: [B,C,T] -> flat: [BT,C]
        B, C, T = x.shape
        flat = x.permute(0, 2, 1).reshape(-1, C)
        if self.training and not self._initialized.item():
            self._init_buf.append(flat.detach().float().cpu())
            accumulated = torch.cat(self._init_buf, dim=0)
            if accumulated.shape[0] >= self.num_codes * 4:
                self._kmeans_init(accumulated.to(flat.device))
                self._init_buf = []
        # cast to codebook dtype (flat can be fp16 under AMP)
        flat = flat.to(self.codebook.dtype)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )
        ind = dist.argmin(1)  # shape [BT]
        one_hot = F.one_hot(ind, self.num_codes).float()  # [BT, num_codes]
        if self.training:
            with torch.no_grad():
                self.cluster_size.mul_(self.decay).add_(one_hot.sum(0), alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(one_hot.t() @ flat, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                smoothed = (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
                self.codebook.copy_(self.embed_avg / smoothed.unsqueeze(1))
        # Perplexity
        avg_probs = one_hot.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        quant = F.embedding(ind, self.codebook).view(B, T, C).permute(0, 2, 1)
        commit_loss = F.mse_loss(x, quant.detach())
        quant = x + (quant - x).detach()  # STE
        return quant, ind, commit_loss, perplexity

class RVQ(nn.Module):
    def __init__(self, num_quantizers, num_codes, d):
        super().__init__()
        self.quantizers = nn.ModuleList(
            [VectorQuantizer(num_codes=num_codes, d=d) for _ in range(num_quantizers)]
        )

    def forward(self, x):
        yhat = 0
        res = x
        inds = []
        commit_loss = 0.0
        perplexities = []
        for vq in self.quantizers:
            q_x, ind, vq_commit, perp = vq(res)
            res = res - q_x
            yhat = yhat + q_x
            inds.append(ind)
            commit_loss = commit_loss + vq_commit
            perplexities.append(perp)
        return yhat, inds, commit_loss, perplexities
