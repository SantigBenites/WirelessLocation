from typing import List, Tuple
import torch


def compute_pca_from_cov(global_sum: torch.Tensor, global_XtX: torch.Tensor, N: int,
                         device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    D = global_sum.shape[0]
    mu = (global_sum / N).to(device=device, dtype=torch.float64)
    cov = (global_XtX / N) - torch.outer(mu, mu)
    cov = (cov + cov.T) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(cov)
    idx = torch.arange(D - 1, -1, -1, device=device)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs, mu


def select_components(eigvals: torch.Tensor, variance_threshold: float = 0.95, n_components: int = 0) -> int:
    total = eigvals.sum()
    if n_components and n_components > 0:
        return min(n_components, eigvals.numel())
    ratios = eigvals / (total + 1e-12)
    cum = torch.cumsum(ratios, dim=0)
    if eigvals.numel() == 0:
        return 0
    k = int((cum >= variance_threshold).nonzero(as_tuple=False)[0].item() + 1)
    return k


def feature_importance_from_pca(eigvals: torch.Tensor, eigvecs: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return torch.zeros(eigvecs.shape[0], dtype=torch.float64, device=eigvecs.device)
    V_k = eigvecs[:, :k]
    lam_k = eigvals[:k]
    imp = (V_k ** 2) @ lam_k
    imp = imp / (imp.sum() + 1e-12)
    return imp


def tolist(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().tolist()]