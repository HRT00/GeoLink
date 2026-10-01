import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple, Optional
from torch import Tensor

import numpy as np

import math

# class InfoNCE(nn.Module):

#     def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
#         super().__init__()
        
#         self.loss_function = loss_function
#         self.device = device

#     def forward(self, image_features1, image_features2, logit_scale):
#         image_features1 = F.normalize(image_features1, dim=-1)
#         image_features2 = F.normalize(image_features2, dim=-1)
        
#         logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
#         logits_per_image2 = logits_per_image1.T
        
#         labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
#         loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

#         return los

EPS = 1e-8

def get_w_regularization(w1, w2):
    """
    预算约束下的竞争正则化 (BCCR)
    w1, w2: [B, K, 1] - Router 输出的权重
    target_k: 预期每张图中包含的“目标物体”数量（例如设定为 3 或 5）
    """
    # 合并两个视角的权重进行统一处理
    w = torch.cat([w1, w2], dim=0) # [2B, K, 1]
    
    # --- 3. 排除中间态 (Binarization/Polarization) ---
    # 强制 W 趋向 0 或 1。w(1-w) 在 0.5 时最大，在 0 和 1 时为 0。
    # 这能显著提升“剔除”效果，让噪声 Slot 彻底归零。
    loss_polar = torch.mean(w * (1 - w))

    # --- 4. 空间多样性 (Spatial Diversity) ---
    # 防止所有 Slot 都去抢同一个物体（导致其他 Slot 变为空/噪声）
    # 鼓励 w 在 K 个 Slot 之间具有差异性
    loss_div = - torch.var(w, dim=1).mean()

    return 0.1 * loss_polar + 0.1 * loss_div

def batched_cosine_sim(A: torch.Tensor, B: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    # A, B: (B, K, D)
    A_n = F.normalize(A, p=2, dim=-1, eps=eps)
    B_n = F.normalize(B, p=2, dim=-1, eps=eps)
    return torch.einsum('bik,bjk->bij', A_n, B_n)  # (B, K, K)


def attention_overlap(A_att: torch.Tensor, B_att: torch.Tensor) -> torch.Tensor:
    # A_att, B_att: (B, H*W, K) attention maps of slots over spatial tokens
    # returns affinity by computing JSD or dot-overlap between slot attention vectors
    # Here we use simple dot product of normalized attention vectors
    # Output: (B, K, K) where entry (i,j) is overlap between slot_i in A and slot_j in B
    # We assume A_att and B_att are L2-normalized along spatial dim
    A_n = F.normalize(A_att, p=2, dim=1, eps=EPS)  # (B,HW,K)
    B_n = F.normalize(B_att, p=2, dim=1, eps=EPS)
    # compute KxK overlap per batch: einsum over spatial dim
    # result shape: (B, K, K)
    return torch.einsum('bsk,btk->bst', A_n, B_n)


# -------------------------
# Graph construction
# -------------------------
def build_slot_graph(slots: torch.Tensor,
                     mode: str = 'cosine',
                     att_maps: Optional[torch.Tensor] = None,
                     sigma: float = 0.1) -> torch.Tensor:
    """
    Construct a symmetric affinity matrix G for each batch.
    slots: (B, K, D)
    mode: 'cosine' | 'rbf' | 'attention'
      - 'cosine': pairwise cosine similarity
      - 'rbf': RBF (gaussian) kernel on euclidean distance
      - 'attention': uses att_maps (B, HW, K) for cross-slot overlap
    returns G: (B, K, K) symmetric, non-negative
    """
    B, K, D = slots.shape
    if mode == 'cosine':
        G = batched_cosine_sim(slots, slots)  # [-1,1]
        # shift to non-negative: (x+1)/2
        G = (G + 1.0) / 2.0
        G = torch.clamp(G, min=0.0)
    elif mode == 'rbf':
        # compute pairwise squared distances
        # slots: (B,K,D)
        x_norm2 = (slots ** 2).sum(dim=-1, keepdim=True)  # (B, K, 1)
        # pairwise dist squared: (B, K, K)
        dist2 = x_norm2 + x_norm2.transpose(1, 2) - 2.0 * torch.einsum('bik,bjk->bij', slots, slots)
        G = torch.exp(-dist2 / (2.0 * sigma * sigma))
    elif mode == 'attention':
        if att_maps is None:
            raise ValueError("att_maps required for mode='attention'")
        # att_maps: (B, HW, K) -> compute overlap
        # convert to (B, K, HW) for convenience
        A = att_maps.permute(0, 2, 1)
        # normalize
        A_n = F.normalize(A, p=2, dim=-1, eps=EPS)
        G = torch.einsum('bip,bjp->bij', A_n, A_n)
        G = torch.clamp(G, min=0.0)
    else:
        raise ValueError(f"unknown mode {mode}")
    # optionally set diagonal to zero or keep as self-affinity (we'll keep diag as 1)
    # set diagonal to 1 for stable Laplacian
    eye = torch.eye(K, device=G.device, dtype=G.dtype).unsqueeze(0)
    G = G * (1.0 - eye) + eye  # ensure diagonal ones
    return G


# -------------------------
# Laplacian & spectral embedding
# -------------------------
def compute_graph_laplacian(G: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    """
    G: (B, K, K) affinity (non-negative, symmetric)
    returns L: (B, K, K) Laplacian (symmetric)
    """
    deg = G.sum(dim=-1)  # (B, K)
    B, K = deg.shape
    Dinv_sqrt = torch.diag_embed(torch.pow(deg + EPS, -0.5))  # (B, K, K)
    if normalized:
        # normalized Laplacian: L = I - D^{-1/2} G D^{-1/2}
        I = torch.eye(K, device=G.device, dtype=G.dtype).unsqueeze(0)
        DGD = torch.einsum('bij,bjk,bkl->bil', Dinv_sqrt, G, Dinv_sqrt)  # (B,K,K)
        L = I - DGD
    else:
        D = torch.diag_embed(deg)
        L = D - G
    # ensure symmetry
    L = 0.5 * (L + L.transpose(1, 2))
    return L


def spectral_embedding(L: torch.Tensor, r: int = 4) -> torch.Tensor:
    """
    Compute r smallest non-trivial eigenvectors of Laplacian per batch.
    L: (B, K, K) symmetric positive semidef.
    Returns U: (B, K, r) spectral embedding.
    Note: smallest eigenvector is often constant (trivial), so we skip the first eigenvector (index 0).
    We'll take eigenvectors 1..r (if available), else take first r.
    """
    B, K, _ = L.shape
    L = L.float()
    # use torch.linalg.eigh (returns ascending eigenvalues)
    # To save memory, compute batchwise; torch.linalg.eigh supports batch
    _, eigvecs = torch.linalg.eigh(L)  # eigvals: (B,K) ascending
    # skip first trivial eigenvector if K > r
    # choose indices start_idx..start_idx+r-1
    if K > r:
        start = 1  # skip eigenvector 0
        end = start + r
        if end <= K:
            U = eigvecs[:, :, start:end]  # (B, K, r)
        else:
            # fallback: take last r eigenvectors if not enough after skipping
            U = eigvecs[:, :, -r:]
    else:
        # K <= r, return all eigenvectors (may include trivial)
        U = eigvecs  # (B, K, K)
        if U.shape[-1] != r:
            # pad with zeros
            pad = r - U.shape[-1]
            U = F.pad(U, (0, pad), 'constant', 0.0)
    return U.to(L.dtype)  # (B, K, r)


# -------------------------
# Spectral alignment loss (orthogonal Procrustes)
# -------------------------
def orthogonal_procrustes_loss(Ua: torch.Tensor, Ub: torch.Tensor) -> torch.Tensor:
    """
    Ua, Ub: (B, K, r) spectral embeddings
    minimizes min_Q || Ua - Ub Q ||_F^2 over orthogonal Q.
    Closed-form: Q* = V U^T where SVD of (Ub^T Ua) = U S V^T.
    Loss = ||Ua||^2 + ||Ub||^2 - 2 * trace(S)
    We'll compute per-batch and average.
    """
    B, K, r = Ua.shape
    # center columns? spectral embeddings may be sign/scale ambiguous; we can optionally normalize columns
    # compute BtA = Ub^T Ua -> shape (B, r, r)
    BtA = torch.einsum('bkr,bkp->brp', Ub, Ua)  # (B, r, r)
    BtA = BtA.float()
    # SVD per batch
    # torch.linalg.svd supports batch and returns U, S, Vh (Vh is V^H)
    U_svd, S_svd, Vh_svd = torch.linalg.svd(BtA)  # U: (B,r,r), S: (B,r), Vh: (B,r,r)
    # trace(S) = sum S
    traceS = S_svd.sum(dim=1)  # (B,)
    # norms
    Ua_norm2 = (Ua ** 2).sum(dim=(1, 2))
    Ub_norm2 = (Ub ** 2).sum(dim=(1, 2))
    loss_per_batch = Ua_norm2 + Ub_norm2 - 2.0 * traceS
    loss = loss_per_batch.mean()
    return loss


# -------------------------
# SPOT loss integrator
# -------------------------
def reg_InfoNCE(Sa: torch.Tensor,
              Sb: torch.Tensor,
              att_maps_a: Optional[torch.Tensor] = None,
              att_maps_b: Optional[torch.Tensor] = None,
              graph_mode: str = 'cosine',
              r: int = 4,
              lambda_struct: float = 1.0) -> dict:
    """
    Compute SPOT losses for a batch:
     - spectral structural alignment loss L_struct
    Sa, Sb: (B, K, D) slot embeddings per view
    att_maps_* optional: (B, HW, K) for 'attention' graph_mode
    """
    # 1) build graphs
    Ga = build_slot_graph(Sa, mode=graph_mode, att_maps=att_maps_a)
    Gb = build_slot_graph(Sb, mode=graph_mode, att_maps=att_maps_b)

    # 2) Laplacian
    La = compute_graph_laplacian(Ga, normalized=True)
    Lb = compute_graph_laplacian(Gb, normalized=True)

    # 3) spectral embedding
    Ua = spectral_embedding(La, r=r)  # (B, K, r)
    Ub = spectral_embedding(Lb, r=r)

    # 4) spectral alignment loss
    L_struct = orthogonal_procrustes_loss(Ua, Ub)

    loss = lambda_struct * L_struct

    return loss  

def slot_infoNCE(slots1, slots2, temperature=0.07):
    """
    slots1: B × K × D
    slots2: B × K × D
    Returns scalar loss
    """

    # 1) Normalize along feature dimension
    slots1 = F.normalize(slots1, dim=-1)
    slots2 = F.normalize(slots2, dim=-1)

    B, K, D = slots1.shape

    # 2) Flatten slots: B*K × D
    slots1_flat = slots1.reshape(B*K, D)
    slots2_flat = slots2.reshape(B*K, D)

    # 3) Compute similarity matrix: (B*K) × (B*K)
    logits = torch.matmul(slots1_flat, slots2_flat.t()) / temperature

    # 4) Positive pairs are on the diagonal
    labels = torch.arange(B*K, device=logits.device)

    # 5) Cross-entropy as InfoNCE
    loss = F.cross_entropy(logits, labels)

    return loss

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        logits_per_image1 = logit_scale * image_features1 @ image_features1.T
        
        logits_per_image1_self = logit_scale * image_features1 @ image_features1.T
        logits_per_image2_self = logit_scale * image_features2 @ image_features2.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss_ssl = (
            self.loss_function(logits_per_image1_self, labels) + 
            self.loss_function(logits_per_image2_self, labels)
        ) / 2
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2 + loss_ssl

        # loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss


class SymmetricInfoNCE(nn.Module):
    """Correct symmetric cross-view InfoNCE.

    ``InfoNCE`` above is retained as a legacy reproduction path because older
    GeoCore experiments were trained with it.  In that implementation the
    forward cross-view logits are overwritten by self-view logits.  New runs
    can opt into this explicit, auditable implementation instead.
    """

    def __init__(self, loss_function, device=None):
        super().__init__()
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        logits = logit_scale.clamp(max=100.0) * (
            image_features1 @ image_features2.t()
        )
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (
            self.loss_function(logits, labels)
            + self.loss_function(logits.t(), labels)
        )


class UncertaintyAwareSoftInfoNCE(SymmetricInfoNCE):
    """UA-Geo-style NIG uncertainty and Top-K soft-negative labels.

    Epochs 1..``head_freeze_epochs`` use symmetric InfoNCE and do not train
    the uncertainty head.  The following ``head_warmup_epochs`` fit the NIG
    distribution while retrieval remains symmetric.  Soft labels are enabled
    only after those two stages, so random NIG predictions cannot perturb the
    pretrained representation at startup.
    """

    def __init__(
        self,
        loss_function,
        label_smoothing: float = 0.1,
        topk: int = 5,
        soft_weight: float = 0.05,
        nig_regularization: float = 0.01,
        nig_loss_weight: float = 0.5,
        head_freeze_epochs: int = 4,
        head_warmup_epochs: int = 3,
        device=None,
    ) -> None:
        super().__init__(loss_function=loss_function, device=device)
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if topk < 1 or soft_weight < 0.0:
            raise ValueError("topk must be positive and soft_weight non-negative")
        if nig_regularization < 0.0 or nig_loss_weight < 0.0:
            raise ValueError("NIG loss weights must be non-negative")
        if head_freeze_epochs < 0 or head_warmup_epochs < 0:
            raise ValueError("uncertainty warm-up epochs must be non-negative")
        self.label_smoothing = float(label_smoothing)
        self.topk = int(topk)
        self.soft_weight = float(soft_weight)
        self.nig_regularization = float(nig_regularization)
        self.nig_loss_weight = float(nig_loss_weight)
        self.head_freeze_epochs = int(head_freeze_epochs)
        self.head_warmup_epochs = int(head_warmup_epochs)
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    @staticmethod
    def _split_nig(parameters: Tensor):
        if parameters.ndim != 2 or parameters.size(1) != 4:
            raise ValueError("NIG parameters must have shape [B, 4]")
        return parameters.unbind(dim=1)

    def _nig_loss(self, observations: Tensor, parameters: Tensor) -> Tensor:
        gamma, nu, alpha, beta = self._split_nig(parameters.float())
        gamma = gamma[:, None]
        nu = nu[:, None]
        alpha = alpha[:, None]
        beta = beta[:, None]
        observations = observations.float()
        omega = 2.0 * beta * (1.0 + nu)
        residual2 = (observations - gamma).square()
        nll = (
            0.5 * torch.log(torch.pi / nu)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log(residual2 * nu + omega)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        regularization = residual2 * (
            2.0 * nu + alpha + beta.reciprocal()
        )
        return (nll + self.nig_regularization * regularization).mean()

    @staticmethod
    def _raw_uncertainty(parameters: Tensor) -> Tensor:
        _, nu, alpha, beta = UncertaintyAwareSoftInfoNCE._split_nig(
            parameters.float()
        )
        return beta * (nu + 1.0) / (
            nu * (alpha - 1.0).clamp_min(1e-6)
        )

    @staticmethod
    def _uncertainty(parameters: Tensor) -> Tensor:
        uncertainty = UncertaintyAwareSoftInfoNCE._raw_uncertainty(parameters)
        # The paper assumes u < 1 when proving the positive target remains
        # valid. The head is initialized near u=0.5 and gets a three-epoch NIG
        # warm-up before this value enters labels; this final clamp is only a
        # probability-safety guard and its raw value is logged separately.
        return uncertainty.clamp(0.0, 1.0 - 1e-6)

    def _direction_loss(
        self,
        logits: Tensor,
        raw_similarity: Tensor,
        candidate_indices: Tensor,
        uncertainty: Tensor,
    ) -> Tensor:
        batch_size = logits.size(0)
        base = self.label_smoothing / max(batch_size - 1, 1)
        # Construct probability targets in FP32 even under AMP. Apart from
        # avoiding severe rounding across a full row of small probabilities,
        # this also prevents indexed assignment from mixing an autocast FP16
        # destination with the FP32 residual positive mass.
        targets = torch.full(
            (batch_size, batch_size),
            base,
            device=logits.device,
            dtype=torch.float32,
        )
        diagonal = torch.arange(batch_size, device=logits.device)
        targets[diagonal, diagonal] = 0.0
        selected_similarity = raw_similarity.detach().gather(
            1, candidate_indices
        ).float()
        selected_targets = (
            base
            + selected_similarity.clamp_min(0.0)
            * uncertainty.detach().float().unsqueeze(1)
            * self.soft_weight
        )
        targets.scatter_(1, candidate_indices, selected_targets)
        targets[diagonal, diagonal] = 1.0 - targets.sum(dim=1)
        if bool((targets < 0).any()):
            raise FloatingPointError(
                "soft-negative targets exhausted the positive probability mass"
            )
        return -(
            targets * F.log_softmax(logits.float(), dim=1)
        ).sum(dim=1).mean()

    def forward_outputs(self, outputs: Dict[str, Tensor], logit_scale: Tensor):
        satellite = F.normalize(outputs["satellite_descriptor"], dim=-1)
        query = F.normalize(outputs["query_descriptor"], dim=-1)
        if satellite.size(0) < 2:
            raise ValueError("uncertainty-aware InfoNCE needs batch size >= 2")
        if "satellite_nig" not in outputs:
            raise KeyError("uncertainty-aware loss requires model NIG outputs")

        # Evaluation is query -> satellite.  Keep both directions symmetric
        # during training, matching the repository's retrieval convention.
        raw_q2s = query @ satellite.t()
        diagonal_mask = torch.eye(
            satellite.size(0), device=satellite.device, dtype=torch.bool
        )
        hard_k = min(self.topk, satellite.size(0) - 1)
        q2s_indices = raw_q2s.detach().masked_fill(
            diagonal_mask, float("-inf")
        ).topk(hard_k, dim=1).indices
        # UA-Geo defines t_ij from same-modal positive/negative reference CLS
        # features. GeoCore's semantic branch is the projected DINO CLS token;
        # the full descriptor additionally contains the regional readout.
        if "satellite_nig_feature" not in outputs:
            raise KeyError("uncertainty-aware loss requires reference CLS features")
        satellite_cls = F.normalize(
            outputs["satellite_nig_feature"], dim=-1
        )
        satellite_same = satellite_cls @ satellite_cls.t()
        # t_ij is the DER observation/target. Stop its feature path so the NIG
        # objective learns to predict neighborhood complexity instead of
        # changing the target similarities to fit its own distribution.
        satellite_observations = satellite_same.gather(
            1, q2s_indices
        ).detach()
        nig_loss = self._nig_loss(
            satellite_observations, outputs["satellite_nig"]
        )

        nig_active = self.current_epoch > self.head_freeze_epochs
        soft_start = self.head_freeze_epochs + self.head_warmup_epochs
        soft_active = self.current_epoch > soft_start
        scaled_q2s = logit_scale.clamp(max=100.0) * raw_q2s
        scaled_s2q = scaled_q2s.t()
        labels = torch.arange(satellite.size(0), device=satellite.device)
        if soft_active:
            retrieval = 0.5 * (
                self._direction_loss(
                    scaled_q2s,
                    raw_q2s,
                    q2s_indices,
                    self._uncertainty(outputs["satellite_nig"]),
                )
                + self.loss_function(scaled_s2q, labels)
            )
        else:
            retrieval = 0.5 * (
                self.loss_function(scaled_q2s, labels)
                + self.loss_function(scaled_s2q, labels)
            )
        combined = retrieval + (
            self.nig_loss_weight * nig_loss if nig_active else 0.0
        )
        uncertainty = self._uncertainty(outputs["satellite_nig"]).mean()
        raw_uncertainty = self._raw_uncertainty(
            outputs["satellite_nig"]
        ).mean()
        return combined, {
            "retrieval_contrastive": retrieval.detach(),
            "nig": nig_loss.detach(),
            "uncertainty": uncertainty.detach(),
            "raw_uncertainty": raw_uncertainty.detach(),
            "nig_active": retrieval.new_tensor(float(nig_active)),
            "soft_active": retrieval.new_tensor(float(soft_active)),
        }


# -----------------------------------------------------------------------------#
# GeoCore Loss                                                                 #
# -----------------------------------------------------------------------------#

def bidirectional_contrastive_loss(a: Tensor,
                                   b: Tensor,
                                   logit_scale: Tensor) -> Tensor:
    """Default bidirectional retrieval loss used when no external InfoNCE is supplied."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = logit_scale.clamp(max=100.0) * (a @ b.t())
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.t(), labels))


def branch_decorrelation_loss(semantic: Tensor, regional: Tensor) -> Tensor:
    """Weak cross-covariance regularizer from method draft Eq. (16)."""
    semantic = semantic - semantic.mean(dim=0, keepdim=True)
    regional = regional - regional.mean(dim=0, keepdim=True)
    cross_covariance = semantic.t() @ regional / max(semantic.size(0), 1)
    return cross_covariance.square().sum()


def compute_geocore_loss(outputs: Dict[str, Tensor],
                         logit_scale: Tensor,
                         lambda_semantic: float = 0.5,
                         lambda_region: float = 1.0,
                         lambda_decorrelation: float = 0.01,
                         contrastive_loss=None,
                         lambda_self_ssl: float = 1.0) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Original GeoCore retrieval, SSL, semantic, regional and decorrelation losses."""
    if contrastive_loss is None:
        contrastive_loss = bidirectional_contrastive_loss

    extra_components = {}
    if hasattr(contrastive_loss, "forward_outputs"):
        retrieval, extra_components = contrastive_loss.forward_outputs(
            outputs, logit_scale
        )
    else:
        retrieval = contrastive_loss(outputs["satellite_descriptor"],
                                     outputs["query_descriptor"],
                                     logit_scale)

    if lambda_self_ssl < 0.0:
        raise ValueError("lambda_self_ssl must be non-negative")
    if lambda_self_ssl > 0.0:
        retrieval_ssl_q = contrastive_loss(
            outputs["query_descriptor"],
            outputs["query_descriptor"],
            logit_scale,
        )
        retrieval_ssl_g = contrastive_loss(
            outputs["satellite_descriptor"],
            outputs["satellite_descriptor"],
            logit_scale,
        )
        ssl = retrieval_ssl_g + retrieval_ssl_q
    else:
        # Sample4Geo and UA-Geo optimize cross-view retrieval directly. Skip
        # constructing the same-view repulsion graph when it is disabled.
        ssl = retrieval.new_zeros(())

    if lambda_semantic > 0.0:
        semantic = contrastive_loss(
            outputs["satellite_semantic"],
            outputs["text_semantic"],
            logit_scale,
        )
        semantic += contrastive_loss(
            outputs["query_semantic"],
            outputs["text_semantic"],
            logit_scale,
        )
    else:
        semantic = retrieval.new_zeros(())

    if lambda_region > 0.0:
        regional = F.binary_cross_entropy_with_logits(
            outputs["satellite_weight_logits"], outputs["satellite_teacher"]
        )
        regional += F.binary_cross_entropy_with_logits(
            outputs["query_weight_logits"], outputs["query_teacher"]
        )
    else:
        regional = retrieval.new_zeros(())

    if lambda_decorrelation > 0.0:
        decorrelation = branch_decorrelation_loss(
            outputs["satellite_semantic"], outputs["satellite_regional"]
        )
        decorrelation += branch_decorrelation_loss(
            outputs["query_semantic"], outputs["query_regional"]
        )
    else:
        decorrelation = retrieval.new_zeros(())

    total = (4.0 * (retrieval + lambda_self_ssl * ssl) +
             lambda_semantic * semantic +
             lambda_region * regional +
             lambda_decorrelation * decorrelation)

    components = {
        "total": total.detach(),
        "retrieval": retrieval.detach(),
        "self_ssl": ssl.detach(),
        "semantic": semantic.detach(),
        "regional": regional.detach(),
        "decorrelation": decorrelation.detach(),
        **extra_components,
    }
    return total, components


def _pairwise_balanced_region_score(source: Tensor,
                                    target: Tensor,
                                    temperature: float,
                                    iterations: int) -> Tensor:
    """Return balanced soft region-matching scores for all image pairs."""
    source = F.normalize(source.detach().float(), dim=-1)
    target = F.normalize(target.detach().float(), dim=-1)
    similarity = torch.einsum("bkd,jld->bjkl", source, target)
    logits = similarity / max(float(temperature), 1e-6)
    logits = logits - logits.amax(dim=(-2, -1), keepdim=True)
    transport = logits.exp().clamp_min(torch.finfo(logits.dtype).tiny)
    region_count = source.size(1)
    target_mass = 1.0 / region_count
    for _ in range(max(int(iterations), 1)):
        transport = transport / transport.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        transport = transport * target_mass
        transport = transport / transport.sum(dim=-2, keepdim=True).clamp_min(1e-12)
        transport = transport * target_mass
    return (transport * similarity).sum(dim=(-2, -1))


def late_interaction_hard_negative_loss(
    outputs: Dict[str, Tensor],
    sinkhorn_temperature: float = 0.07,
    sinkhorn_iterations: int = 5,
    topk: int = 3,
    margin: float = 0.05,
    smoothing_temperature: float = 0.05,
    descriptor_kind: str = "distilled",
) -> Tensor:
    """Separate regional-confusable negatives in the deployed descriptor.

    Balanced late interaction is used only to select hard negatives.  The
    optimization target remains the fixed K-body cosine score, so inference
    still uses one 1024-D descriptor and no score fusion.
    """
    if topk < 1:
        raise ValueError("topk must be positive")
    if margin < 0 or smoothing_temperature <= 0:
        raise ValueError("margin must be non-negative and temperature positive")

    if descriptor_kind == "base":
        satellite_key = "satellite_descriptor"
        query_key = "query_descriptor"
    elif descriptor_kind == "distilled":
        satellite_key = "satellite_distilled_descriptor"
        query_key = "query_distilled_descriptor"
    else:
        raise ValueError("descriptor_kind must be 'base' or 'distilled'")

    satellite = F.normalize(outputs[satellite_key].float(), dim=-1)
    query = F.normalize(outputs[query_key].float(), dim=-1)
    descriptor_scores = satellite @ query.t()
    with torch.no_grad():
        regional_scores = _pairwise_balanced_region_score(
            outputs["satellite_regions"], outputs["query_regions"],
            temperature=sinkhorn_temperature,
            iterations=sinkhorn_iterations,
        )
        batch_size = regional_scores.size(0)
        diagonal_mask = torch.eye(
            batch_size, device=regional_scores.device, dtype=torch.bool
        )
        hard_k = min(int(topk), max(batch_size - 1, 1))
        forward_indices = regional_scores.masked_fill(
            diagonal_mask, float("-inf")
        ).topk(hard_k, dim=1).indices
        backward_indices = regional_scores.t().masked_fill(
            diagonal_mask, float("-inf")
        ).topk(hard_k, dim=1).indices

    positive = descriptor_scores.diagonal()
    forward_negative = descriptor_scores.gather(1, forward_indices)
    backward_negative = descriptor_scores.t().gather(1, backward_indices)

    def smooth_margin(negative: Tensor) -> Tensor:
        violation = negative - positive[:, None] + float(margin)
        return (
            F.softplus(violation / float(smoothing_temperature))
            * float(smoothing_temperature)
        ).mean()

    return 0.5 * (
        smooth_margin(forward_negative) + smooth_margin(backward_negative)
    )


def queued_late_interaction_hard_negative_loss(
    satellite_descriptor: Tensor,
    query_descriptor: Tensor,
    satellite_regions: Tensor,
    query_regions: Tensor,
    location_ids: Tensor,
    queued_satellite_descriptor: Tensor,
    queued_query_descriptor: Tensor,
    queued_satellite_regions: Tensor,
    queued_query_regions: Tensor,
    queued_location_ids: Tensor,
    sinkhorn_temperature: float = 0.07,
    sinkhorn_iterations: int = 5,
    topk: int = 3,
    margin: float = 0.05,
    smoothing_temperature: float = 0.05,
) -> Tensor:
    """Mine region-confusable negatives from a detached cross-batch queue."""
    if topk < 1:
        raise ValueError("topk must be positive")
    satellite_descriptor = F.normalize(satellite_descriptor.float(), dim=-1)
    query_descriptor = F.normalize(query_descriptor.float(), dim=-1)
    queued_satellite_descriptor = F.normalize(
        queued_satellite_descriptor.detach().float(), dim=-1
    )
    queued_query_descriptor = F.normalize(
        queued_query_descriptor.detach().float(), dim=-1
    )
    positive = (satellite_descriptor * query_descriptor).sum(dim=-1)

    with torch.no_grad():
        satellite_to_query = _pairwise_balanced_region_score(
            satellite_regions, queued_query_regions,
            temperature=sinkhorn_temperature,
            iterations=sinkhorn_iterations,
        )
        query_to_satellite = _pairwise_balanced_region_score(
            query_regions, queued_satellite_regions,
            temperature=sinkhorn_temperature,
            iterations=sinkhorn_iterations,
        )
        same_location = location_ids[:, None].eq(queued_location_ids[None])
        satellite_to_query.masked_fill_(same_location, float("-inf"))
        query_to_satellite.masked_fill_(same_location, float("-inf"))
        valid_per_row = (~same_location).sum(dim=1)
        if int(valid_per_row.min()) < 1:
            return positive.new_zeros(())
        hard_k = min(int(topk), int(valid_per_row.min()))
        query_indices = satellite_to_query.topk(hard_k, dim=1).indices
        satellite_indices = query_to_satellite.topk(hard_k, dim=1).indices

    negative_query = torch.einsum(
        "bd,bkd->bk",
        satellite_descriptor,
        queued_query_descriptor[query_indices],
    )
    negative_satellite = torch.einsum(
        "bd,bkd->bk",
        query_descriptor,
        queued_satellite_descriptor[satellite_indices],
    )

    def smooth_margin(negative: Tensor) -> Tensor:
        violation = negative - positive[:, None] + float(margin)
        return (
            F.softplus(violation / float(smoothing_temperature))
            * float(smoothing_temperature)
        ).mean()

    return 0.5 * (
        smooth_margin(negative_query) + smooth_margin(negative_satellite)
    )


class WeightedInfoNCE(nn.Module):
    def __init__(self, loss_function, label_smoothing, k=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.loss_function = loss_function
        self.label_smoothing = label_smoothing
        self.device = device
        self.k = k

    def loss(self, similarity_matrix, eps_all):
        n = similarity_matrix.shape[0]
        total_loss = 0.0
        for i in range(n):
            eps = eps_all[i]
            total_loss += (1 - eps) * (-1. * similarity_matrix[i, i] + torch.logsumexp(similarity_matrix[i, :], dim=0))
            total_loss += eps * (-1. / n * similarity_matrix[i, :].sum() + torch.logsumexp(similarity_matrix[i, :], dim=0))
        total_loss /= n
        return total_loss

    def forward(self, image_features1, image_features2, logit_scale, positive_weights=None):
        # Normalize the image features
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        # Compute similarity logits
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        # Apply positive weights if provided
        if positive_weights is not None:
            eps = 1. - 1. / (1 + torch.exp(-self.k * positive_weights))
        else:
            eps = [self.label_smoothing for _ in range(image_features1.shape[0])]
        
        logits_per_image2 = logits_per_image1.T
        
        # Generate labels
        # labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

        loss1 = self.loss(logits_per_image1, eps)
        loss2 = self.loss(logits_per_image2, eps)
        # # Compute loss
        logits_per_image1_self = logit_scale * image_features1 @ image_features1.T
        logits_per_image2_self = logit_scale * image_features2 @ image_features2.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss_ssl = (
            self.loss_function(logits_per_image1_self, labels) + 
            self.loss_function(logits_per_image2_self, labels)
        ) / 2
        loss = (loss1 + loss2) / 2 + loss_ssl

        return loss

def relational_distillation_3d_loss(student_features, teacher_features, metric='euclidean', temperature=1.0, normalize=True):
    """
    Compute Relational Distillation loss between student and teacher features.
    
    Args:
        student_features: Tensor of shape (batch_size, dim_s)
        teacher_features: Tensor of shape (batch_size, dim_t)
        metric: 'euclidean' or 'cosine' to measure relational structure.
        temperature: scaling factor for distances or similarities (if applicable).
        normalize: whether to normalize distances by teacher's mean distance.
        
    Returns:
        loss: scalar Tensor representing the relational distillation loss.
    """
    assert student_features.dim() == 2 and teacher_features.dim() == 2, "Features should be 2D tensors"
    N = student_features.size(0)
    
    if metric == 'euclidean':
        # Normalize features to unit vectors
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)
        # Pairwise differences for teacher and student (shape: N x N x D)
        teacher_diffs = teacher_features.unsqueeze(1) - teacher_features.unsqueeze(0)
        student_diffs = student_features.unsqueeze(1) - student_features.unsqueeze(0)
        # Pairwise Euclidean distances (shape: N x N)
        teacher_dists = torch.norm(teacher_diffs, dim=2, p=2)
        student_dists = torch.norm(student_diffs, dim=2, p=2)
        
        if normalize:
            # Normalize by mean teacher distance to focus on relative structure
            eps = 1e-8
            teacher_rel = teacher_dists / teacher_dists.mean() + eps
            student_rel = student_dists / student_dists.mean() + eps
        else:
            teacher_rel = teacher_dists
            student_rel = student_dists
        
        # Apply temperature scaling if needed
        if temperature != 1.0:
            teacher_rel = teacher_rel / temperature
            student_rel = student_rel / temperature
        
        # Compute MSE loss between relational structures
        loss = F.mse_loss(student_rel, teacher_rel)
    
    elif metric == 'cosine':
        # Normalize features to unit vectors
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)
        # Compute pairwise cosine similarity matrices (shape: N x N)
        teacher_sim = torch.matmul(teacher_norm, teacher_norm.t())
        student_sim = torch.matmul(student_norm, student_norm.t())
        
        # Apply temperature scaling to similarities (logits) if needed
        if temperature != 1.0:
            teacher_sim = teacher_sim / temperature
            student_sim = student_sim / temperature
        
        # Compute MSE loss between relational structures
        loss = F.mse_loss(student_sim, teacher_sim)
    
    else:
        raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")
    
    return loss

import torch
import torch.nn.functional as F

# ---------- MMD Loss with kernels ----------
def pairwise_distances(x, y):
    """Compute pairwise squared Euclidean distances."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.clamp(dist, min=0.0)

def gaussian_rbf(dist_sq, sigma=1.0):
    return torch.exp(-dist_sq / (2 * sigma**2))

def mmd_loss(x, y, kernel="rbf", sigma=1.0, degree=2):
    """
    Maximum Mean Discrepancy with multiple kernels.
    kernel: 'linear' | 'poly' | 'rbf'
    """
    if kernel == "linear":
        Kxx = torch.mm(x, x.t())
        Kyy = torch.mm(y, y.t())
        Kxy = torch.mm(x, y.t())
    elif kernel == "poly":
        Kxx = (torch.mm(x, x.t()) + 1) ** degree
        Kyy = (torch.mm(y, y.t()) + 1) ** degree
        Kxy = (torch.mm(x, y.t()) + 1) ** degree
    elif kernel == "rbf":
        dist_xx = pairwise_distances(x, x)
        dist_yy = pairwise_distances(y, y)
        dist_xy = pairwise_distances(x, y)
        Kxx = gaussian_rbf(dist_xx, sigma)
        Kyy = gaussian_rbf(dist_yy, sigma)
        Kxy = gaussian_rbf(dist_xy, sigma)
    else:
        raise ValueError("Unsupported kernel: choose from 'linear', 'poly', 'rbf'")
    
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


# ---------- Graph Loss ----------
def graph_loss(s_features, t_features):
    s_sim = torch.matmul(F.normalize(s_features, dim=1),
                         F.normalize(s_features, dim=1).t())
    t_sim = torch.matmul(F.normalize(t_features, dim=1),
                         F.normalize(t_features, dim=1).t())
    s_deg = torch.diag(s_sim.sum(1))
    t_deg = torch.diag(t_sim.sum(1))
    s_lap = s_deg - s_sim
    t_lap = t_deg - t_sim
    return F.mse_loss(s_lap, t_lap)


# ---------- Ranking Loss ----------
def ranking_loss(s_features, t_features, margin=0.1):
    s_sim = torch.matmul(F.normalize(s_features, dim=1),
                         F.normalize(s_features, dim=1).t())
    t_sim = torch.matmul(F.normalize(t_features, dim=1),
                         F.normalize(t_features, dim=1).t())
    loss = 0.0
    N = s_sim.size(0)
    for i in range(N):
        pos_idx = torch.argmax(t_sim[i])  # teacher 认为最近邻
        for j in range(N):
            if j == i or j == pos_idx: 
                continue
            loss += F.relu(margin + s_sim[i, j] - s_sim[i, pos_idx])
    return loss / N


def mg_relational_distillation_3d_loss(student_features, teacher_features,
                                    metric='euclidean',
                                    temperature=1.0,
                                    normalize=True,
                                    use_mmd=True,
                                    use_graph=True,
                                    use_rank=True,
                                    mmd_kernel='rbf',
                                    mmd_sigma=1.0,
                                    mmd_degree=2,
                                    weights=(1.0, 0.5, 0.5, 0.5)):
    """
    Multi-relational distillation loss (extended from original).
    
    Args:
        metric: 'euclidean' | 'cosine'
        use_mmd, use_graph, use_rank: 是否引入 MMD/Graph/Ranking
        mmd_kernel: 'linear' | 'poly' | 'rbf'
        weights: (w_rel, w_mmd, w_graph, w_rank)
    """
    assert student_features.dim() == 2 and teacher_features.dim() == 2
    w_rel, w_mmd, w_graph, w_rank = weights

    # --- 原始 relational distillation ---
    if metric == 'euclidean':
        teacher_diffs = teacher_features.unsqueeze(1) - teacher_features.unsqueeze(0)
        student_diffs = student_features.unsqueeze(1) - student_features.unsqueeze(0)
        teacher_dists = torch.norm(teacher_diffs, dim=2, p=2)
        student_dists = torch.norm(student_diffs, dim=2, p=2)
        if normalize:
            eps = 1e-8
            teacher_rel = teacher_dists / (teacher_dists.mean() + eps)
            student_rel = student_dists / (student_dists.mean() + eps)
        else:
            teacher_rel, student_rel = teacher_dists, student_dists
        if temperature != 1.0:
            teacher_rel = teacher_rel / temperature
            student_rel = student_rel / temperature
        rel_loss = F.mse_loss(student_rel, teacher_rel)

    elif metric == 'cosine':
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)
        teacher_sim = torch.matmul(teacher_norm, teacher_norm.t())
        student_sim = torch.matmul(student_norm, student_norm.t())
        if temperature != 1.0:
            teacher_sim = teacher_sim / temperature
            student_sim = student_sim / temperature
        rel_loss = F.mse_loss(student_sim, teacher_sim)
    else:
        raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")

    # --- 额外蒸馏项 ---
    total_loss = w_rel * rel_loss
    if use_mmd:
        total_loss += w_mmd * mmd_loss(student_features, teacher_features,
                                       kernel=mmd_kernel, sigma=mmd_sigma, degree=mmd_degree)
    if use_graph:
        total_loss += w_graph * graph_loss(student_features, teacher_features)
    if use_rank:
        total_loss += w_rank * ranking_loss(student_features, teacher_features)

    return total_loss


def pairwise_dist_sq(x, y=None):
    if y is None:
        y = x
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).t()
    d2 = x2 + y2 - 2.0 * (x @ y.t())
    return torch.clamp(d2, min=0.0)

def rbf_kernel_matrix(x, y, sigma):
    d2 = pairwise_dist_sq(x, y)
    return torch.exp(-d2 / (2.0 * (sigma ** 2 + 1e-12)))

# ---------------- Multi-kernel MMD ----------------
def multi_kernel_mmd(x, y, sigma_list=(0.5, 1.0, 2.0, 4.0)):
    Kxx = 0.0
    Kyy = 0.0
    Kxy = 0.0
    for s in sigma_list:
        Kxx = Kxx + rbf_kernel_matrix(x, x, s)
        Kyy = Kyy + rbf_kernel_matrix(y, y, s)
        Kxy = Kxy + rbf_kernel_matrix(x, y, s)
    # average kernels
    Kxx = Kxx / len(sigma_list)
    Kyy = Kyy / len(sigma_list)
    Kxy = Kxy / len(sigma_list)
    return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()


def mg_relational_distillation_3d_loss_(student_features, teacher_features,
                                    metric='euclidean',
                                    temperature=1.0,
                                    normalize=True,
                                    use_mmd=True,
                                    use_graph=True,
                                    use_rank=True,
                                    mmd_kernel='rbf',
                                    mmd_sigma_list=(0.5,1.0,2.0,4.0),
                                    mmd_degree=2,
                                    weights=(1.0, 0.8, 0.5, 0.5)):
    """
    Multi-relational distillation loss (extended from original).
    
    Args:
        metric: 'euclidean' | 'cosine'
        use_mmd, use_graph, use_rank: 是否引入 MMD/Graph/Ranking
        mmd_kernel: 'linear' | 'poly' | 'rbf'
        weights: (w_rel, w_mmd, w_graph, w_rank)
    """
    assert student_features.dim() == 2 and teacher_features.dim() == 2
    w_rel, w_mmd, w_graph, w_rank = weights

    # --- 原始 relational distillation ---
    if metric == 'euclidean':
        teacher_diffs = teacher_features.unsqueeze(1) - teacher_features.unsqueeze(0)
        student_diffs = student_features.unsqueeze(1) - student_features.unsqueeze(0)
        teacher_dists = torch.norm(teacher_diffs, dim=2, p=2)
        student_dists = torch.norm(student_diffs, dim=2, p=2)
        if normalize:
            eps = 1e-8
            teacher_rel = teacher_dists / (teacher_dists.mean() + eps)
            student_rel = student_dists / (student_dists.mean() + eps)
        else:
            teacher_rel, student_rel = teacher_dists, student_dists
        if temperature != 1.0:
            teacher_rel = teacher_rel / temperature
            student_rel = student_rel / temperature
        rel_loss = F.mse_loss(student_rel, teacher_rel)

    elif metric == 'cosine':
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)
        teacher_sim = torch.matmul(teacher_norm, teacher_norm.t())
        student_sim = torch.matmul(student_norm, student_norm.t())
        if temperature != 1.0:
            teacher_sim = teacher_sim / temperature
            student_sim = student_sim / temperature
        rel_loss = F.mse_loss(student_sim, teacher_sim)
    else:
        raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")

    # --- 额外蒸馏项 ---
    total_loss = w_rel * rel_loss
    if use_mmd:
        total_loss += w_mmd * multi_kernel_mmd(student_features, teacher_features, sigma_list=mmd_sigma_list)
    if use_graph:
        total_loss += w_graph * graph_loss(student_features, teacher_features)
    if use_rank:
        total_loss += w_rank * ranking_loss(student_features, teacher_features)

    return total_loss

def spectral_subspace_loss(s_features, t_features, k_eig=6):
    """
    s_sim, t_sim: similarity matrices (B,B), e.g., cosine sims
    compute normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    get smallest k eigenvectors of L (or largest of normalized adjacency),
    match subspace by projection matrices P = U U^T
    """
    s_sim = torch.matmul(F.normalize(s_features, dim=1),
                         F.normalize(s_features, dim=1).t())
    t_sim = torch.matmul(F.normalize(t_features, dim=1),
                         F.normalize(t_features, dim=1).t())
    device = s_sim.device

    def laplacian_from_sim(sim):
        # ensure symmetric, non-negative
        A = (sim + sim.t()) / 2.0
        # zero diag
        A = A - torch.diag(torch.diag(A))
        deg = A.sum(dim=1)
        # avoid div by zero
        deg_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg + 1e-8))
        L = torch.eye(A.size(0), device=device) - deg_inv_sqrt @ A @ deg_inv_sqrt
        # symmetric
        return (L + L.t()) / 2.0
    
    Ls = laplacian_from_sim(s_sim)
    Lt = laplacian_from_sim(t_sim)
    # eigendecomposition (small B is assumed; if B large, use randomized methods)
    # eigenvalues ascending; take smallest k eigenvectors
    try:
        eigvals_s, eigvecs_s = torch.linalg.eigh(Ls)
        eigvals_t, eigvecs_t = torch.linalg.eigh(Lt)
    except Exception:
        # fallback for older torch
        eigvals_s, eigvecs_s = torch.symeig(Ls, eigenvectors=True)
        eigvals_t, eigvecs_t = torch.symeig(Lt, eigenvectors=True)
    Us = eigvecs_s[:, :k_eig]   # (B, k)
    Ut = eigvecs_t[:, :k_eig]
    # projection matrices
    Ps = Us @ Us.t()
    Pt = Ut @ Ut.t()
    return F.mse_loss(Ps, Pt)

def mg_relational_distillation_3d_loss1(student_features, teacher_features,
                                    metric='euclidean',
                                    temperature=1.0,
                                    normalize=True,
                                    use_mmd=True,
                                    use_graph=True,
                                    use_rank=True,
                                    spectral_k=6,
                                    mmd_sigma_list=(0.5,1.0,2.0,4.0),
                                    weights=(1.0, 0.5, 0.5, 0.5)):
    """
    Multi-relational distillation loss (extended from original).
    
    Args:
        metric: 'euclidean' | 'cosine'
        use_mmd, use_graph, use_rank: 是否引入 MMD/Graph/Ranking
        mmd_kernel: 'linear' | 'poly' | 'rbf'
        weights: (w_rel, w_mmd, w_graph, w_rank)
    """
    assert student_features.dim() == 2 and teacher_features.dim() == 2
    w_rel, w_mmd, w_graph, w_rank = weights

    # --- 原始 relational distillation ---
    if metric == 'euclidean':
        teacher_diffs = teacher_features.unsqueeze(1) - teacher_features.unsqueeze(0)
        student_diffs = student_features.unsqueeze(1) - student_features.unsqueeze(0)
        teacher_dists = torch.norm(teacher_diffs, dim=2, p=2)
        student_dists = torch.norm(student_diffs, dim=2, p=2)
        if normalize:
            eps = 1e-8
            teacher_rel = teacher_dists / (teacher_dists.mean() + eps)
            student_rel = student_dists / (student_dists.mean() + eps)
        else:
            teacher_rel, student_rel = teacher_dists, student_dists
        if temperature != 1.0:
            teacher_rel = teacher_rel / temperature
            student_rel = student_rel / temperature
        rel_loss = F.mse_loss(student_rel, teacher_rel)

    elif metric == 'cosine':
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)
        teacher_sim = torch.matmul(teacher_norm, teacher_norm.t())
        student_sim = torch.matmul(student_norm, student_norm.t())
        if temperature != 1.0:
            teacher_sim = teacher_sim / temperature
            student_sim = student_sim / temperature
        rel_loss = F.mse_loss(student_sim, teacher_sim)
    else:
        raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")

    # --- 额外蒸馏项 ---
    total_loss = w_rel * rel_loss
    if use_mmd:
        total_loss += w_mmd * multi_kernel_mmd(student_features, teacher_features, sigma_list=mmd_sigma_list)
    if use_graph:
        total_loss += w_graph * spectral_subspace_loss(student_features, teacher_features, k_eig=spectral_k)
    if use_rank:
        total_loss += w_rank * ranking_loss(student_features, teacher_features)

    return total_loss

def sliced_wasserstein_distance(x, y, num_projections=100):
    """
    x: [N, D] - joint samples
    y: [N, D] - product of marginals samples
    """
    d = x.shape[1]
    projections = torch.randn((num_projections, d), device=x.device)
    projections = F.normalize(projections, dim=1)  # shape: [num_proj, D]
    
    proj_x = x @ projections.T  # [N, num_proj]
    proj_y = y @ projections.T  # [N, num_proj]

    proj_x = torch.sort(proj_x, dim=0)[0]
    proj_y = torch.sort(proj_y, dim=0)[0]

    return ((proj_x - proj_y) ** 2).mean()

def iic_loss(Zc, Zv, num_proj=100):
    """
    Zc: [bs, 4096] - content features
    Zv: [bs, 768]  - viewpoint features
    """
    bs = Zc.size(0)

    # 归一化到单位球面
    Zc = F.normalize(Zc, dim=1)  # [bs, 4096]
    Zv = F.normalize(Zv, dim=1)  # [bs, 768]

    # 构造联合分布样本
    joint = torch.cat([Zc, Zv], dim=1)  # [bs, 4864]

    # 构造边缘分布近似样本（打乱 Zc 和 Zv）
    Zc_shuffle = Zc[torch.randperm(bs)]
    Zv_shuffle = Zv[torch.randperm(bs)]
    product = torch.cat([Zc_shuffle, Zv_shuffle], dim=1)  # [bs, 4864]

    # 计算 SWD 作为 IIC 损失
    loss_iic = sliced_wasserstein_distance(joint, product, num_proj)
    return loss_iic


class CLUBForCategorical(nn.Module): # Update 04/27/2022
    '''
    This class provide a CLUB estimator to calculate MI upper bound between vector-like embeddings and categorical labels.
    Estimate I(X,Y), where X is continuous vector and Y is discrete label.
    '''
    def __init__(self, input_dim, label_num, hidden_size=None):
        '''
        input_dim : the dimension of input embeddings
        label_num : the number of categorical labels 
        '''
        super().__init__()
        
        if hidden_size is None:
            self.variational_net = nn.Linear(input_dim, label_num)
        else:
            self.variational_net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, label_num)
            )
            
    def forward(self, inputs, labels):
        '''
        inputs : shape [batch_size, input_dim], a batch of embeddings
        labels : shape [batch_size], a batch of label index
        '''
        logits = self.variational_net(inputs)  #[sample_size, label_num]
        
        # log of conditional probability of positive sample pairs
        #positive = - nn.functional.cross_entropy(logits, labels, reduction='none')    
        sample_size, label_num = logits.shape
        
        logits_extend = logits.unsqueeze(1).repeat(1, sample_size, 1)  # shape [sample_size, sample_size, label_num]
        labels_extend = labels.unsqueeze(0).repeat(sample_size, 1)     # shape [sample_size, sample_size]

        # log of conditional probability of negative sample pairs
        log_mat = - nn.functional.cross_entropy(
            logits_extend.reshape(-1, label_num),
            labels_extend.reshape(-1, ),
            reduction='none'
        )
        
        log_mat = log_mat.reshape(sample_size, sample_size)
        positive = torch.diag(log_mat).mean()
        negative = log_mat.mean()
        return positive - negative

    def loglikeli(self, inputs, labels):
        logits = self.variational_net(inputs)
        return - nn.functional.cross_entropy(logits, labels)
    
    def learning_loss(self, inputs, labels):
        return - self.loglikeli(inputs, labels)
    

class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
   

class CLUBMean(nn.Module):  # Set variance of q(y|x) to 1, logvar = 0. Update 11/26/2022
    def __init__(self, x_dim, y_dim, hidden_size=None):
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        
        super(CLUBMean, self).__init__()
   
        if hidden_size is None:
            self.p_mu = nn.Linear(x_dim, y_dim)
        else:
            self.p_mu = nn.Sequential(nn.Linear(x_dim, int(hidden_size)),
                                       nn.ReLU(),
                                       nn.Linear(int(hidden_size), y_dim))


    def get_mu_logvar(self, x_samples):
        # variance is set to 1, which means logvar=0
        mu = self.p_mu(x_samples)
        return mu, 0
    
    def forward(self, x_samples, y_samples):

        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2.
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2.

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


    
    
    
class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)

    
class NWJ(nn.Module):   
    def __init__(self, x_dim, y_dim, hidden_size):
        super(NWJ, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
                                    
    def forward(self, x_samples, y_samples): 
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))-1.  #shape [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1) - np.log(sample_size)).exp().mean() 
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


class L1OutUB(nn.Module):  # naive upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(L1OutUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples): 
        batch_size = y_samples.shape[0]
        mu, logvar = self.get_mu_logvar(x_samples)

        positive = (- (mu - y_samples)**2 /2./logvar.exp() - logvar/2.).sum(dim = -1) #[nsample]

        mu_1 = mu.unsqueeze(1)          # [nsample,1,dim]
        logvar_1 = logvar.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)            # [1,nsample,dim]
        all_probs =  (- (y_samples_1 - mu_1)**2/2./logvar_1.exp()- logvar_1/2.).sum(dim = -1)  #[nsample, nsample]

        diag_mask =  torch.ones([batch_size]).diag().unsqueeze(-1).cuda() * (-20.)
        negative = log_sum_exp(all_probs + diag_mask,dim=0) - np.log(batch_size-1.) #[nsample]
      
        return (positive - negative).mean()
        
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

    
class VarUB(nn.Module):  #    variational upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(VarUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
            
    def forward(self, x_samples, y_samples): #[nsample, 1]
        mu, logvar = self.get_mu_logvar(x_samples)
        return 1./2.*(mu**2 + logvar.exp() - 1. - logvar).mean()
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
