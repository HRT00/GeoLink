# Parametric Networks for 3D Point Cloud Classification
import torch
import torch.nn as nn

from .model_utils import *

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # FPS
        fps_idx = farthest_point_sample(xyz, self.group_num).long() 
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x


# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta, block_num, dim_expansion, type):
        super().__init__()
        self.type = type
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)
        if dim_expansion == 1:
            expand = 2
        elif dim_expansion == 2:
            expand = 1
        self.linear1 = Linear1Layer(out_dim * expand, out_dim, bias=False)
        self.linear2 = []
        for i in range(block_num):
            self.linear2.append(Linear2Layer(out_dim, bias=True))
        self.linear2 = nn.Sequential(*self.linear2)


    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):

        # Normalization
        if self.type == 'mn40':
            mean_xyz = lc_xyz.unsqueeze(dim=-2)
            std_xyz = torch.std(knn_xyz - mean_xyz)
            knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        elif self.type == 'scan':
            knn_xyz = knn_xyz.permute(0, 3, 1, 2)
            knn_xyz -= lc_xyz.permute(0, 2, 1).unsqueeze(-1)
            knn_xyz /= torch.abs(knn_xyz).max(dim=-1, keepdim=True)[0]
            knn_xyz = knn_xyz.permute(0, 2, 3, 1)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Linear
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x = self.linear1(knn_x.reshape(B, -1, G*K)).reshape(B, -1, G, K)

        # Geometry Extraction
        knn_x_w = self.geo_extract(knn_xyz, knn_x)

        # Linear
        for layer in self.linear2:
            knn_x_w = layer(knn_x_w)

        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        return lc_x
    

# Linear layer 1
class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


# Linear Layer 2
class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True):
        super(Linear2Layer, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels/2),
                    kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm2d(int(in_channels/2)),
            self.act
        )
        self.net2 = nn.Sequential(
                nn.Conv2d(in_channels=int(in_channels/2), out_channels=in_channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm2d(in_channels)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)
    

# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
   
        
    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().cuda()     
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.cat([sin_embed, cos_embed], -1)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).contiguous()
        position_embed = position_embed.view(B, self.out_dim, G, K)

        # Weigh
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        return knn_x_w

# ----- Shared MLP Expert -----
class SharedExpert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        return self.net(x)

# ----- Router -----
class Router(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, num_experts),
        )

    def forward(self, x):  # x: (B, T_seg, C)
        return self.net(x)

# ----- Fine-Grained Expert Segmentation MoE Fusion -----
class FGESMoEFusion(nn.Module):
    def __init__(self, dim, hidden, num_experts, top_k, output_tokens=4096):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_tokens = output_tokens

        # one router per segment to enable fine-grained segmentation
        self.segment_routers = nn.ModuleList([
            Router(dim, num_experts + 1)  # +1 for shared expert
            for _ in range(num_experts)
        ])

        # experts + shared expert
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden), nn.ReLU(),
                nn.Linear(hidden, dim)
            ) for _ in range(num_experts)
        ])
        self.shared_expert = SharedExpert(dim, hidden)
        self.shared_idx = num_experts

    def forward(self, x_cat):
        # x_cat: (B, C, T)
        # 1) normalize total tokens
        x_flat = F.adaptive_max_pool1d(x_cat, self.output_tokens)  # -> (B, C, T_out)
        x_flat = x_flat.permute(0, 2, 1)  # -> (B, T_out, C)

        # 2) segment tokens evenly
        segments = x_flat.chunk(self.num_experts, dim=1)  # list of (B, T_seg, C)

        # 3) compute per-segment router scores
        seg_scores = []
        for i, seg in enumerate(segments):
            scores = self.segment_routers[i](seg)  # (B, T_seg, E+1)
            delta = scores.new_zeros(scores.shape)
            delta[..., self.shared_idx] = 1e3
            scores = scores + delta
            scores = F.softmax(scores, dim=-1)
            # ensure shared expert always possible
            seg_scores.append(scores)
        # 4) concatenate back along token dimension
        scores = torch.cat(seg_scores, dim=1)  # (B, T_out, E+1)

        # 5) select top-k experts per token
        top_vals, top_idxs = scores.topk(self.top_k, dim=-1)  # (B, T_out, top_k)

        # 6) dispatch and combine expert outputs
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            idx = top_idxs[..., k]           # (B, T_out)
            w = top_vals[..., k].unsqueeze(-1)  # (B, T_out, 1)
            for exp_id in range(self.num_experts + 1):
                mask = (idx == exp_id)         # (B, T_out)
                if mask.any():
                    expert_fn = self.shared_expert if exp_id == self.shared_idx else self.experts[exp_id]
                    expert_out = expert_fn(x_flat)  # (B, T_out, C)
                    output += mask.unsqueeze(-1).float() * w * expert_out

        # (B, T_out, C) -> (B, C, T_out)
        return output.permute(0, 2, 1)

# ----- EncP with DeepSeek MoE Fusion -----
class EncP(nn.Module):
    def __init__(self, in_channels, input_points, num_stages, embed_dim, k_neighbors, alpha, beta, LGA_block, dim_expansion, type):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        self.raw_point_embed = Linear1Layer(in_channels, self.embed_dim, bias=False)

        self.FPS_kNN_list = nn.ModuleList()
        self.LGA_list = nn.ModuleList()
        self.Pooling_list = nn.ModuleList()

        out_dim = self.embed_dim
        group_num = self.input_points

        channel_dims = [512, 256, 128, 64]  # your per-stage output dims
        # self.token_nums = [1152, 2304, 4608, 4608]
        self.proj_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(ch, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU()
            ) for ch in channel_dims
        ])

        for i in range(self.num_stages):
            out_dim = out_dim * dim_expansion[i]
            group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta, LGA_block[i], dim_expansion[i], type))
            self.Pooling_list.append(Pooling(out_dim))

        self.moe_fusion = FGESMoEFusion(dim=128, hidden=256, num_experts=num_stages, top_k=4, output_tokens=4096)

    def forward(self, xyz, x):
        x = self.raw_point_embed(x)
        multi_scale_feats = []

        for i in range(self.num_stages):
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)
            x = self.Pooling_list[i](knn_x_w)  # (B, C_i, N_i)
            x_proj = self.proj_list[i](x.permute(0, 2, 1))     # (B, 128, N_i)
            multi_scale_feats.append(x_proj)

        x = torch.cat(multi_scale_feats, dim=-1)  # (B, 128, total_N)
        x_out = self.moe_fusion(x)                # (B, 128, 4096)
        x_out = x_out.max(1)[0] + x_out.mean(1)       # (B, 4096)
        return x_out
    
# Parametric Network
class Point_PN(nn.Module):
    def __init__(self, in_channels=4, input_points=1024, num_stages=4, embed_dim=36, k_neighbors=40, beta=100, alpha=1000, LGA_block=[2,1,1,1], dim_expansion=[2,2,2,1], type='scan'):
        super().__init__()
        # Parametric Encoder
        self.EncP = EncP(in_channels, input_points, num_stages, embed_dim, k_neighbors, alpha, beta, LGA_block, dim_expansion, type)

    def forward(self, x, xyz):
        # xyz: point coordinates
        # x: point features

        # Parametric Encoder
        x = self.EncP(xyz, x)

        return x
    
if __name__=="__main__":
    x = torch.randn(4, 4, 1024).cuda()
    xyz = torch.randn(4, 1024, 3).cuda()
    model = Point_PN(embed_dim=576).cuda()
    out = model(x, xyz)
    print(out.shape)
    