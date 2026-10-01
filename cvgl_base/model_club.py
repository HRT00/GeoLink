import math
import warnings

import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvgl_base import helper
from .backbones.pointclip import PointCLIP_Model, load_clip_to_cpu
# from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from .point_pn.point_pn import Point_PN
# from Dassl3D.dassl.modeling import load_checkpoint


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class VPRModel(nn.Module):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                 # ---- Backbone 主干网络
                 model_name='dinov2_vitb14_MixVPR',
                 backbone_arch='dinov2_vitb14',
                 pretrained=True,
                 layers_to_freeze=1,
                 layers_to_crop=[],
                 layer1=20,
                 use_cls=False,
                 norm_descs=True,

                 # ---- Aggregator 聚合方法
                 agg_arch='MixVPR',  # CosPlace, NetVLAD, GeM
                 agg_config={},
                 ):
        super().__init__()
        self.pretrained = pretrained  # 是否预训练
        self.layers_to_freeze = layers_to_freeze  # 冻结网络层名称
        self.layers_to_crop = layers_to_crop  # layers_to_crop=[4],  # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
        self.layer1 = layer1
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.agg_config = agg_config  # 聚合方法参数
        # self.save_hyperparameters()  # write hyperparams into a file
        self.model_name = model_name

        self.batch_acc = []  # we will keep track of the % of trivial pairs/triplets at the loss level

        # ----------------------------------
        # get the backbone and the aggregator 获得主干网络和聚合器
        self.backbone = helper.get_backbone(backbone_arch=backbone_arch, pretrained=pretrained, layer1=self.layer1, use_cls=self.use_cls,
                                            norm_descs=self.norm_descs)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        
        # self.proj = nn.Sequential(
        #     nn.Linear(4096, 5120),
        #     nn.ReLU()
        # )

    # the forward pass of the lightning model
    def forward(self, x):   # x: [B, 3, 448, 448]
        x, view_x, cls_x = self.backbone(x)    # x: [B, 1024, 32, 32]
        # x = self.aggregator(x)
        # x = self.proj(x)
        return x, view_x, cls_x


class MultiScaleCompressor(nn.Module):
    def __init__(self, output_dim=4096, groups=4):
        super(MultiScaleCompressor, self).__init__()
        assert output_dim % groups == 0, "output_dim must be divisible by groups"
        bottleneck_dim = output_dim // groups

        self.branch1 = nn.Sequential(
            nn.Conv1d(1, bottleneck_dim, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, bottleneck_dim, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, bottleneck_dim, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(1, bottleneck_dim, kernel_size=1),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = x.unsqueeze(1)  # -> (batch_size, 1, input_dim)

        x1 = self.branch1(x)  # (batch_size, bottleneck_dim, input_dim)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)  # (batch_size, output_dim, input_dim)

        out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)  # -> (batch_size, output_dim)
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleMoECompressor(nn.Module):
    def __init__(self, input_dim, output_dim=4096):
        super().__init__()
        self.num_experts = 3
        self.num_shared = 1
        self.total = self.num_experts + self.num_shared

        assert output_dim % self.total == 0, "output_dim must be divisible by 4"
        self.bottleneck_dim = output_dim // self.total  # 4096/4 = 1024

        # 3 个多尺度专家：dilation=1,2,3
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, self.bottleneck_dim, kernel_size=3, dilation=d, padding=d),
                nn.BatchNorm1d(self.bottleneck_dim),
                nn.ReLU(inplace=True)
            ) for d in [1 for _ in range(self.num_experts)]
        ])

        # 1 个共享专家（1×1 卷积）
        self.shared_expert = nn.Sequential(
            nn.Conv1d(1, self.bottleneck_dim, kernel_size=1),
            nn.BatchNorm1d(self.bottleneck_dim),
            nn.ReLU(inplace=True)
        )

        # 对 3 个专家做加权的门控网络
        self.gating = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, self.num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        x: (B, input_dim)
        return: (B, 4096)
        """
        x_exp = x.unsqueeze(1)  # -> (B,1,input_dim)

        # 1) 计算每个专家的向量输出
        expert_vectors = []
        for expert in self.experts:
            out = expert(x_exp)                               # (B,1024,input_dim)
            vec = F.adaptive_avg_pool1d(out,1).squeeze(-1)    # -> (B,1024)
            expert_vectors.append(vec)

        # 2) 计算共享专家向量
        shared_out = self.shared_expert(x_exp)                # (B,1024,input_dim)
        shared_vector = F.adaptive_avg_pool1d(shared_out,1)   # -> (B,1024,1)
        shared_vector = shared_vector.squeeze(-1)             # -> (B,1024)

        # 3) 门控权重，对 3 个专家分别加权
        gate_weights = self.gating(x)                         # (B,3)
        # 把每个专家向量都乘上对应权重（保持单独通路）
        weighted_experts = []
        for i, vec in enumerate(expert_vectors):
            w = gate_weights[:, i].unsqueeze(-1)              # (B,1)
            weighted_experts.append(vec * w)                  # (B,1024)

        # 4) 直接拼接 3 条加权后专家 + 1 条共享专家 -> (B,4*1024)
        out = torch.cat(weighted_experts + [shared_vector], dim=1)  # -> (B,4096)
        return out


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def softmax_one(x, dim=-1):
    return (x.exp() + 1e-6) / (x.exp().sum(dim, keepdim=True) + 1)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,         # 输入channel
        num_heads: int,             # attention的head数
        downsample_rate: int = 1,   # 下采样
        smax_mode=None
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        # qkv获取
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.smax = smax_mode
    def _separate_heads(self, x, num_heads: int) :
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q, k, v) :
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        # B,N_heads,N_tokens,C_per_head
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B,N_heads,N_tokens,C_per_head
        # Scale
        attn = attn / math.sqrt(c_per_head)
        if self.smax is not None:
            attn = softmax_one(attn, dim=-1)
        else:
            attn = torch.softmax(attn, dim=-1)
        # Get output
        out = attn @ v
        # # B,N_tokens,C
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out

class OutAttenBlock(nn.Module):
    def __init__(self, dim, num_heads, 
                 mlp_ratio=4.0,
                 activation = nn.ReLU,   
                 attention_downsample_rate: int = 1, 
                 norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.norm3 = norm_layer(dim)

    def forward(self, q, k, v):
        attn_out = self.cross_attn(q, k, v)
        queries = q + attn_out
        queries = self.norm1(queries)
        queries = queries + self.attn(queries)
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        return queries


class PromptBlock(nn.Module):
    def __init__(self, 
                embedding_dim: int,         # 输入channel
                num_heads: int,             # attention的head数
                mlp_ratio: float = 4.0,        # MLP中间channel
                activation = nn.ReLU,      # 激活层
                attention_downsample_rate: int = 2,         # 下采样
                skip_first_layer_pe: bool = False,):
        super().__init__()
        self.cross_attn = CrossAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.self_attn = CrossAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        mlp_hidden_dim = int(embedding_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embedding_dim, hidden_features=mlp_hidden_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        return queries, keys
    
class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,         # 输入channel
        num_heads: int,             # attention的head数
        mlp_ratio: float = 4.0,        # MLP中间channel
        activation = nn.ReLU,      # 激活层
        attention_downsample_rate: int = 2,         # 下采样
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = CrossAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = CrossAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        mlp_hidden_dim = int(embedding_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embedding_dim, hidden_features=mlp_hidden_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = CrossAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.skip_first_layer_pe = skip_first_layer_pe
        
    def forward(self, queries, keys, query_pe, key_pe):

        # queries：标记点编码相关(原始标记点编码经过一系列特征提取)
        # keys：原始图像编码相关(原始图像编码经过一系列特征提取)
        # query_pe：原始标记点编码
        # key_pe：原始图像位置编码
        # 第一轮本身queries==query_pe没比较再"残差"
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys

class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        # 层数
        depth: int,
        # 输入channel
        embedding_dim: int,
        # attention的head数
        num_heads: int,
        # MLP内部channel
        mlp_ratio: float,
        activation = nn.ReLU,
        attention_downsample_rate: int = 2,
        use_to_way = True,
        out_method = None,
    ) -> None:
        super().__init__()
        self.depth = depth      # 层数
        self.embedding_dim = embedding_dim          # 输入channel
        self.num_heads = num_heads                  # attention的head数
        self.mlp_ratio = mlp_ratio                      # MLP内部隐藏channel
        self.layers = nn.ModuleList()
        for i in range(depth):
            if use_to_way:
                self.layers.append(
                    TwoWayAttentionBlock(
                        embedding_dim=embedding_dim,    # 输入channel
                        num_heads=num_heads,            # attention的head数
                        mlp_ratio=mlp_ratio,                # MLP中间channel
                        activation=activation,          # 激活层
                        attention_downsample_rate=attention_downsample_rate,      # 下采样
                        skip_first_layer_pe=(i == 0),
                    )
                )
            else:
                self.layers.append(
                    PromptBlock(
                        embedding_dim=embedding_dim,    # 输入channel
                        num_heads=num_heads,            # attention的head数
                        mlp_ratio=mlp_ratio,                # MLP中间channel
                        activation=activation,          # 激活层
                        attention_downsample_rate=attention_downsample_rate,      # 下采样
                        skip_first_layer_pe=(i == 0),
                    )
                )
        self.out_method = out_method
        if out_method is None:
            self.final_attn_token_to_image = OutAttenBlock(
                embedding_dim, num_heads, mlp_ratio=self.mlp_ratio , attention_downsample_rate=attention_downsample_rate
            )
        elif out_method == 'Atten':
            self.final_attn_token_to_image = CrossAttention(
                embedding_dim, num_heads, downsample_rate=attention_downsample_rate
            )

        self.norm_final_attn = nn.LayerNorm(embedding_dim)
        self.out_token = nn.Parameter(torch.zeros(1, self.embedding_dim))
        trunc_normal_(self.out_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        image_embedding,
        image_pe,
        point_embedding,
    ):
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, l, c = image_embedding.shape
        # 图像编码(image_encoder的输出)
        # BxHWxC=>B,N,C
        # image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        # 图像位置编码
        # BxHWxC=>B,N,C
        # image_pe = image_pe.flatten(2).permute(0, 2, 1)
        
        # 标记点编码
        # B,N,C
        queries = point_embedding
        keys = image_embedding
        # -----TwoWayAttention-----
        
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
        # -----TwoWayAttention-----

        q = queries + point_embedding
        k = keys + image_pe
        if self.out_method == 'no_out':
            return queries, keys
        # -----Attention-----
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        # -----Attention-----
        queries = queries + attn_out
        # queries = attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


class TimmModel(nn.Module):

    def __init__(self,
                 args,
                 model_name='dinov2_vitb14_MixVPR',
                 pretrained_path=None,
                 backbone_arch='',
                 pretrained=True,
                 img_size=224,
                 layer1=8,
                 neck = 'no',
                 num_classes = 701, # 701 for University
                 # Aggregator 聚合方法
                 agg_arch='MixVPR',
                 agg_config={},
                 ):

        super(TimmModel, self).__init__()

        self.img_size = img_size
        self.neck = neck
        self.num_patchs = 8
        self.pn_dim = 396 # 576 or 432
        self.in_planes = 3072
        self.num_classes = num_classes
        
        if "dino" in backbone_arch:
            self.model = VPRModel(backbone_arch=backbone_arch, agg_arch=agg_arch, layer1=layer1, agg_config=agg_config)
        elif "vitt" in backbone_arch:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # point models
        # self.pc_model = PointCLIP_Model(args, clip_model)
        self.enc_pn = Point_PN(input_points=1024, embed_dim=self.pn_dim)
        
        # print('Turning off gradients in both visual and textual encoders')
        # if args.use_pretrained:
        #     print('Loading pretrained weights')
        #     model_file = 'ckpt/adapter/model-best.pth.tar'
        #     checkpoint = load_checkpoint(model_file)
        #     state_dict = checkpoint['state_dict']
        #     epoch = checkpoint['epoch']
        #     print('Loading pretrained PointCLIP weights from epoch {}'.format(epoch))
        #     self.pc_model.adapter.load_state_dict(state_dict, strict=False)
        
        # for name, param in self.pc_model.named_parameters():
        #     if 'adapter' not in name:
        #         param.requires_grad_(False)
        #     else:
        #         param.data = param.data.to(torch.float32)

        if pretrained_path:
            # 加载预训练模型的权重，但不包括输出层的权重
            state_dict = torch.load(pretrained_path)
            print("Start from:", pretrained_path)
            self.load_state_dict(state_dict)
        
        # if self.neck == 'no':
        #     self.classifer = nn.Linear(self.in_planes, self.num_classes)
        # elif self.neck == 'bnneck':
        #     self.bottleneck = nn.BatchNorm1d(self.in_planes)
        #     self.bottleneck.bias.requires_grad_(False)  # no shift
        #     self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        #     self.bottleneck.apply(weights_init_kaiming)
        #     self.classifier.apply(weights_init_classifier)

        # self.mix_pc = nn.Sequential(
        #     nn.Linear(4608, self.in_planes),
        #     nn.ReLU()
        # )
        self.mix_pc = MultiScaleMoECompressor(input_dim=3168, output_dim=self.in_planes) # 4608 or 3168

    def get_config(self):
        # data_config = timm.data.resolve_model_data_config(self.model)
        # data_config = self.model.default_cfg
        data_config = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, img1, img2=None, pc=None):
        
        if self.training:
            if img2 is not None:
                image_features1, _, _ = self.model(img1)  # [B, 4096]
                image_features2, _, _  = self.model(img2)  # [B, 4096]

                x = pc.permute(0, 2, 1)
                x = torch.cat((x, x[:, 1:2] - x[:, 1:2].min()), dim=1)
                pc_features = self.enc_pn(x, pc)
                pc_features = self.mix_pc(pc_features)

                return image_features1, image_features2, pc_features
            else:
                image_features = self.model(img1)
        else:
            image_features = self.model(img1)

            return image_features[0]

class TimmModel_eval(nn.Module):

    def __init__(self,
                 args,
                 model_name='dinov2_vitb14_MixVPR',
                 pretrained_path=None,
                 backbone_arch='',
                 pretrained=True,
                 img_size=224,
                 layer1=8,
                 neck = 'no',
                 num_classes = 701, # 701 for University
                 # Aggregator 聚合方法
                 agg_arch='MixVPR',
                 agg_config={},
                 ):

        super(TimmModel_eval, self).__init__()

        self.img_size = img_size
        self.neck = neck
        self.num_patchs = 8
        self.pn_dim = 576
        self.in_planes = 4096
        self.num_classes = num_classes
        
        if "dino" in backbone_arch:
            self.model = VPRModel(backbone_arch=backbone_arch, agg_arch=agg_arch, layer1=layer1, agg_config=agg_config)
        elif "vitt" in backbone_arch:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # point models
        # self.pc_model = PointCLIP_Model(args, clip_model)
        self.enc_pn = Point_PN(embed_dim=self.pn_dim)
        
        # print('Turning off gradients in both visual and textual encoders')
        # if args.use_pretrained:
        #     print('Loading pretrained weights')
        #     model_file = 'ckpt/adapter/model-best.pth.tar'
        #     checkpoint = load_checkpoint(model_file)
        #     state_dict = checkpoint['state_dict']
        #     epoch = checkpoint['epoch']
        #     print('Loading pretrained PointCLIP weights from epoch {}'.format(epoch))
        #     self.pc_model.adapter.load_state_dict(state_dict, strict=False)
        
        # for name, param in self.pc_model.named_parameters():
        #     if 'adapter' not in name:
        #         param.requires_grad_(False)
        #     else:
        #         param.data = param.data.to(torch.float32)

        if pretrained_path:
            # 加载预训练模型的权重，但不包括输出层的权重
            state_dict = torch.load(pretrained_path)
            print("Start from:", pretrained_path)
            self.load_state_dict(state_dict)
        
        if self.neck == 'no':
            self.classifer = nn.Linear(self.in_planes, self.num_classes)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

        # self.mix_pc = nn.Sequential(
        #     nn.Linear(4608, self.in_planes),
        #     nn.ReLU()
        # )
        self.mix_pc = MultiScaleMoECompressor(input_dim=4608, output_dim=self.in_planes)

    def get_config(self):
        # data_config = timm.data.resolve_model_data_config(self.model)
        # data_config = self.model.default_cfg
        data_config = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, img1, img2=None, pc=None):
        
        if self.training:
            if img2 is not None:
                image_features1, _, _ = self.model(img1)  # [B, 4096]
                image_features2, _, _  = self.model(img2)  # [B, 4096]

                x = pc.permute(0, 2, 1)
                x = torch.cat((x, x[:, 1:2] - x[:, 1:2].min()), dim=1)
                pc_features = self.enc_pn(x, pc)
                pc_features = self.mix_pc(pc_features)

                return image_features1, image_features2, pc_features
            else:
                image_features = self.model(img1)
        else:
            if img2 is not None:
                image_features1, _, _ = self.model(img1)
                image_features2, _, _  = self.model(img2)  # [B, 4096]

                x = pc.permute(0, 2, 1)
                x = torch.cat((x, x[:, 1:2] - x[:, 1:2].min()), dim=1)
                pc_features = self.enc_pn(x, pc)
                pc_features = self.mix_pc(pc_features)

                return image_features1, image_features2, pc_features
            else:
                image_features = self.model(img1)

                return image_features
