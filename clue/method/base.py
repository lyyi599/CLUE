import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from src.model import get_encoder, get_projector, get_predictor
import time

class BaseMethod(nn.Module):
    """
        Base class for self-supervised loss implementation.
        It includes encoder and projector for training function.
    """
    def __init__(self, args):
        super().__init__()
        self.is_vit = args.is_vit = ('vit' in args.arch)
        self.encoder, self.out_size = get_encoder(args)
        self.projector = get_projector(self.out_size, args)
        self.predictor = get_predictor(args)
        self.momentum = args.momentum
        self.device = args.device

        # ================ feature attention =======================
        # self.attn = args.attn
        # if self.attn is True:
        #     self.attention = nn.Sequential(
        #         nn.Linear(2048, 2048),
        #         nn.ReLU(),
        #         nn.Linear(2048, 1),
        #         nn.Sigmoid()
        #     )
        #     for layer in self.attention:
        #         if isinstance(layer, nn.Linear):
        #             nn.init.constant_(layer.weight, 1.0)  # 权重全1
        #             if layer.bias is not None:
        #                 nn.init.constant_(layer.bias, 1.0)  # 偏置全1（如果存在）

        # ================ clip的融合方式 =======================
        self.clip_fusion = args.clip_fusion
        self.with_texts = args.with_texts

        # 加载 CLIP 模型和处理器
        if self.clip_fusion!=None:
            self.clip_model = load_clip_to_cpu("ViT-B/16")
            self.clip_model = self.clip_model.eval()
            # clip_model = load_clip_to_cpu("RN50")
            # self.clip_model_v = clip_model.visual

            # 冻结 CLIP 模块
            for param in self.clip_model.parameters():
                param.requires_grad = False
        if self.with_texts!=None:
            self.clip_model = load_clip_to_cpu("ViT-B/16")
            self.clip_model = self.clip_model.eval()
            # self.clip_model_t = clip_model.transformer

            # 冻结 CLIP 模块
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def ForwardWrapper(self, samples, encoder, projector):

        if self.is_vit:
            # concate views with the same image size
            views = len(samples)
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in samples]),
                return_counts=True,
            )[1], 0)

            start_idx, h = 0, torch.empty(0).to(self.device)

            for end_idx in idx_crops:
                _out = encoder(torch.cat(samples[start_idx: end_idx]))
                h = torch.cat((h, _out))
                start_idx = end_idx
            # Run the projector forward on the concatenated features.
            emb = projector(h)
            if torch.is_grad_enabled() and self.predictor:
                emb = self.predictor(emb)

            emb = F.normalize(emb)
            emb = emb.chunk(views)
            h = h.chunk(views)
        else:
            # do not concate different views if BN is in the model 
            # As it will disrupt the zero-mean, unit-variance distribution
            # h = [encoder(x) for x in samples]
            h = []
            feature_map = []
            for x in samples:
                h_item, fm_item = encoder(x)
                h.append(h_item)
                feature_map.append(fm_item)
            emb = [projector(x) for x in h]
            if torch.is_grad_enabled() and self.predictor:
                emb = [self.predictor(x) for x in emb]

            emb = [F.normalize(x) for x in emb]

        for i in range(len(h)):
            h[i] = F.normalize(h[i])
        # if self.attn is True:
        #     attn_weights = self.attention(h[0])  # [N, 1]
        #     h[0] = h[0] * attn_weights
        # h_weak = F.normalize(h[0])
        # h_stansard = F.normalize(h[1])

        # if not torch.is_grad_enabled():
        #     # print("not grad enabled")
        #     emb = [concat_all_gather(x) for x in emb]
        #     h_weak = concat_all_gather(h_weak)
        #     # h_stansard = concat_all_gather(h_stansard)

        # return h_weak, emb, h_stansard, feature_map
        return h, emb, feature_map

    def cross_entropy(self, s, q):
        return - torch.sum(q * F.log_softmax(s, dim=1), dim=-1).mean()
    
    @torch.no_grad()
    def sinkhorn_knopp(self, scores, temp=0.05, n_iterations=3):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(scores / temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    @torch.no_grad()
    def update_momentum_params(self, m):
        """
        Update of the momentum encoder and projector
        """
        for param_q, param_k in zip(self.encoder.parameters(), 
                                    self.momentum_encoder.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
        for param_q, param_k in zip(self.projector.parameters(),
                                    self.momentum_projector.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
        
    def forward(self, samples):
        raise NotImplementedError

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output



# clip 模块
from method.clip import clip
import os
# 设置 Hugging Face Mirror 环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch.nn.functional as F


def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, './pretrain')
    model = torch.jit.load(model_path, map_location="cpu").eval()
    state_dict = None

    model = clip.build_model(state_dict or model.state_dict())
    model = model.float()
    # print(f"Loaded model from {model_path}")
    # print("model keys: ", model)
    # time.sleep(1000)

    return model
