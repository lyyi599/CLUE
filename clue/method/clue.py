import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
from itertools import chain
from method.base import BaseMethod
import torch.distributed as dist
from method.pa import PA
import time
import pdb
from method.clip import clip

class AlignLoss(nn.Module):
    def __init__(self, t_q=1, t_k=1):
        super().__init__()

        self.t_q = t_q
        self.t_k = t_k

        self.loss_fn = nn.MSELoss()

    def self_dist(self, q, k):
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        loss = self.loss_fn(q, k)

        return loss

    def forward(self, q_feats, k_feats):
        loss = torch.tensor(0).to(q_feats[0])
        for q_feat, k_feat in zip(q_feats, k_feats):
            loss += self.self_dist(q_feat, k_feat)

        return loss

# 基于part对比的对比损失
def contrastive_loss(q, k, temperature=0.07):
    """
    计算两个张量之间的对比损失
    q, k: 形状为(N, K, D)的张量
    temperature: 对比损失中的温度参数
    """
    # 获取batch的大小和簇数
    N, K, D = q.shape
    
    # 将张量转换为形状(N*K, D)以计算相似度
    q_flat = q.view(N * K, D)
    k_flat = k.view(N * K, D)

    # 计算所有样本之间的余弦相似度
    similarity_matrix = F.cosine_similarity(q_flat.unsqueeze(1), k_flat.unsqueeze(0), dim=-1)
    
    # 对角线上的相似度是正样本
    labels = torch.arange(N * K).to(q.device)
    
    # 计算对比损失
    sim_pos = similarity_matrix[labels, labels]  # 正样本相似度
    sim_neg = similarity_matrix - torch.eye(N * K).to(q.device) * 1e9  # 负样本相似度，屏蔽对角线

    # 计算logits
    logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)  # [N*K, N*K]的logits矩阵
    
    # 使用softmax计算损失
    labels = torch.zeros(N * K, dtype=torch.long).to(q.device)
    loss = F.cross_entropy(logits / temperature, labels)
    
    return loss


# 基于样本的对比损失
def contrastive_loss_sample(z1, z2, temperature=0.07):
    """
    计算对比损失，并返回总损失、正样本损失和负样本损失
    
    参数:
        z1 (torch.Tensor): 形状为 (N, D) 的张量
        z2 (torch.Tensor): 形状为 (N, D) 的张量
        temperature (float): 温度参数，默认0.07
        
    返回:
        total_loss (torch.Tensor): 总对比损失
        pos_loss (torch.Tensor): 正样本损失
        neg_loss (torch.Tensor): 负样本损失
    """
    # L2归一化
    z1_norm = F.normalize(z1, p=2, dim=1)
    z2_norm = F.normalize(z2, p=2, dim=1)
    
    # 计算相似度矩阵 (N, N)
    sim_matrix = torch.mm(z1_norm, z2_norm.T) / temperature
    
    # 提取正样本对的相似度（对角线）
    pos_sim = sim_matrix.diag()  # (N,)
    
    # 计算每个样本的分母的对数值（包括正样本）
    log_denominator = torch.logsumexp(sim_matrix, dim=1)  # (N,)
    
    # 分解损失项
    pos_loss = -pos_sim.mean()
    neg_loss = log_denominator.mean()
    total_loss = pos_loss + neg_loss
    
    return total_loss, pos_loss, neg_loss
    
class CLUE(BaseMethod):
    # CLUE using the momentum network, better performance

    def __init__(self, args):
        super().__init__(args)

        # ============== part clustering ============
        self.is_parts = args.is_parts
        self.part_method = args.part_method
        self.n_parts = args.n_parts
        if self.is_parts=="proto":
            temp_vector = generate_orthonormal_vectors(self.n_parts, 2048)
            self.part_proto = nn.Parameter(temp_vector, requires_grad=True)
        elif self.is_parts=="pa":
            self.net_vlad = PA(num_clusters=args.n_parts, dim=2048, alpha=1.0)
            if self.part_method=="_global_part":
                self.fc_q = nn.Sequential(
                    nn.Linear(self.n_parts*2048 + 2048, 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(),
                    nn.Linear(2048, 2048),
                )
                self.fc_k = nn.Sequential(
                    nn.Linear(self.n_parts*2048 + 2048, 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(),
                    nn.Linear(2048, 2048),
                )
                if self.with_texts=="sample_level":
                    self.fc_text = nn.Sequential(
                        nn.Linear(self.n_parts*2048 + 2048, 2048),
                        nn.BatchNorm1d(2048),
                        nn.ReLU(),
                        nn.Linear(2048, 512),
                    )
            # 仅仅part
            if self.part_method=="_part":
                self.fc_q = nn.Sequential(
                    nn.Linear(self.n_parts*2048, 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(),
                    nn.Linear(2048, 2048),
                )
                self.fc_k = nn.Sequential(
                    nn.Linear(self.n_parts*2048, 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(),
                    nn.Linear(2048, 2048),
                )
                if self.with_texts=="sample_level":
                    self.fc_text = nn.Sequential(
                        nn.Linear(self.n_parts*2048, 2048),
                        nn.BatchNorm1d(2048),
                        nn.ReLU(),
                        nn.Linear(2048, 512),
                    )
        self.sd_loss = AlignLoss()
        
        # ============= momentum encoder ================
        self.momentum_encoder = copy.deepcopy(self.encoder)
        self.momentum_projector = copy.deepcopy(self.projector)

        for param in chain(self.momentum_encoder.parameters(), 
                           self.momentum_projector.parameters()):
            param.requires_grad = False

        self.temp = args.temperature

    def forward(self, samples):
        if len(samples)>0:
            # 注意要先提出来
            texts = samples[1]
            labels = samples[2]
            samples = samples[0]

        samples = [x.cuda(non_blocking=True) for x in samples]
        texts = torch.cat([clip.tokenize(t, truncate=True) for t in texts])
        if self.with_texts!=None:
            texts = [x.cuda(non_blocking=True) for x in texts]

        h, emb, fp = self.ForwardWrapper(samples, self.encoder, self.projector)
        h = h[0]
        with torch.no_grad():
            self.update_momentum_params(self.momentum)
            h_m, emb_m, fp_m = self.ForwardWrapper(samples[:2], self.momentum_encoder, self.momentum_projector)
            h_m_stand = h_m[1]

        emb_m = [concat_all_gather(x) for x in emb_m]
        h_m = [concat_all_gather(x) for x in h_m]
        h_m = h_m[0]      # 在global对比学习需要weak aug来得到H   TODO 之前跑的结果是standard aug得到的
        assign = self.sinkhorn_knopp(h @ h_m.T)

        # ================part clustering==================
        if self.is_parts=='proto':
            ## query分支
            q_feat = self.layer4_features
            q_feat_global = h
            q_feat_part = self._get_part_feature(q_feat, self.part_proto)
            q_feat_global_part = torch.cat([q_feat_global, q_feat_part], dim=1)
            q_norm = F.normalize(q_feat_global, dim=1)

            ## key分支
            with torch.no_grad():
                k_feat = self.layer4_features_momentum_standard.detach()
                k_feat_global = h_m_standard
                k_feat_part = self._get_part_feature(k_feat, self.part_proto)
                k_feat_global_part = torch.cat([k_feat_global, k_feat_part], dim=1)
                k_norm = F.normalize(k_feat_global, dim=1)
            
            sd_loss = self.sd_loss(q_feat_global_part, k_feat_global_part)

        elif self.is_parts == "pa":
            ## query分支
            q_feat = fp[0]
            q_feat_part = self.net_vlad(q_feat)
            q_feat_part = q_feat_part.view(q_feat_part.size(0), -1)  # flatten
            if self.part_method=="_global_part":
                q_feat_global = F.normalize(h, p=2, dim=1) 
                q_feat_global_part = torch.cat([q_feat_global, q_feat_part], dim=1)   # concat
                q_feat_global_part = F.normalize(q_feat_global_part, p=2, dim=1)  # L2 normalize
                q_feat_proj = self.fc_q(q_feat_global_part)
            elif self.part_method=="_part":
                q_feat_proj = self.fc_q(q_feat_part)
            q_feat_proj = q_feat_proj / q_feat_proj.norm(dim=-1, keepdim=True)
            
            ## key分支
            # with torch.no_grad():
            k_feat = fp_m[1]
            k_feat = concat_all_gather(k_feat)    # 收集负样本
            k_feat_part = self.net_vlad(k_feat)
            k_feat_part = k_feat_part.view(k_feat_part.size(0), -1)  # flatten
            if self.part_method=="_global_part":
                k_feat_global = F.normalize(h_m_stand, p=2, dim=1) 
                k_feat_global = concat_all_gather(k_feat_global)
                k_feat_global_part = torch.cat([k_feat_global, k_feat_part], dim=1)
                k_feat_global_part = F.normalize(k_feat_global_part, p=2, dim=1)  # L2 normalize
                k_feat_proj = self.fc_k(k_feat_global_part)
            elif self.part_method=="_part":
                k_feat_proj = self.fc_k(k_feat_part)
            k_feat_proj = k_feat_proj / k_feat_proj.norm(dim=-1, keepdim=True)

            # sd_loss, _, _ = contrastive_loss_sample(q_feat_proj, k_feat_proj)
            part_sim = q_feat_proj @ k_feat_proj.T / self.temp
            sd_loss = self.cross_entropy(part_sim, assign)

        elif self.is_parts == None:
            sd_loss = torch.tensor(0).cuda()
        else:
            assert("The part clustering is not supported!!!")
            
        # ================text loss==================
        # 图片和自己的描述是正样本对，图片和其他的描述是负样本对
        if self.with_texts=="sample_level":
            if self.part_method=="_global_part":
                q_feat_text = self.fc_text(q_feat_global_part)
            elif self.part_method=="_part":
                q_feat_text = self.fc_text(q_feat_part)
            q_feat_text = q_feat_text / q_feat_text.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                texts_batch = torch.stack(texts, dim=0)
                texts_batch = self.clip_model.encode_text(texts_batch)
                texts_batch = texts_batch / texts_batch.norm(dim=-1, keepdim=True)
                texts_batch = concat_all_gather(texts_batch)
            
            text_loss,_,_ = contrastive_loss_sample(q_feat_text, texts_batch)
        
        elif self.with_texts==None:
            text_loss = torch.tensor(0).cuda()

        else:
            assert("The text fusion is not supported!!!")

        total_loss = 0
        n_loss_terms = 0
        identity_matrix = generate_hard_assign(assign, dist.get_rank())
        for q in range(len(emb)):
            for v in range(len(emb_m)):
                if v == q:
                    continue
                emb_sim = emb[q] @ emb_m[v].T / self.temp
                total_loss += self.cross_entropy(emb_sim, identity_matrix)
                n_loss_terms += 1

        return total_loss / n_loss_terms, sd_loss, text_loss


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

def generate_orthonormal_vectors(n, dim):
    A = torch.randn(dim, n)
    U, S, Vt = torch.svd(A)
    return U.T


@torch.no_grad()
def generate_hard_assign(tensor, device_id):
    """
    Generate a hard assignment matrix of shape (N, 4N), where:
    - Each device's identity matrix is placed in a different block.
    - Other devices' blocks are filled with zeros.
    
    Args:
        tensor: Tensor of shape (N, D) where N is the total number of samples.
        device_id: The device ID for the current machine (used for identifying the current device's samples).
    
    Returns:
        assign: A hard assignment matrix of shape (N, 4N).
    """
    N = tensor.size(0)  # Get the number of samples
    world_size = dist.get_world_size()  # Get the number of devices (world size)

    # 1. Create a full zero matrix of shape (N, 4N)
    assign = torch.zeros(N, 4 * N, device=tensor.device)

    # 2. Calculate the start and end indices for the current device's block
    start_idx = device_id * N
    end_idx = (device_id + 1) * N
    
    # 3. Place the identity matrix in the corresponding block for the current device
    assign[:, start_idx:end_idx] = torch.eye(N, device=tensor.device)
    
    return assign