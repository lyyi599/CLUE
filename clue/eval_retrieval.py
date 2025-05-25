"""
Mostly copy-paste from DINO.
https://github.com/facebookresearch/dino/blob/main/eval_knn.py
"""
import os
import logging
import torch
from torch import nn
import random
import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms
from args import get_args
import src.resnet as resnet
import src.vision_transformer as vits
from tqdm import tqdm
from src.utils import (
    setup_logging,
    load_pretrained_im,
    load_pretrained_clue,
    init_distributed_device,
)


def random_seed(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

def main():
    args = get_args()
    random_seed(args)
    # fully initialize distributed device environment
    device, args = init_distributed_device(args)

    if not os.path.exists(args.dump_path):
        # Create the folder if it doesn't exist
        os.makedirs(args.dump_path)

    setup_logging(os.path.join(args.dump_path,'out.log'), logging.INFO)
    if args.local_rank != 0:
        def log_pass(*args): pass
        logging.info = log_pass

    # ============ preparing data ... ============
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)
    dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "test"), transform=transform)

    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    logging.info(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # build model
    logging.info(f"creating model '{args.arch}'")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch.startswith('vit'):
        model, _ = vits.__dict__[args.arch](patch_size=args.patch_size)
    else:
        model, _ = resnet.__dict__[args.arch]()

    load_pretrained_clue(model, args.pretrained)
    if args.ckpt_from_impre:
        load_pretrained_im(model, args.ckpt_from_impre)

    # copy model to GPU
    torch.cuda.set_device(device)
    model.cuda(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    model.eval()
    cudnn.benchmark = True

    # ============ extract features ... ============
    logging.info("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args)
    logging.info("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val, args)

    if args.rank == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()

    if args.rank == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()

        logging.info("Features are ready!\nStart the retrieval.")

        # ============ retrieval ... ============
        rank1, rank5, rank10 = retrieval_rank(train_features, train_labels, 
                                      test_features, test_labels)
        logging.info(f"Rank@1: {rank1:.1f}, Rank@5: {rank5:.1f}, Rank@10: {rank10:.1f}")
    dist.barrier()
    dist.destroy_process_group()
    

@torch.no_grad()
def extract_features(model, data_loader, args):
    features = None
    for i, (samples, index) in enumerate(tqdm(data_loader, desc="extracting features", unit="batch")):

        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        feats_gap, fp = model(samples)
        feats = feats_gap.clone()

        # init storage feature matrix
        if args.rank == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if args.use_cuda:
                features = features.cuda(non_blocking=True)
            logging.info(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if args.rank == 0:
            if args.use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


@torch.no_grad()
def retrieval_rank(train_features, train_labels, test_features, test_labels):
    sim = torch.mm(test_features, train_features.t())
    # 需要完成的代码
    # 使用相似度完成rank1, rank5, rank10的计算，返回rank1, rank5, rank10
    # 计算相似度矩阵
    sim = torch.mm(test_features, train_features.t())  # (N_test, N_train)
    
    # 排序：对于每个查询，找到与之最相似的训练图像
    _, sorted_indices = torch.topk(sim, k=10, dim=1, largest=True, sorted=True)  # (N_test, 10)
    
    # 计算 Rank@k
    rank1 = 0
    rank5 = 0
    rank10 = 0
    
    # 遍历每个查询，检查排名前k的训练图像是否包含正确的标签
    for i in range(sorted_indices.size(0)):
        # 获取查询图像的真实标签
        query_label = test_labels[i]
        
        # 获取排序后的前10个训练图像的标签
        top_k_labels = train_labels[sorted_indices[i]]
        
        # 检查 Rank@1
        if query_label == top_k_labels[0]:
            rank1 += 1
        
        # 检查 Rank@5
        if query_label in top_k_labels[:5]:
            rank5 += 1
        
        # 检查 Rank@10
        if query_label in top_k_labels:
            rank10 += 1
    
    # 计算准确率
    rank1 = rank1 / len(test_labels) * 100  # 转为百分比
    rank5 = rank5 / len(test_labels) * 100
    rank10 = rank10 / len(test_labels) * 100
    
    return rank1, rank5, rank10


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx

if __name__ == "__main__":
    main()
