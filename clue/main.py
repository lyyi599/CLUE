import os
import time
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms
from src.transform import MuiltiCropDataset
from args import get_args
from method import get_method
import src.vision_transformer as vits
import src.resnet as resnet
from tqdm import tqdm

from src.utils import (
    setup_logging,
    cosine_scheduler,
    build_optimizer,
    load_pretrained_im,
    load_pretrained_clue,
    restart_from_checkpoint,
    init_distributed_device,
    AverageMeter,
)

from eval_retrieval import (
    extract_features,
    retrieval_rank,

)


def random_seed(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def main():
    args = get_args()
    random_seed(args)

    # fully initialize distributed device environment
    device, args = init_distributed_device(args)

    print(f"Rank {args.rank} running on device {args.device}")

    if not os.path.exists(args.dump_path):
        # Create the folder if it doesn't exist
        os.makedirs(args.dump_path)
    
    setup_logging(os.path.join(args.dump_path,'out.log'), logging.INFO)

    if args.local_rank != 0:
        def log_pass(*args): pass
        logging.info = log_pass

    # build data
    traindir = os.path.join(args.data_path, 'train')
    
    train_dataset = MuiltiCropDataset(
        traindir,
        args,
        return_index=False,
        json_path=args.text_path,
        qa_idx=args.qa_idx
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logging.info(f"Building data done with {len(train_dataset)} images loaded.")

    # ================== retrieval setting ==================
    # ============ preparing data ... ============
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)
    dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "test"), transform=transform)

    sampler_val_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)  # val过程中的train set
    sampler_val_test = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)  # val过程中的test set
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_val_train,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val_test,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    logging.info(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")
    best_rank1 = best_rank5 = best_rank10 = 0.0
    # ================= retrieval setting==================

    # build model
    model = get_method(args)

    if args.ckpt_from_impre:
        load_pretrained_im(model, args.ckpt_from_impre)
    
    # synchronize batch norm layers
    if "vit" not in args.arch:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # copy model to GPU
    torch.cuda.set_device(device)
    model.cuda(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    logging.info(model)
    logging.info("Building model done.")

    # build optimizer
    args.lr = args.lr * args.batch_size * args.world_size / 256

    optimizer = build_optimizer(model.parameters(), args)

    # ============ init schedulers ... ============
    args.lr_schedule = cosine_scheduler(
        args.lr,
        args.final_lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    args.momentum_schedule = cosine_scheduler(
            args.momentum, 1,
            args.epochs, len(train_loader)
    )

    logging.info(f"Building {args.optimizer} optimizer done.")

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_rank1":0.0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
    )
    start_epoch = to_restore["epoch"]
    best_rank1 = to_restore["best_rank1"]

    cudnn.benchmark = True

    # scaler = torch.cuda.amp.GradScaler()
    scaler = torch.amp.GradScaler('cuda')

    # ==============val process=============
    global rank1_temp, rank5_temp, rank10_temp
    rank1_temp = rank5_temp = rank10_temp = 0.0

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logging.info(f"============ Starting epoch {epoch} ... ============")

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        loss = train(train_loader, model, scaler, optimizer, epoch, args)
        
        # retrieval network
        if args.arch.startswith('vit'):
            eval_model, _ = vits.__dict__[args.arch](patch_size=args.patch_size)
        else:
            eval_model, _ = resnet.__dict__[args.arch]()
        if args.local_rank == 0:
            # remove the DDP wrapper
            eval_model = unwrap_model(eval_model)
            state_dict = model.state_dict()
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.encoder'):
                    new_key = k[len("module.encoder."):]
                    filtered_state_dict[new_key] = v
                elif not (k.startswith('module.momentum_encoder') 
                        or k.startswith('module.momentum_projector')
                        or k.startswith('module.projector')
                        or k.startswith('module.predictor')
                        or k.startswith('module.part_proto')
                        or k.startswith('module.net_vlad')
                        or k.startswith('module.clip')
                        ):
                    filtered_state_dict[k] = v

            temp_path = os.path.join(args.dump_path, "temp_checkpoint.pth")
            torch.save(filtered_state_dict, temp_path)
            
        torch.distributed.barrier()

        temp_path = os.path.join(args.dump_path, "temp_checkpoint.pth")
        checkpoint = torch.load(temp_path, map_location='cpu')
        msg = eval_model.load_state_dict(checkpoint, strict=False)
        # logging.info(f"Load checkpoint message: {msg}")
        logging.info(f"Load checkpoint message: logger记得加回来")
        # time.sleep(1000)
        eval_model.cuda(device)
        eval_model = nn.parallel.DistributedDataParallel(eval_model, device_ids=[device])
        cudnn.benchmark = True
        
        val_retrieval(args, eval_model, dataset_train, dataset_val, data_loader_train, data_loader_val)
        
        # save checkpoints
        if args.local_rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )

            if rank1_temp > best_rank1:
                logging.info("Best rank1 found. Saving the model...")
                best_rank1 = rank1_temp
                torch.save(
                    save_dict,
                    os.path.join(args.dump_path, "best_checkpoint.pth.tar"),
                )
        del eval_model
        torch.cuda.empty_cache()

    logging.info("Training done. Saving the final model ...")
        
def train(loader, model, scaler, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_global = AverageMeter()
    losses_part = AverageMeter()
    losses_text = AverageMeter()
    model.train()

    end = time.time()
    for it, samples in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update parameters
        iters = len(loader) * epoch + it  # global training iteration
        adjust_parameters(model, optimizer, args, iters)

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        # with torch.cuda.amp.autocast():
        with torch.amp.autocast('cuda'):
            loss_global, loss_part, loss_text = model(samples)
            loss = loss_global
            if args.is_parts!=None:
                loss = loss + loss_part
            if args.with_texts!=None:
                loss = loss + loss_text
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ============ misc ... ============
        samples = samples[0]
        losses.update(loss.item(), samples[0].size(0))
        losses_global.update(loss_global.item(), samples[0].size(0))
        losses_part.update(loss_part.item(), samples[0].size(0))
        losses_text.update(loss_text.item(), samples[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.local_rank == 0 and it % 5 == 0:
            logging.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                "Data {data_time.val:.4f} ({data_time.avg:.4f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
            logging.info(
                "Epoch: [{0}][{1}]\t"
                "Loss_assign {loss_assign.val:.4f} ({loss_assign.avg:.4f})\t"
                "Loss_part {loss_part.val:.4f} ({loss_part.avg:.4f})\t"
                "Loss_text {loss_text.val:.4f} ({loss_text.avg:.4f})".format(
                    epoch,
                    it,
                    loss_assign=losses_global,
                    loss_part=losses_part,
                    loss_text=losses_text
                )
            )
    return losses.avg


def val_retrieval(args, model, dataset_train, dataset_val, data_loader_train, data_loader_val):

    model.eval()

    global rank1_temp, rank5_temp, rank10_temp  # 添加全局声明
    with torch.no_grad():
        train_features = extract_features(model, data_loader_train, args)
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

        # ============ retrieval ... ============
        rank1_temp, rank5_temp, rank10_temp = retrieval_rank(train_features, train_labels, 
                                      test_features, test_labels)
        
    rank1_tensor = torch.tensor(rank1_temp).cuda(args.device)
    rank5_tensor = torch.tensor(rank5_temp).cuda(args.device)
    rank10_tensor = torch.tensor(rank10_temp).cuda(args.device)

    torch.distributed.broadcast(rank1_tensor, src=0)
    torch.distributed.broadcast(rank5_tensor, src=0)
    torch.distributed.broadcast(rank10_tensor, src=0)

    rank1_temp = rank1_tensor.item()
    rank5_temp = rank5_tensor.item()
    rank10_temp = rank10_tensor.item()


def adjust_parameters(model, optimizer, args, iters):
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr_schedule[iters]

    unwrap_model(model).momentum = args.momentum_schedule[iters]

class ReturnIndexDataset_withText(datasets.ImageFolder):
    def __init__(self, json_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 加载JSON文本数据
        with open(json_path, 'r') as f:
            self.text_data = json.load(f)
        # 获取数据集根目录的Path对象
        self.root_path = Path(self.root).resolve()  # 确保路径标准化
        print("===========!!!!!!!!!!!!!+++++++++++++++++")

    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset_withText, self).__getitem__(idx)

        img_path = self.loader(idx)
        assert 1==0
        print(image_path)
        time.sleep(1000)
        # 获取当前图片的绝对路径
        img_abs_path = Path(self.samples[idx][0]).resolve()
        # 计算相对于根目录的相对路径
        relative_path = img_abs_path.relative_to(self.root_path)
        # 转换为字符串用作键（例如：'class1/image1.jpg'）
        key = str(relative_path)
        # 获取对应文本
        text = self.text_data[key]
        return img, idx, text

class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx

if __name__ == "__main__":
    main()
