import os
# optimizer
import torch.optim as optim
# dataloader
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
# utils
from utils.logger import *
from utils.misc import *
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader, Dataset, Sampler,BatchSampler
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
import itertools

class CustomDistributedBatchSamplerDDP(BatchSampler):
    def __init__(self, dataset, sample_ids, batch_size, num_replicas, rank):
        self.dataset = dataset
        self.sample_ids = sample_ids
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank

        self.num_samples = len(self.dataset)
        self.total_size = self.num_samples

        # 计算每个GPU的批次大小
        self.per_gpu_batch_size = self.batch_size // self.num_replicas
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # 获取每个sample_id对应的所有tile的数量
        sample_id_to_tiles = {}
        for idx, sample_id in enumerate(self.sample_ids):
            if sample_id not in sample_id_to_tiles:
                sample_id_to_tiles[sample_id] = []
            sample_id_to_tiles[sample_id].append(idx)
        
        # 根据每个sample_id的tile数量来划分批次
        batches = []
        for sample_id, tiles in sample_id_to_tiles.items():
            num_tiles = len(tiles)
            if num_tiles < self.num_replicas:
                #print("what the hell! there is some slide has no more than 8 tiles!!! ",sample_id)
                continue
            if num_tiles < self.batch_size:
                # 如果该sample_id的tile数小于batch_size，作为一个完整的批次
                batches.append(tiles)
            else:
                # 如果tile数大于batch_size，先那前batch_size个
                batches.append(tiles[0:0 + self.batch_size])
        
        # 现在有了每个批次的tile列表，根据rank和num_replicas划分mini-batch
        for batch in batches:
            # 将批次平均分配给每个GPU
            num_per_gpu = len(batch) // self.num_replicas
            start = self.rank * num_per_gpu
            end = start + num_per_gpu
            per_gpu_batch = batch[start:end]
            yield per_gpu_batch
        print_log(f'Sampler ready', logger=logger)

    def __len__(self):
        # 总的批次数量
        return len(self.sample_ids) // self.batch_size

from torch.utils.data import BatchSampler

class CustomBatchSamplerDP(BatchSampler):
    def __init__(self, dataset, sample_ids, batch_size):
        self.dataset = dataset
        self.sample_ids = sample_ids
        self.batch_size = batch_size

        self.num_samples = len(self.dataset)

    def __iter__(self):
        # 统计 sample_id -> tile 列表
        sample_id_to_tiles = {}
        for idx, sample_id in enumerate(self.sample_ids):
            if sample_id not in sample_id_to_tiles:
                sample_id_to_tiles[sample_id] = []
            sample_id_to_tiles[sample_id].append(idx)
        
        # 生成完整的 batch 列表
        batches = []
        for sample_id, tiles in sample_id_to_tiles.items():
            batches.append(tiles)
            '''
            num_tiles = len(tiles)
            if num_tiles < self.batch_size:
                # 若该 sample_id 的 tile 数量小于 batch_size，作为一个完整的 batch
                batches.append(tiles)
            else:
                # 如果 tile 数量多于 batch_size，按 batch_size 划分
                for i in range(0, num_tiles, self.batch_size):
                    batches.append(tiles[i:i + self.batch_size])
            '''
        # 生成 batch
        for batch in batches:
            yield batch
        #print_log(f'Sampler Finished', logger=logger)

    def __len__(self):
        return len(list(set(self.sample_ids))) #// self.batch_size

def custom_collate_fn(batch):
    rgb_batch = []
    expr_batch = []
    res_batch = []
    label_batch = []
    sample_batch = []
    
    for item in batch:
        rgb, expr, res, other1, other2 = item
        
        rgb_batch.append(rgb)
        #if expr:
        if expr is not None and expr.numel() > 0:#qbw 11-17 change
            if expr.layout == torch.sparse_csr:
                expr = expr.to_sparse_coo()  # 转换为 COO 格式
            if expr.is_sparse:
                expr_batch.append(expr.coalesce())  # 确保稀疏张量被处理为可堆叠的格式
            else:
                expr_batch.append(expr)  # 如果不稀疏或者就没有，也可以直接添加
        else:
            expr_batch.append(expr)
        
        # 处理 res 张量（sparse，直接收集）
        res_batch.append(res)
        
        label_batch.append(other1)
        sample_batch.append(other2)

        #other_batch.append((other1, other2))
    
    # 将 rgb 和 res 张量堆叠成 batch 张量,不用会报错
    rgb_batch = torch.stack(rgb_batch)
    res_batch = torch.stack(res_batch)
    #label_batch = torch.stack(label_batch)
    #sample_batch = torch.stack(sample_batch)

    if expr_batch[0] is not None and expr_batch[0].numel() > 0:
        expr_batch = torch.stack(expr_batch)

    return rgb_batch, expr_batch, res_batch, label_batch,sample_batch

def custom_collate_fn_expr(batch):
    '''
    for pretrain data preprocess
    '''
    rgb_batch = []
    expr_batch = []
    res_batch = []
    label_batch = []
    sample_batch = []
    pos_batch = []

    
    for item in batch:
        expr, res, other1, other2,other3 = item
        #if expr:
        if expr is not None and expr.numel() > 0:#qbw 11-17 change
            if expr.layout == torch.sparse_csr:
                expr = expr.to_sparse_coo()  # trans to COO
            if expr.is_sparse:
                expr_batch.append(expr.coalesce())  # for sparse
            else:
                expr_batch.append(expr)  
        else:
            expr_batch.append(expr)
        res_batch.append(res)
        pos_batch.append(other3)
        label_batch.append(other1)
        sample_batch.append(other2)
    
    res_batch = torch.stack(res_batch)
    label_batch = torch.stack(label_batch)

    if expr_batch[0] is not None and expr_batch[0].numel() > 0:
        expr_batch = torch.stack(expr_batch)

    return expr_batch, res_batch, label_batch,sample_batch,pos_batch



def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    print("datasets ",len(dataset))
    logger = get_logger(args.log_name)
    print("tiles ",len(dataset[0]))
    #sample_ids = [dataset[i][-2] for i in range(len(dataset))]  #获取所有的sampleid 预训练需要注释掉
    shuffle = config.others.subset == 'train'
    if args.distributed:
        if config.others.if_abmil:
            sample_ids = [dataset[i][-2] for i in range(len(dataset))]  #获取所有的sampleid 预训练需要注释掉
            world_size = torch.distributed.get_world_size() 
            rank = torch.distributed.get_rank()
            per_gpu_batch_size = config.others.bs // world_size  
            print("if distributed abmil now!")
            print("config.others.bs",config.others.bs)
            if config.others.if_benchmark:#use embedding for abmil
                print("benchmark sampler start")
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                         num_workers=int(args.num_workers),
                                                         drop_last=config.others.subset == 'train',
                                                         worker_init_fn=worker_init_fn,
                                                         )
            else:            
                sampler = CustomDistributedBatchSamplerDDP(dataset, sample_ids, config.others.bs, num_replicas=world_size, rank=rank)#resample in sampler for DDP
                dataloader = DataLoader(
                        dataset,
                        batch_sampler=sampler,  # 使用修改后的 GrouspedBatchSampler
                        num_workers=args.num_workers
                        #
                    )
        elif config.others.if_embedding:#Use embedding as input
            print("if_embedding now!")
            sampler = None
            print(len(dataset))
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                     drop_last=config.others.subset == 'train',
                                                     num_workers=int(args.num_workers),
                                                     worker_init_fn=worker_init_fn,
                                                     sampler=sampler
                                                     )
            print(len(dataloader))
        else:#classification
            print("pretrain sampler start")
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                     num_workers=int(args.num_workers),
                                                     drop_last=config.others.subset == 'train',
                                                     worker_init_fn=worker_init_fn,
                                                     collate_fn=custom_collate_fn,
                                                     sampler=sampler)
    elif config.others.if_abmil:#abmil
        sample_ids = [dataset[i][-2] for i in range(len(dataset))]  #获取所有的sampleid 预训练需要注释掉
        config.others.bs = 128
        sampler = CustomBatchSamplerDP(dataset, sample_ids, config.others.bs)#resample in sampler for DDP
        print("if_abmil now!")
        print("config.others.bs",config.others.bs)

        if config.others.if_benchmark:#use features
            print("benchmark sampler start no ddp")
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                     num_workers=int(args.num_workers)
                                                     #drop_last=config.others.subset == 'train',
                                                     #worker_init_fn=worker_init_fn,
                                                     )
        else:#full abmil
            print_log(f'Start dataloader for full finetune', logger=logger)
            '''
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                             drop_last=config.others.subset == 'train',
                                                 num_workers=int(args.num_workers),
                                                 worker_init_fn=worker_init_fn,
                                                 collate_fn=custom_collate_fn
                                                 )
            '''
            dataloader = DataLoader(
                        dataset,
                        batch_sampler=sampler,  # 使用修改后的 GrouspedBatchSampler
                        num_workers=args.num_workers
                        #
                    )
            



    elif config.others.if_embedding:#getembedding
        print("if_embedding now!")
        sampler = None
        print(len(dataset))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                 shuffle=shuffle,
                                                 drop_last=config.others.subset == 'train',
                                                 num_workers=int(args.num_workers),
                                                 worker_init_fn=worker_init_fn,
                                                 )
        print(len(dataloader))
    else:#classification
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                 shuffle=shuffle,
                                                 drop_last=config.others.subset == 'train',
                                                 num_workers=int(args.num_workers),
                                                 worker_init_fn=worker_init_fn,
                                                 collate_fn=custom_collate_fn
                                                 )
    return sampler, dataloader


def model_builder(config):
    model = build_model_from_cfg(config)
    return model


def build_opti_sche(base_model, config):
    opti_config = config.optimizer
    freeze = opti_config.freeze
    opti_config.kwargs.lr = float(opti_config.kwargs.lr)
    opti_config.kwargs.weight_decay = float(opti_config.kwargs.weight_decay)
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]

        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=True, **opti_config.kwargs)
    else:
        raise NotImplementedError()

    sche_config = config.scheduler
    if sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=sche_config.kwargs.epochs,
                                      lr_min=float(sche_config.kwargs.lr_min),
                                      warmup_lr_init=float(sche_config.kwargs.warmup_lr_init),
                                      warmup_t=sche_config.kwargs.initial_epochs,
                                      cycle_limit=sche_config.kwargs.cycle_limit,
                                      t_in_epochs=True)
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'function':
        scheduler = None
    else:
        raise NotImplementedError()

    return optimizer, scheduler


def resume_model(base_model, args, logger=None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger=logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger=logger)

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # parameter resume of base model
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt, strict=True)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()
    # print(best_metrics)

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})',
              logger=logger)
    return start_epoch, best_metrics


def resume_optimizer(optimizer, args, logger=None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger=logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger=logger)
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])


def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger=None):
    if args.local_rank == 0:
        torch.save({
            'base_model': base_model.module.state_dict() if args.distributed else base_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics.state_dict() if metrics is not None else dict(),
            'best_metrics': best_metrics.state_dict() if best_metrics is not None else dict(),
        }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger=logger)


def load_model(base_model, ckpt_path, logger=None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger=logger)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')

    incompatible = base_model.load_state_dict(base_ckpt, strict=False)
    if incompatible.missing_keys:
        print_log('missing_keys', logger='Squall')
        print_log(
            get_missing_parameters_message(incompatible.missing_keys),
            logger='Squall'
        )
    if incompatible.unexpected_keys:
        print_log('unexpected_keys', logger='Squall')
        print_log(
            get_unexpected_parameters_message(incompatible.unexpected_keys),
            logger='Squall'
        )

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger=logger)
    return
