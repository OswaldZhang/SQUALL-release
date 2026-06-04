import torch
import pickle
import numpy as np
import os
import torch.nn as nn
from models import builder
from utils import dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.distributed as dist
from torch.autograd import Function
import itertools
from torch.distributed.nn import all_gather
import random
import json
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from torch.cuda.amp import autocast

class AccMetric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'AccMetric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


class AllGatherWithGrad(Function):
    @staticmethod
    def forward(ctx, input):
        world_size = dist.get_world_size()
        gathered = [torch.empty_like(input) for _ in range(world_size)]
        dist.all_gather(gathered, input)
        ctx.save_for_backward(input)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        world_size = dist.get_world_size()
        
        # 获取当前进程的梯度分块
        grad_input = grad_output.chunk(world_size)[dist.get_rank()]
        
        # 对所有进程的梯度进行 all_reduce 平均
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM)
        grad_input /= world_size
        
        return grad_input

def run_net_freeze(args, config, train_writer=None, val_writer=None):
    def print_gpu_memory():
        if torch.cuda.is_available():
            print(f"当前GPU占用内存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
            #print(f"模型计算图保留的内存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
    print("start")
    print_gpu_memory()
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    # build model
    print("DataLoader")
    print_gpu_memory()
    base_model = builder.model_builder(config.model)
    if config.fwd_type != 'benchmark':
        base_model.load_model_from_ckpt(args.ckpts)
    # parameter setting
    start_epoch = 0
    best_metrics = AccMetric(0.)
    metrics = AccMetric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = AccMetric(best_metrics)

    if args.use_gpu:
        base_model.to(args.local_rank)
    print("build model")
    print_gpu_memory()
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)
    print("build optimizer")
    print_gpu_memory()

    # trainval
    # training
    #base_model._set_static_graph() 
    base_model.zero_grad()
    print("lr start:",optimizer.param_groups[0]['lr'])
    local_attention_map = dict()
    '''
    for idx, (rgb, res, label, sample_id, pos) in enumerate(train_dataloader):
        if "0" in str(base_model.device):
            print(list(set(label))[0],list(set(sample_id))[0],rgb.shape)
    '''
    
    # Step 2: 仅第一次获取特征
    all_features = []
    all_labels = []
    all_sample_id = []
    all_pos = []
    all_features_expr_test = []
    all_labels_expr_test = []
    all_sample_id_expr_test = []
    all_pos_expr_test = []
    n_batches = len(train_dataloader)
    for idx, (rgb, res, label, sample_id, pos) in enumerate(train_dataloader):
        torch.autograd.set_detect_anomaly(True)

        if args.use_gpu:
            rgb, res, label = rgb.to(args.local_rank), res.to(args.local_rank), label.to(args.local_rank)

        # 转换数据类型
        rgb = rgb  # .half() 如果需要支持混合精度训练，可以解注释
        res = res  # .half()
        label = label.float()  # .half()

        # 提取预训练特征
        with torch.no_grad():  # 提取特征不需要计算梯度
            if config.fwd_type == 'expr':
                #print("expression extracted")
                gathered_features = base_model.module.forward_get_embedding_expr(rgb, res)
            elif config.fwd_type == 'benchmark':
                gathered_features =rgb.squeeze(0)
                #print("gathered_features.shape",gathered_features.shape)
            else:
                gathered_features = base_model.module.forward_get_embedding(rgb, res)
        #print("sample ",sample_id,"gathered_features",gathered_features.shape)
        # 将特征和标签存储起来
        #if "0" in str(base_model.device):
        all_features.append(gathered_features)
        all_labels.append(label)
        all_sample_id.append(sample_id[0])
        all_pos.append(pos)
    #if config.fwd_type == 'expr' or config.fwd_type == 'benchmark':
    for idx, (rgb, res, label, sample_id, pos) in enumerate(test_dataloader):
        torch.autograd.set_detect_anomaly(True)

        if args.use_gpu:
            rgb, res, label = rgb.to(args.local_rank), res.to(args.local_rank), label.to(args.local_rank)

        # 转换数据类型
        rgb = rgb  # .half() 如果需要支持混合精度训练，可以解注释
        res = res  # .half()
        label = label.float()  # .half()

        # 提取预训练特征
        with torch.no_grad():  # 提取特征不需要计算梯度
            if config.fwd_type == 'expr':
                #print("expression extracted")
                gathered_features = base_model.module.forward_get_embedding_expr(rgb, res)
            elif config.fwd_type == 'benchmark':
                gathered_features =rgb.squeeze(0)
                #print("gathered_features.shape",gathered_features.shape)
            else:
                gathered_features = base_model.module.forward_get_embedding(rgb, res)
        all_features_expr_test.append(gathered_features)
        all_labels_expr_test.append(label)
        all_sample_id_expr_test.append(sample_id[0])
        all_pos_expr_test.append(pos)
    # 将特征和标签拼接为单个张量
    #all_features = torch.cat(all_features, dim=0)
    #all_labels = torch.cat(all_labels, dim=0)
    # Step 3: 训练过程
    batch_size = 1
    accumulation_steps = config.accumulation_steps
    #if "0" in str(base_model.device):

    for epoch in range(start_epoch, config.max_epoch + 1):#:
        batch_ret = []
        batch_times = []
        batch_status = []
        batch_all_ret = []
        batch_all_times = []
        batch_all_status = []
        num_iter = 0
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        print("len(all_features)",len(all_features))
        for idx in range(len(all_features)):#// batch_size 
            num_iter+=1
            if config.abmil_type == "Survival":#dirty
                batch_features = all_features[idx]#all_features[idx * batch_size: (idx + 1) * batch_size]
                batch_labels = all_labels[idx]
                batch_features =batch_features
                times = batch_labels[:,0]
                status = batch_labels[:,1]
                if args.use_gpu:
                    rgb,res, times,status = rgb.to(args.local_rank),res.to(args.local_rank), times.to(args.local_rank),status.to(args.local_rank)
                rgb = rgb.float()
                res = res.float()
                times = times.float()
                status = status.float()
                ret, attn_scores = base_model.module.forward_from_embedding(batch_features)
                batch_ret.append(ret)
                batch_times.append(times[0])
                batch_status.append(status[0])
                batch_all_ret.append(ret)
                batch_all_times.append(times[0])
                batch_all_status.append(status[0])
            else:
                batch_features = all_features[idx]#all_features[idx * batch_size: (idx + 1) * batch_size]
                batch_labels = all_labels[idx]#all_labels[idx * batch_size: (idx + 1) * batch_size]
                #print("batch_features.shape",batch_features.shape)
                #print("batch_labels.shape",batch_labels.shape)
                '''
                if args.use_gpu:
                    batch_features = batch_features.to(args.local_rank)
                    batch_labels = batch_labels.to(args.local_rank)
                '''

                # Step 3.2: 训练模型
                ret, attn_scores = base_model.module.forward_from_embedding(batch_features)
                #ret, attn_scores = base_model.module.forward_from_embedding_qzk(batch_features)
                #print("attn_scores.shape",attn_scores.shape)
                attn_scores = attn_scores.squeeze(1)  # Shape: (batch_size,)
                #print("attn_scores.shape",attn_scores.shape)

                # 记录注意力分数
                '''
                for i in range(len(all_pos[idx])):
                    pos_str = all_pos[idx][i]
                    attn_value = attn_scores[i].item()
                    local_attention_map[pos_str] = attn_value
                '''
                # 计算损失和准确率
                loss, acc = base_model.module.get_loss_acc(ret, batch_labels[0])

                # 梯度累积处理
                loss = loss / accumulation_steps
                with torch.autograd.detect_anomaly():
                    loss.backward()
                if args.distributed:
                    if "0" in str(base_model.device) and idx%100 == 0:
                        #print("ret:",ret)
                        #print("label:",label[0])
                        print("loss:",loss,"acc:",acc,"lr:",optimizer.param_groups[0]['lr'],"sample_id ",all_sample_id[idx],"size",batch_features.shape," epoch ",epoch)
                elif idx%100 == 0:
                    print("loss:",loss,"acc:",acc,"lr:",optimizer.param_groups[0]['lr'],"sample_id ",all_sample_id[idx],"size",batch_features.shape," epoch ",epoch)
                batch_ret.append(ret)
                #print("ret",ret)
                #print("label ",label.shape)
                batch_status.append(label[0])
            if (idx + 1) % accumulation_steps == 0:  # 每 accumulation_steps 个 batch 更新一次权重
                if config.abmil_type == "Survival":
                    batch_ret = torch.stack(batch_ret, dim=0)  # shape: (batch_size, 4)
                    batch_times = torch.stack(batch_times, dim=0)  # shape: (batch_size,)
                    batch_status = torch.stack(batch_status, dim=0)  # shape: (batch_size,)
                    #print("batch_ret",batch_ret.shape)
                    #print("batch_times",batch_times.shape)
                    print("batch_status",batch_status)
                    loss, acc = base_model.module.get_loss_acc(batch_ret, (batch_times,batch_status))
                    if config.get('grad_norm_clip') is not None:
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                    with torch.autograd.detect_anomaly():
                        loss.backward()
                    for name, param in base_model.named_parameters():
                        if param.grad is not None:
                            print(f"{name} - grad mean: {param.grad.abs().mean().item()}")
                    optimizer.step()  # 更新权重
                    base_model.zero_grad()  # 清零梯度
                    batch_ret = []
                    batch_times = []
                    batch_status = []
                else:
                    if config.get('grad_norm_clip') is not None:
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                    '''
                    for name, param in base_model.named_parameters():
                        if param.grad is not None:
                            print(f"{name} - grad mean: {param.grad.abs().mean().item()}")
                    '''
                    optimizer.step()
                    base_model.zero_grad()
            ''''''
        if config.abmil_type != "Survival":
            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
            if args.distributed:
                torch.cuda.synchronize()
            

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
        if num_iter % accumulation_steps != 0:
            if config.abmil_type == "Survival":
                    batch_ret = torch.stack(batch_ret, dim=0)  # shape: (batch_size, 4)
                    batch_times = torch.stack(batch_times, dim=0)  # shape: (batch_size,)
                    batch_status = torch.stack(batch_status, dim=0)  # shape: (batch_size,)
                    #print("batch_ret",batch_ret.shape)
                    #print("batch_times",batch_times.shape)
                    #print("batch_status",batch_status)
                    loss, acc = base_model.module.get_loss_acc(batch_ret, (batch_times,batch_status))
                    if config.get('grad_norm_clip') is not None:
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                    loss.backward()
                    if args.distributed:
                        loss = dist_utils.reduce_tensor(loss, args)
                        torch.cuda.synchronize()
                    optimizer.step()  # 更新权重
                    base_model.zero_grad()  # 清零梯度
            else:
                optimizer.step()
                base_model.zero_grad()
        if config.abmil_type == "Survival":#dirty
            batch_ret = torch.stack(batch_all_ret, dim=0)  # shape: (batch_size, 4)
            batch_times = torch.stack(batch_all_times, dim=0)  # shape: (batch_size,)
            batch_status = torch.stack(batch_all_status, dim=0)  # shape: (batch_size,)
            #print("batch_ret",batch_ret.shape)
            #print("batch_times",batch_times.shape)
            #print("batch_status",batch_status.shape)

            loss, acc = base_model.module.get_loss_acc(batch_ret, (batch_times,batch_status))
            print("loss:",loss,"Cindex:",acc)
            print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %s lr = %.6f' %
                              (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                               ['%.4f' % l for l in (loss,acc[0])], optimizer.param_groups[0]['lr']), logger=logger)
        elif config.abmil_type == "classifier":
            batch_ret = torch.stack(batch_ret, dim=0)  # shape: (batch_size, 4)
            batch_status = torch.stack(batch_status, dim=0)  # shape: (batch_size,)
            '''
            print("batch_ret",batch_ret.shape)
            print("batch_status",batch_status.shape)
            '''

            loss, acc = base_model.module.get_loss_acc(batch_ret, batch_status)
            print("loss:",loss,"acc:",acc)
            print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %s lr = %.6f' %
                              (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                               ['%.4f' % l for l in (loss,acc[0][0],acc[0][1],acc[0][2],acc[1],acc[2])], optimizer.param_groups[0]['lr']), logger=logger)
        else:
            batch_ret = torch.stack(batch_ret, dim=0)  # shape: (batch_size, 4)
            batch_status = torch.stack(batch_status, dim=0)  # shape: (batch_size,)
            '''
            print("batch_ret",batch_ret.shape)
            print("batch_status",batch_status.shape)
            '''

            loss, acc = base_model.module.get_loss_acc(batch_ret, batch_status)
            print("loss:",loss,"acc:",acc)
            print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %s lr = %.6f' %
                              (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                               ['%.4f' % l for l in (loss,acc[0],acc[1],acc[2])], optimizer.param_groups[0]['lr']), logger=logger)


        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss', loss, epoch)
        if config.abmil_type == "classifier":
            print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                      (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in (loss,acc[0][0],acc[0][1],acc[0][2],acc[1],acc[2])],
                       optimizer.param_groups[0]['lr']), logger=logger)
        else:
            print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                      (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in (loss,acc[0],acc[1],acc[2])],
                       optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            if config.abmil_type == "Survival": #dirty
                #metrics = validate_survival(base_model, test_dataloader, epoch, val_writer, args, config, best_metrics, logger=logger)
                metrics = validate_survival_freeze(base_model, all_features_expr_test,all_labels_expr_test,all_sample_id_expr_test,all_pos_expr_test, epoch, val_writer, args, config, best_metrics, logger=logger)
                better = metrics.better_than(best_metrics)
                # Save ckeckpoints
                if better:
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)
                    print_log(
                        "--------------------------------------------------------------------------------------------",
                        logger=logger)
            elif config.abmil_type == "classifier":
                if config.fwd_type == 'expr' or config.fwd_type == 'benchmark':
                    metrics = validate_classifier_expr(base_model, all_features_expr_test,all_labels_expr_test,all_sample_id_expr_test,all_pos_expr_test, epoch, val_writer, args, config, best_metrics, logger=logger)

                else:
                    metrics = validate_classifier_freeze(base_model, test_dataloader, epoch, val_writer, args, config, best_metrics, logger=logger)

                better = metrics.better_than(best_metrics)
                # Save ckeckpoints
                if better:
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)
                    print_log(
                        "--------------------------------------------------------------------------------------------",
                        logger=logger)
            else:
                metrics = validate_regression_freeze(base_model, all_features_expr_test,all_labels_expr_test,all_sample_id_expr_test,all_pos_expr_test, epoch, val_writer, args, config, best_metrics, logger=logger)

                better = metrics.better_than(best_metrics)
                # Save ckeckpoints
                if better:
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)
                    print_log(
                        "--------------------------------------------------------------------------------------------",
                        logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
    
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()



def run_net(args, config, train_writer=None, val_writer=None):
    def print_gpu_memory():
        if torch.cuda.is_available():
            print(f"当前GPU占用内存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
            #print(f"模型计算图保留的内存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
    print("start")
    print_gpu_memory()
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    # build model
    print("DataLoader")
    print_gpu_memory()
    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt(args.ckpts)
    # parameter setting
    start_epoch = 0
    best_metrics = AccMetric(0.)
    metrics = AccMetric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = AccMetric(best_metrics)

    if args.use_gpu:
        base_model.to(args.local_rank)
    print("build model")
    print_gpu_memory()
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)
    print("build optimizer")
    print_gpu_memory()

    # trainval
    # training
    #base_model._set_static_graph() 
    base_model.zero_grad()
    print("lr start:",optimizer.param_groups[0]['lr'])
    local_attention_map = dict()
    n_batches = len(train_dataloader)

    for epoch in range(start_epoch, config.max_epoch + 1):#
        #for epoch in range(1):
        print("epoch ",epoch)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_loader_time = AverageMeter()
        data_time = AverageMeter()
        run_time = AverageMeter()
        model_time = AverageMeter()
        synchronize_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        batch_ret = []
        batch_times = []
        batch_status = []
        tag_stop = 0
        #print("abmil_type:",config.abmil_type)
        accumulation_steps = config.accumulation_steps
        for idx,(rgb,res, label, sample_id,pos) in enumerate(train_dataloader):
            data_loader_time.update(time.time() - batch_start_time)
            sample_id = list(set(sample_id))[0]
            local_attention_map = {}
            #print("Sampleid ",list(set(sample_id)),epoch,str(base_model.device))
            tag_stop+=1
            num_iter += 1
            n_itr = epoch * n_batches + idx
            data_time.update(time.time() - batch_start_time)
            run_start_time = time.time()
            if config.abmil_type == "Survival":#dirty
                times = label[:,0]
                status = label[:,1]
                if args.use_gpu:
                    rgb,res, times,status = rgb.to(args.local_rank).float(),res.to(args.local_rank).float(), times.to(args.local_rank).float(),status.to(args.local_rank).float()
                #print("rgb ",rgb.shape,"time",time,"status",status,sample_id,pos)
                rgb = rgb.to(torch.bfloat16)
                res = res.to(torch.bfloat16)
                base_model = base_model.to(torch.bfloat16)

                with autocast(dtype=torch.bfloat16):
                    ret,attn_scores = base_model(rgb,res)
                batch_ret.append(ret)
                batch_times.append(times[0])
                batch_status.append(status[0])
                loss, acc = base_model.module.get_loss_acc(ret, (times[0],status[0]))
                loss = loss / accumulation_steps
                loss.backward()
                if args.distributed:
                    if "0" in str(base_model.device) and idx%100 == 0:
                        print("loss:",loss,"acc:",acc,"lr:",optimizer.param_groups[0]['lr'],"sample_id ",sample_id,"size",rgb.shape," epoch ",epoch)
                else:
                    if idx%100 == 0:
                        print("loss:",loss,"acc:",acc,"lr:",optimizer.param_groups[0]['lr'],"sample_id ",sample_id,"size",rgb.shape," epoch ",epoch)
                        print_gpu_memory()

            else:
                torch.autograd.set_detect_anomaly(True)
                if args.use_gpu:
                    rgb,res, label = rgb.to(args.local_rank),res.to(args.local_rank), label.to(args.local_rank)
                num_samples = rgb.size(0)
                run_time.update(time.time() - run_start_time)
                #bag_feature = base_model.module.forward_ddp(rgb,res)
                model_start_time = time.time()
                ret, attn_scores = base_model(rgb,res)
                attn_scores = attn_scores.squeeze(1)  # Shape: (batch_size,)
                #torch.cuda.synchronize()
                
                '''
                for i in range(len(pos)):
                    pos_str = pos[i]
                    attn_value = attn_scores[i].item()
                    local_attention_map[pos_str] = attn_value
                '''
                batch_ret.append(ret)
                #print("label ",label.shape)
                batch_status.append(label[0])
                loss, acc = base_model.module.get_loss_acc(ret, label[0])
                loss = loss / accumulation_steps
                loss.backward()  # Backpropagate gradients for this batch
                model_time.update(time.time() - model_start_time)
                #print(str(base_model.device),"pos:",pos)
                if args.distributed:
                    if "0" in str(base_model.device) and idx%100 == 0:
                        print("loss:",loss,"acc:",acc,"lr:",optimizer.param_groups[0]['lr'],"sample_id ",sample_id,"size",rgb.shape," epoch ",epoch)
                else:
                    if idx%20 == 0:
                        print("loss:",loss,"acc:",acc,"lr:",optimizer.param_groups[0]['lr'],"sample_id ",sample_id,"size",rgb.shape," epoch ",epoch)
                        print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) LoadTime = %.3f (s) ModelTime = %.3f (s)' %(epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),run_time.val(),model_time.val()), logger=logger)
            # forward
            '''
            for name, param in base_model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"Gradient contains NaN or Inf in layer {name}")
            '''
            if (idx + 1) % accumulation_steps == 0:  # 每 accumulation_steps 个 batch 更新一次权重
                if config.abmil_type == "Survival":
                    '''
                    batch_ret = torch.stack(batch_ret, dim=0)  # shape: (batch_size, 4)
                    batch_times = torch.stack(batch_times, dim=0)  # shape: (batch_size,)
                    batch_status = torch.stack(batch_status, dim=0)  # shape: (batch_size,)
                    '''
                    
                    #loss, acc = base_model.module.get_loss_acc(batch_ret, (batch_times,batch_status))
                    if config.get('grad_norm_clip') is not None:
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                    #loss.backward()
                    optimizer.step()  # 更新权重
                    base_model.zero_grad()  # 清零梯度
                    '''
                    batch_ret = []
                    batch_times = []
                    batch_status = []
                    '''

                else:
                    if config.get('grad_norm_clip') is not None:
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                    '''
                    for name, param in base_model.named_parameters():
                        if param.grad is not None:
                            print(f"{name} - grad mean: {param.grad.abs().mean().item()}")
                    '''
                    
                    optimizer.step()  # 更新权重
                    base_model.zero_grad()  # 清零梯度

            ''''''
            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                torch.cuda.synchronize()
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            '''
            with open(f'attention_map/DR_TCGA/attention_map_rank_{sample_id}_rank_{args.local_rank}.json', 'w') as f:
                    json.dump(local_attention_map, f)
            '''
        if num_iter % accumulation_steps != 0:
            optimizer.step()
            base_model.zero_grad()

            


        batch_times = [t.unsqueeze(0) if t.dim() == 0 else t for t in batch_times]
        batch_status = [s.unsqueeze(0) if s.dim() == 0 else s for s in batch_status]
        if config.abmil_type == "Survival":#dirty
            batch_ret = torch.stack(batch_ret, dim=0)  # shape: (batch_size, 4)
            batch_times = torch.stack(batch_times, dim=0)  # shape: (batch_size,)
            batch_status = torch.stack(batch_status, dim=0)  # shape: (batch_size,)

            loss, acc = base_model.module.get_loss_acc(batch_ret, (batch_times,batch_status))
            print("loss:",loss,"Cindex:",acc)
            print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %s lr = %.6f' %
                              (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                               ['%.4f' % l for l in (loss,acc[0])], optimizer.param_groups[0]['lr']), logger=logger)
        elif config.abmil_type == "classifier":
            batch_ret = torch.stack(batch_ret, dim=0)  # shape: (batch_size, 4)
            batch_status = torch.stack(batch_status, dim=0)  # shape: (batch_size,)
            '''
            print("batch_ret",batch_ret.shape)
            print("batch_status",batch_status.shape)
            '''

            loss, acc = base_model.module.get_loss_acc(batch_ret, batch_status)
            print("loss:",loss,"acc:",acc)
            print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %s lr = %.6f' %
                              (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                               ['%.4f' % l for l in (loss,acc[0][0],acc[0][1],acc[0][2],acc[1],acc[2])], optimizer.param_groups[0]['lr']), logger=logger)
        else:
            batch_ret = torch.stack(batch_ret, dim=0)  # shape: (batch_size, 4)
            batch_status = torch.stack(batch_status, dim=0)  # shape: (batch_size,)
            '''
            print("batch_ret",batch_ret.shape)
            print("batch_status",batch_status.shape)
            '''

            loss, acc = base_model.module.get_loss_acc(batch_ret, batch_status)
            print("loss:",loss,"acc:",acc)
            print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %s lr = %.6f' %
                              (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                               ['%.4f' % l for l in (loss,acc[0],acc[1],acc[2])], optimizer.param_groups[0]['lr']), logger=logger)


        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss', loss, epoch)
        if config.abmil_type == "classifier":
            print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                      (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in (loss,acc[0][0],acc[0][1],acc[0][2],acc[1],acc[2])],
                       optimizer.param_groups[0]['lr']), logger=logger)
        else:
            print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                      (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in (loss,acc[0],acc[1],acc[2])],
                       optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            if config.abmil_type == "Survival": #dirty
                metrics = validate_survival(base_model, test_dataloader, epoch, val_writer, args, config, best_metrics, logger=logger)

                better = metrics.better_than(best_metrics)
                # Save ckeckpoints
                if better:
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)
                    print_log(
                        "--------------------------------------------------------------------------------------------",
                        logger=logger)
            elif config.abmil_type == "classifier":
                metrics = validate_classifier(base_model, test_dataloader, epoch, val_writer, args, config, best_metrics, logger=logger)

                better = metrics.better_than(best_metrics)
                # Save ckeckpoints
                if better:
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)
                    print_log(
                        "--------------------------------------------------------------------------------------------",
                        logger=logger)
            else:
                metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, best_metrics, logger=logger)

                better = metrics.better_than(best_metrics)
                # Save ckeckpoints
                if better:
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)
                    print_log(
                        "--------------------------------------------------------------------------------------------",
                        logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
    
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate_survival(base_model, test_dataloader, epoch, val_writer, args, config, best_metrics, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    batch_ret = []
    batch_times = []
    batch_status = []
    with torch.no_grad():
        for idx, (rgb,res, label, sample_id,pos) in enumerate(test_dataloader):
            local_attention_map = {}
            sample_id = list(set(sample_id))[0]
            times = label[:,0]
            status = label[:,1]
            ##some batch all censored
            '''
            if not torch.any(status == 1):
                continue
            else:
                print(times,status)
            '''
            rgb = rgb.to(torch.bfloat16)
            res = res.to(torch.bfloat16)
            base_model = base_model.to(torch.bfloat16)
            if args.use_gpu:
                rgb,res, times,status = rgb.to(args.local_rank),res.to(args.local_rank), times.to(args.local_rank),status.to(args.local_rank)
            with autocast(dtype=torch.bfloat16):
                ret,attn_scores = base_model(rgb,res)
            batch_ret.append(ret)
            batch_times.append(times[0])
            batch_status.append(status[0])
            '''
            with open(f'attention_map/survival_TCGA/valid_attention_map_rank_{sample_id}_rank_{args.local_rank}.json', 'w') as f:
                json.dump(local_attention_map, f)
            '''
        batch_times = [t.unsqueeze(0) if t.dim() == 0 else t for t in batch_times]
        batch_status = [s.unsqueeze(0) if s.dim() == 0 else s for s in batch_status]
        batch_ret = torch.stack(batch_ret, dim=0)  # shape: (batch_size, 4)
        batch_times = torch.stack(batch_times, dim=0)  # shape: (batch_size,)
        batch_status = torch.stack(batch_status, dim=0)  # shape: (batch_size,)
        '''
        print("batch_ret",batch_ret)
        print("batch_times",batch_times)
        print("batch_status",batch_status)
        '''
        loss, acc = base_model.module.get_loss_acc(batch_ret, (batch_times,batch_status))
        print_log('[Validation] EPOCH: %d  acc = %.4f, best_acc = %.4f' % (epoch, acc[0], max(best_metrics.acc, acc[0])),
                  logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc[0], epoch)

    return AccMetric(acc[0])


def validate_survival_freeze(base_model, all_features_expr_test,all_labels_expr_test,all_sample_id_expr_test,all_pos_expr_test, epoch, val_writer, args, config, best_metrics, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    batch_ret = []
    batch_times = []
    batch_status = []
    with torch.no_grad():
        for idx in range(len(all_features_expr_test)):#// batch_size 
            label = all_labels_expr_test[idx]
            times = label[:,0]
            status = label[:,1]
            gathered_features = all_features_expr_test[idx]
            ret, attn_scores = base_model.module.forward_from_embedding(gathered_features)
            label = all_labels_expr_test[idx]
            batch_ret.append(ret)
            batch_times.append(times[0])
            batch_status.append(status[0])
        batch_times = [t.unsqueeze(0) if t.dim() == 0 else t for t in batch_times]
        batch_status = [s.unsqueeze(0) if s.dim() == 0 else s for s in batch_status]
        batch_ret = torch.stack(batch_ret, dim=0)  # shape: (batch_size, 4)
        #print(batch_ret)
        #print(batch_status)
        batch_times = torch.stack(batch_times, dim=0)  # shape: (batch_size,)
        batch_status = torch.stack(batch_status, dim=0)  # shape: (batch_size,)
        '''
        print("batch_ret",batch_ret)
        print("batch_times",batch_times)
        print("batch_status",batch_status)
        '''
        loss, acc = base_model.module.get_loss_acc(batch_ret, (batch_times,batch_status))
        
        print_log('[Test] EPOCH: %d  C-index = %.4f, best_acc = %.4f' % (epoch, acc[0], max(best_metrics.acc, acc[0])),
                  logger=logger)
        acc = acc[0]
        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return AccMetric(acc)


def validate_regression_freeze(base_model, all_features_expr_test,all_labels_expr_test,all_sample_id_expr_test,all_pos_expr_test, epoch, val_writer, args, config, best_metrics, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    batch_ret = []
    batch_times = []
    batch_status = []
    with torch.no_grad():
        for idx in range(len(all_features_expr_test)):#// batch_size 
            gathered_features = all_features_expr_test[idx]
            ret, attn_scores = base_model.module.forward_from_embedding(gathered_features)
            label = all_labels_expr_test[idx]
            #ret,attn_scores = base_model(rgb,res)
            loss, acc = base_model.module.get_loss_acc(ret, label[0])
            torch.cuda.synchronize()
            #print("ret ",ret.shape)
            batch_ret.append(ret)
            batch_status.append(label[0])
        batch_ret = torch.stack(batch_ret, dim=0)  # shape: (batch_size, 4)
        batch_status = torch.stack(batch_status, dim=0)  # shape: (batch_size,)
        ''''''
        print("batch_ret",batch_ret.shape)
        print("batch_status",batch_status.shape)
        
        loss, acc = base_model.module.get_loss_acc(batch_ret, batch_status)

        
        print_log('[Test] EPOCH: %d  Pearson = %.4f,R square = %.4f,cos = %.4f, best_acc = %.4f' % (epoch, acc[0],acc[1],acc[2], max(best_metrics.acc, acc[0])),
                  logger=logger)
        acc = acc[0]
        batch_ret = batch_ret.squeeze(1)
        if epoch %20 == 0:
            num_genes = batch_ret.shape[1]  # 基因数量
            pearson_corrs = []
            r2_scores = []
            cos_similarities = []
            
            for gene_idx in range(num_genes):
                # 提取当前基因在所有样本中的预测值和标签值
                label_gene = batch_status[:, gene_idx].clone().detach().cpu().numpy()
                ret_gene = batch_ret[:, gene_idx].clone().detach().cpu().numpy()
                if np.all(label_gene == 0):  # 如果所有值都为 0，跳过
                    continue
                
                # 1. 计算 Pearson 相关系数
                corr, _ = pearsonr(label_gene, ret_gene)
                pearson_corrs.append((gene_idx, corr))  # 存储索引和相关性
                
                # 2. 计算 R² 分数
                r2 = r2_score(label_gene, ret_gene)
                r2_scores.append((gene_idx, r2))  # 存储索引和 R²
                
                # 3. 计算余弦相似度
                cos_sim = cosine_similarity(label_gene.reshape(1, -1), ret_gene.reshape(1, -1))[0][0]
                cos_similarities.append((gene_idx, cos_sim))  # 存储索引和相似度
            
            # 计算所有基因的平均值
            mean_pearson_corr = np.mean([x[1] for x in pearson_corrs])
            mean_r2 = np.mean([x[1] for x in r2_scores])
            mean_cos_sim = np.mean([x[1] for x in cos_similarities])

            # 计算 top 2000 基因
            pearson_corrs_sorted = sorted(pearson_corrs, key=lambda x: abs(x[1]), reverse=True)[:2000]

            # 用字典提高索引效率
            r2_dict = dict(r2_scores)
            cos_sim_dict = dict(cos_similarities)

            top2000_pearson = np.mean([x[1] for x in pearson_corrs_sorted])
            top2000_r2 = np.mean([r2_dict[x[0]] for x in pearson_corrs_sorted if x[0] in r2_dict])
            top2000_cos_sim = np.mean([cos_sim_dict[x[0]] for x in pearson_corrs_sorted if x[0] in cos_sim_dict])

            # 计算 Pearson 大于 0.5 的基因数量
            pearson_gt_0_5_count = sum(1 for x in pearson_corrs if x[1] > 0.5)

            print(f"Epoch {epoch}")
            print(f"Mean Pearson Corr: {mean_pearson_corr:.4f}")
            print(f"Mean R² Score: {mean_r2:.4f}")
            print(f"Mean Cosine Similarity: {mean_cos_sim:.4f}")
            print(f"Top 2000 Genes - Mean Pearson Corr: {top2000_pearson:.4f}")
            print(f"Top 2000 Genes - Mean R² Score: {top2000_r2:.4f}")
            print(f"Top 2000 Genes - Mean Cosine Similarity: {top2000_cos_sim:.4f}")
            print(f"Number of Genes with Pearson > 0.5: {pearson_gt_0_5_count}")

        if epoch %10 == 0:
            save_path = os.path.join(args.experiment_path, "result.pkl")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({'label_gene': batch_status, 'ret_gene': batch_ret}, f)

            print(f"Saved label_gene and ret_gene to {save_path}")

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return AccMetric(acc)


def validate_classifier_freeze(base_model, all_features_expr_test,all_labels_expr_test,all_sample_id_expr_test,all_pos_expr_test, epoch, val_writer, args, config, best_metrics, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    batch_ret = []
    batch_times = []
    batch_status = []
    with torch.no_grad():
        for idx in range(len(all_features_expr_test)):#// batch_size 
            gathered_features = all_features_expr_test[idx]
            ret, attn_scores = base_model.module.forward_from_embedding(gathered_features)
            label = all_labels_expr_test[idx]
            #ret,attn_scores = base_model(rgb,res)
            loss, acc = base_model.module.get_loss_acc(ret, label[0])
            torch.cuda.synchronize()
            #print("ret ",ret.shape)
            batch_ret.append(ret)
            batch_status.append(label[0])
        batch_ret = torch.stack(batch_ret, dim=0)  # shape: (batch_size, 4)
        batch_status = torch.stack(batch_status, dim=0)  # shape: (batch_size,)
        '''
        print("batch_ret",batch_ret)
        print("batch_times",batch_times)
        print("batch_status",batch_status)
        '''
        loss, acc = base_model.module.get_loss_acc(batch_ret, batch_status)
        
        print_log('[Test] EPOCH: %d  Pearson = %.4f,R square = %.4f,cos = %.4f, best_acc = %.4f' % (epoch, acc[0],acc[1],acc[2], max(best_metrics.acc, acc[0])),
                  logger=logger)
        acc = acc[0]
        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return AccMetric(acc)


def validate_classifier(base_model, test_dataloader, epoch, val_writer, args, config, best_metrics, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    batch_ret = []
    batch_times = []
    batch_status = []
    with torch.no_grad():
        for idx, (rgb,res, label, sample_id,pos) in enumerate(test_dataloader):
            local_attention_map = {}
            sample_id = list(set(sample_id))[0]
            if args.use_gpu:
                rgb,res, label = rgb.to(args.local_rank),res, label.to(args.local_rank)

            ret,attn_scores = base_model(rgb,res)
            loss, acc = base_model.module.get_loss_acc(ret, label[0])
            attn_scores = attn_scores.squeeze(1)
            torch.cuda.synchronize()
            '''
            for i in range(len(pos)):
                pos_str = pos[i]
                attn_value = attn_scores[i].item()
                local_attention_map[pos_str] = attn_value
            #print("ret ",ret.shape)
            '''
            batch_ret.append(ret)
            batch_status.append(label[0])
        batch_ret = torch.stack(batch_ret, dim=0)  # shape: (batch_size, 4)
        batch_status = torch.stack(batch_status, dim=0)  # shape: (batch_size,)
        '''
        print("batch_ret",batch_ret)
        print("batch_times",batch_times)
        print("batch_status",batch_status)
        '''
        loss, acc = base_model.module.get_loss_acc(batch_ret, batch_status)
        
        print_log('[Test] EPOCH: %d  TOP1acc = %.4f,TOP3acc = %.4f,TOP5acc = %.4f,Weighted F1 = %.4f,AUC = %.4f, best_acc = %.4f' % (epoch, acc[0][0],acc[0][1],acc[0][2],acc[1],acc[2], max(best_metrics.acc, acc[0][0])),
                  logger=logger)
        acc = acc[0][0]
        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return AccMetric(acc)

def validate_classifier_expr(base_model, all_features_expr_test,all_labels_expr_test,all_sample_id_expr_test,all_pos_expr_test, epoch, val_writer, args, config, best_metrics, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    batch_ret = []
    batch_times = []
    batch_status = []
    with torch.no_grad():
        for idx in range(len(all_features_expr_test)):#// batch_size 
            gathered_features = all_features_expr_test[idx]
            ret, attn_scores = base_model.module.forward_from_embedding(gathered_features)
            label = all_labels_expr_test[idx]
            #ret,attn_scores = base_model(rgb,res)
            loss, acc = base_model.module.get_loss_acc(ret, label[0])
            torch.cuda.synchronize()
            #print("ret ",ret.shape)
            batch_ret.append(ret)
            batch_status.append(label[0])
        batch_ret = torch.stack(batch_ret, dim=0)  # shape: (batch_size, 4)
        batch_status = torch.stack(batch_status, dim=0)  # shape: (batch_size,)
        '''
        print("batch_ret",batch_ret)
        print("batch_times",batch_times)
        print("batch_status",batch_status)
        '''
        loss, acc = base_model.module.get_loss_acc(batch_ret, batch_status)
        
        print_log('[Test] EPOCH: %d  TOP1acc = %.4f,TOP3acc = %.4f,TOP5acc = %.4f,Weighted F1 = %.4f,AUC = %.4f, best_acc = %.4f' % (epoch, acc[0][0],acc[0][1],acc[0][2],acc[1],acc[2], max(best_metrics.acc, acc[0][0])),
                  logger=logger)
        acc = acc[0][0]
        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return AccMetric(acc)



def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.val)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()

    test_pred = []
    test_label = []
    base_model.eval()
    with torch.no_grad():
        for idx, (embedding, label) in enumerate(test_dataloader):
            if args.use_gpu:
                embedding, label = embedding.to(args.local_rank), label.to(args.local_rank)

            logits = base_model(embedding)

            target = label.view(-1)
            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] acc = %.4f' % acc, logger=logger)

def get_encoder_batch(args, config, train_writer=None, val_writer=None):
    def print_gpu_memory():
        if torch.cuda.is_available():
            print(f"当前GPU占用内存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
            #print(f"模型计算图保留的内存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
    print("start")
    print_gpu_memory()
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    # build model
    print("DataLoader")
    print_gpu_memory()
    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt(args.ckpts)
    # parameter setting
    start_epoch = 0
    best_metrics = AccMetric(0.)
    metrics = AccMetric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = AccMetric(best_metrics)

    if args.use_gpu:
        base_model.to(args.local_rank)
    print("build model")
    print_gpu_memory()
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)
    print("build optimizer")
    print_gpu_memory()

    # trainval
    # training
    #base_model._set_static_graph() 
    base_model.zero_grad()
    print("lr start:",optimizer.param_groups[0]['lr'])
    local_attention_map = dict()
    for batch in train_dataloader:
        rgb,res, label, sample_id,pos = batch  # 假设你的数据集返回的是 (inputs, labels)
        if "0" in str(base_model.device):
            print(list(set(label))[0],list(set(sample_id))[0],rgb.shape)
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    batch_ret = []
    batch_times = []
    batch_status = []
    with torch.no_grad():
        for idx, (rgb,res, label, sample_id,pos) in enumerate(test_dataloader):
            local_attention_map = {}
            sample_id = list(set(sample_id))[0]
            if args.use_gpu:
                rgb,res, label = rgb.to(args.local_rank),res, label.to(args.local_rank)

            ret,attn_scores = base_model(rgb,res)
            loss, acc = base_model.module.get_loss_acc(ret, label[0])
            attn_scores = attn_scores.squeeze(1)
            torch.cuda.synchronize()
            for i in range(len(pos)):
                pos_str = pos[i]
                attn_value = attn_scores[i].item()
                local_attention_map[pos_str] = attn_value
            #print("ret ",ret.shape)
            batch_ret.append(ret)
            batch_status.append(label[0])
            if "0" in str(base_model.device):
                print("valid acc:",acc,"sample_id ",sample_id)
            if config.abmil_type == "CNV":
                with open(f'attention_map/cnv_HD/attention_map_rank_{sample_id}_rank_{args.local_rank}.json', 'w') as f:
                    json.dump(local_attention_map, f)
            elif config.abmil_type == "RNA":
                with open(f'attention_map/RNA_TCGA/attention_map_rank_{sample_id}_rank_{args.local_rank}.json', 'w') as f:
                    json.dump(local_attention_map, f)
            elif config.abmil_type == "Survival":
                with open(f'attention_map/survival_TCGA/attention_map_rank_{sample_id}_rank_{args.local_rank}.json', 'w') as f:
                    json.dump(local_attention_map, f)
            else:
                with open(f'attention_map/DR_TCGA/attention_map_rank_{sample_id}_rank_{args.local_rank}.json', 'w') as f:
                    json.dump(local_attention_map, f)
        batch_ret = torch.stack(batch_ret, dim=0)  # shape: (batch_size, 4)
        atch_status = torch.stack(batch_status, dim=0)  # shape: (batch_size,)



def get_attention_batch(args, config, train_writer=None, val_writer=None):
    print("start")
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    # build model
    print("DataLoader")
    print_gpu_memory()
    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt(args.ckpts)
    # parameter setting
    start_epoch = 0
    best_metrics = AccMetric(0.)
    metrics = AccMetric(0.)

    # resume ckpts
    save_attention_dir = args.experiment_path
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = AccMetric(best_metrics)

    if args.use_gpu:
        base_model.to(args.local_rank)
    print("build model")
    print_gpu_memory()
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)
    print("build optimizer")
    print_gpu_memory()

    # trainval
    # training
    #base_model._set_static_graph() 
    base_model.zero_grad()
    print("lr start:",optimizer.param_groups[0]['lr'])
    local_attention_map = dict()
    for batch in train_dataloader:
        rgb,res, label, sample_id,pos = batch  # 假设你的数据集返回的是 (inputs, labels)
        if "0" in str(base_model.device):
            print(list(set(label))[0],list(set(sample_id))[0],rgb.shape)
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    batch_ret = []
    batch_times = []
    batch_status = []
    with torch.no_grad():
        for idx, (rgb,res, label, sample_id,pos) in enumerate(test_dataloader):
            local_attention_map = {}
            sample_id = list(set(sample_id))[0]
            if args.use_gpu:
                rgb,res, label = rgb.to(args.local_rank),res, label.to(args.local_rank)

            ret,attn_scores = base_model(rgb,res)
            loss, acc = base_model.module.get_loss_acc(ret, label[0])
            attn_scores = attn_scores.squeeze(1)
            torch.cuda.synchronize()
            for i in range(len(pos)):
                pos_str = pos[i]
                attn_value = attn_scores[i].item()
                local_attention_map[pos_str] = attn_value
            #print("ret ",ret.shape)
            batch_ret.append(ret)
            batch_status.append(label[0])
            if "0" in str(base_model.device):
                print("valid acc:",acc,"sample_id ",sample_id)
            if config.abmil_type == "CNV":
                with open(f'attention_map/cnv_HD/attention_map_rank_{sample_id}_rank_{args.local_rank}.json', 'w') as f:
                    json.dump(local_attention_map, f)
            elif config.abmil_type == "RNA":
                with open(f'attention_map/RNA_TCGA/attention_map_rank_{sample_id}_rank_{args.local_rank}.json', 'w') as f:
                    json.dump(local_attention_map, f)
            elif config.abmil_type == "Survival":
                with open(f'attention_map/survival_TCGA/attention_map_rank_{sample_id}_rank_{args.local_rank}.json', 'w') as f:
                    json.dump(local_attention_map, f)
            else:
                with open(f'attention_map/DR_TCGA/attention_map_rank_{sample_id}_rank_{args.local_rank}.json', 'w') as f:
                    json.dump(local_attention_map, f)
        batch_ret = torch.stack(batch_ret, dim=0)  # shape: (batch_size, 4)
        batch_status = torch.stack(batch_status, dim=0)  # shape: (batch_size,)Z
