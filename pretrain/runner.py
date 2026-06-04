import torch
import torch.nn as nn
from models import builder
from utils import dist_utils
import os
import time
import numpy as np
from utils.logger import *
from utils.AverageMeter import AverageMeter
from sklearn.svm import LinearSVC
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import json
import yaml


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
def is_main_process(args):
    """"""
    return not hasattr(args, 'local_rank') or args.local_rank == 0


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC(C=0.075)
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0] * 100

def setup_distributed(args):
    args.rank = int(os.environ.get('RANK', 0))  #  -1
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    print("world_size",args.world_size)
    print("node_rank",args.rank)
    print("local_rank",args.local_rank)
    torch.cuda.set_device(args.local_rank)
    print(f"Running DDP on rank {args.rank}.")


def run_net(args, config, train_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (pretrain_sampler, pretrain_dataloader),(_, test_dataloader) =builder.dataset_builder(args, config.dataset.train),builder.dataset_builder(args, config.dataset.val)

    # build model
    base_model = builder.model_builder(config.model)
    print("qbw test loading pretrain")
    ###qbw test
    # load checkpoints
    if args.resume:
        builder.load_model(base_model, args.ckpts, logger=logger)
    ###qbw test
    if args.local_rank == 0:
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                num_params = param.numel() / 1e6
                print(f"Layer: {name} | Number of parameters: {num_params:.3f} M")

        total_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad) / 1e6
        print(f"Total number of parameters: {total_params:.3f} M")

    if args.use_gpu:
        base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    best_metrics = AccMetric(0.)
    metrics = AccMetric(0.)
    scaler = GradScaler()
    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = AccMetric(best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    # DDP
    if args.distributed:
        setup_distributed(args)
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()],
                                                         find_unused_parameters=False,bucket_cap_mb=5000)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # training
    base_model.zero_grad()
    '''
    with profile(activities=[
        ProfilerActivity.CPU,  # CPU
        ProfilerActivity.CUDA  # CUDA
    ], 
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1), 
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_ddp_qbw_lowres_12_5_praameters'),  # TensorBoard
        record_shapes=True,  # 
        profile_memory=True,  # 
        with_stack=True  # 
    ) as prof:  
    '''
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            pretrain_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_loader_time = AverageMeter()
        data_time = AverageMeter()
        run_time = AverageMeter()
        model_time = AverageMeter()
        synchronize_time = AverageMeter()
        rgb_losses = AverageMeter(['rgb_loss'])
        expr_losses = AverageMeter(['expr_loss'])
        num_iter = 0

        avg_batch_data_time = 0
        avg_batch_run_time = 0
        avg_batch_model_time = 0
        avg_batch_synchronize_time = 0

        base_model.train()  # set model to training mode
        n_batches = len(pretrain_dataloader)
        batch_start_time = time.time()
        for idx, (rgb, expr, res, _,_) in enumerate(pretrain_dataloader): #qbw 10.16 5-4
            #  # Start time for the batch processing
            data_loader_time.update(time.time() - batch_start_time)

            num_iter += 1
            n_itr = epoch * n_batches + idx
            data_time.update(time.time() - batch_start_time)
            run_start_time = time.time()

            if args.use_gpu:
                rgb, expr, res = rgb.to(args.local_rank,non_blocking=True), expr.to(args.local_rank,non_blocking=True), res.to(args.local_rank,non_blocking=True)
            run_time.update(time.time() - run_start_time)

            model_start_time = time.time()
            rgb = rgb.permute(0, 3, 1, 2)
            #expr = expr.permute(0, 3, 1, 2) #move to squall forward
            with autocast():
                rgb_loss, expr_loss = base_model(rgb, expr, res)
            loss = rgb_loss + expr_loss
            scaler.scale(loss).backward()            
            # forward
            if num_iter == config.step_per_update:
                    num_iter = 0
                    scaler.step(optimizer)  
                    scaler.update() 
                    base_model.zero_grad()
            ''''''
            model_time.update(time.time() - model_start_time)
            if args.distributed:
                rgb_loss = dist_utils.reduce_tensor(rgb_loss, args)
                rgb_losses.update([rgb_loss.item() * 1000])
                expr_loss = dist_utils.reduce_tensor(expr_loss, args)
                expr_losses.update([expr_loss.item() * 1000])
            else:
                rgb_losses.update([rgb_loss.item() * 1000])
                expr_losses.update([expr_loss.item() * 1000])

            
            synchronize_start_time = time.time()
            if args.distributed:
                torch.cuda.synchronize()
            synchronize_time.update(time.time() - synchronize_start_time)

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/rgb_loss', rgb_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/expr_loss', expr_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            avg_batch_data_time += data_time.val()
            avg_batch_run_time += run_time.val()
            avg_batch_model_time += model_time.val()
            avg_batch_synchronize_time += synchronize_time.val()

            if (idx+1) % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) LoadTime = %.3f (s) ModelTime = %.3f (s) synchronizeTime = %.3f (s) RGB_Loss = %s '
                          'Expr_Loss = %s lr = %.6f' %
                          (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),run_time.val(),model_time.val(),synchronize_time.val(),
                           ['%.4f' % l for l in rgb_losses.val()], ['%.4f' % l for l in expr_losses.val()],
                           optimizer.param_groups[0]['lr']), logger=logger)

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) RGB_Loss = %s Expr_Loss = %s lr = %.6f (s) DataTime = %.3f (s) LoadTime = %.3f (s) ModelTime = %.3f (s) synchronizeTime = %.3f (s)' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in rgb_losses.avg()],
                   ['%.4f' % l for l in expr_losses.avg()], optimizer.param_groups[0]['lr'], avg_batch_data_time,avg_batch_run_time,avg_batch_model_time,avg_batch_synchronize_time), logger=logger)
        if config.if_validate:
            if epoch % args.val_freq == 0:#and epoch != 0
                # Validate the current model
                metrics = validate(base_model, pretrain_dataloader, epoch, args, config, best_metrics, logger=logger)
                better = metrics.better_than(best_metrics)
                # Save ckeckpoints
                if better:
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                            logger=logger)
                    print_log(
                        "--------------------------------------------------------------------------------------------",
                        logger=logger)
        if not dist.is_initialized() or dist.get_rank() == 0:
            builder.save_checkpoint(base_model, optimizer, epoch, None, None, 'ckpt-last', args, logger=logger)
            if epoch %50 == 0:
                builder.save_checkpoint(base_model, optimizer, epoch, None, None, 'ckpt-epoch-' + str(epoch), args, logger=logger)  # qbw change        
    if train_writer is not None:
        train_writer.close()


def validate(base_model, pretrain_dataloader,epoch, args, config, best_metrics, logger=None):
    base_model.eval()  # set model to eval mode
    train_list = []
    val_list = []
    train_subset_dict = json.load(open(config.train_test_split))

    for i in train_subset_dict.keys():
        if train_subset_dict[i]["set"] == "train":
            train_list.append(i)
        else:
            val_list.append(i)
    print("train:",len(train_list))
    print("test:",len(val_list))
    print("train_list",train_list[:10])
    test_features = []
    test_label = []
    train_features = []
    train_label = []
    with torch.no_grad():
        for idx, (rgb, _, res, label, sample_id) in enumerate(pretrain_dataloader):
            if args.use_gpu:
                rgb, res, label = rgb.to(args.local_rank), res.to(args.local_rank), label.to(args.local_rank)

            rgb = rgb.permute(0, 3, 1, 2)
            #expr = expr.permute(0, 3, 1, 2)

            feature = base_model.module.forward_rgb(rgb, res)
            feature = torch.mean(feature, dim=1)
            if idx == 1:
                print("feature.shape",feature.shape)
            target = label.view(-1)
            if idx  == 1:
                print("train feature.shape",feature[0,:].detach().shape)
            for sample in sample_id:
                if sample in train_list:
                    train_features.append(feature[sample_id.index(sample),:].detach())
                    train_label.append(target[sample_id.index(sample)].detach())
                else:
                    test_features.append(feature[sample_id.index(sample),:].detach())
                    test_label.append(target[sample_id.index(sample)].detach())
        print("train_features:",len(train_features))
        print("test_features:",len(test_features))

        train_features = torch.stack(train_features, dim=0)
        train_label = torch.stack(train_label, dim=0)
        test_features = torch.stack(test_features, dim=0)
        test_label = torch.stack(test_label, dim=0)
        print("train_features.shape",train_features.shape)
        print("train_label.shape",train_label.shape)
        print("test_features.shape",test_features.shape)
        print("test_label.shape",test_label.shape)

        ''''''
        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)
        print("gather ready")
        

        acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(),
                           test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f, best_acc = %.4f' % (epoch, acc, max(best_metrics.acc, acc)),
                  logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    return AccMetric(acc)


def test_svm(args, config, train_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
                                                               builder.dataset_builder(args, config.dataset.val)

    # build model
    base_model = builder.model_builder(config.model)

    if args.use_gpu:
        base_model.to(args.local_rank)

    ckpts = torch.load(args.ckpts, map_location='cpu')
    base_model.load_state_dict(ckpts['base_model'])
    print_log('Successful load ckpts from %s' % args.ckpts, logger=logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()],
                                                         find_unused_parameters=False)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()

    # training
    base_model.zero_grad()
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []
    train_features = []
    train_label = []
    with torch.no_grad():
        for idx, (rgb, _, res, label) in enumerate(train_dataloader):
            if args.use_gpu:
                rgb, res, label = rgb.to(args.local_rank), res.to(args.local_rank), label.to(args.local_rank)

            rgb = rgb.permute(0, 3, 1, 2)

            feature = base_model.module.forward_rgb(rgb, res)
            feature = torch.mean(feature, dim=1)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (rgb, _, res, label) in enumerate(test_dataloader):
            if args.use_gpu:
                rgb, res, label = rgb.to(args.local_rank), res.to(args.local_rank), label.to(args.local_rank)

            rgb = rgb.permute(0, 3, 1, 2)

            feature = base_model.module.forward_rgb(rgb, res)
            feature = torch.mean(feature, dim=1)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())

        train_features = torch.stack(train_features, dim=0)
        train_label = torch.stack(train_label, dim=0)
        test_features = torch.stack(test_features, dim=0)
        test_label = torch.stack(test_label, dim=0)
        '''
        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)
        '''

        acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(),
                           test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] acc = %.4f' % acc, logger=logger)

        if args.distributed:
            torch.cuda.synchronize()