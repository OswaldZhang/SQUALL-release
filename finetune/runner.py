import os
import torch
import torch.nn as nn
from models import builder
from utils import dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import pickle

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


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)

    if config.model.type != "full":
        for name, param in base_model.named_parameters():
            if not 'cls' in name:
                param.requires_grad = False

    # parameter setting
    start_epoch = 0
    best_metrics = AccMetric(0.)
    metrics = AccMetric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = AccMetric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    # DDP
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
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        for idx, (rgb, res, label, _) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            data_time.update(time.time() - batch_start_time)

            if args.use_gpu:
                rgb, res, label = rgb.to(args.local_rank), res.to(args.local_rank), label.to(args.local_rank)
            rgb = rgb.permute(0, 3, 1, 2)

            cls_feat = base_model.module.forward_rgb(rgb, res)
            ret = base_model.module.classifier(cls_feat)
            loss, acc = base_model.module.get_loss_acc(ret, label)

            loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %s lr = %.6f' %
                          (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                           ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger=logger)

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
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







def run_net_freeze(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    # build model
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
    '''
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)
    '''

    if args.use_gpu:
        base_model.to(args.local_rank)

    # DDP
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
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # trainval
    # training
    base_model.zero_grad()
    print("lr start:",optimizer.param_groups[0]['lr'])
    local_attention_map = dict()
    '''
    for idx, (rgb, res, label, sample_id, pos) in enumerate(train_dataloader):
        if "0" in str(base_model.device):
            print(list(set(label))[0],list(set(sample_id))[0],rgb.shape)
    '''
    
    # Step 2: 
    all_features = []
    all_labels = []
    all_sample_id = []
    all_pos = []
    all_features_expr_test = []
    all_labels_expr_test = []
    all_sample_id_expr_test = []
    all_pos_expr_test = []
    n_batches = len(train_dataloader)
    print("train total ",n_batches)
    for idx, (rgb, res, label, sample_id, pos) in enumerate(train_dataloader):

        torch.autograd.set_detect_anomaly(True)
        if idx %100 ==0:
            print("train features ",idx)
            print("rgb.shape",rgb.shape)
        if args.use_gpu:
            rgb, res,label= rgb.to(args.local_rank), res.to(args.local_rank), label.to(args.local_rank)

        # 
        rgb = rgb  # .half() 
        res = res  # .half()
        label = label.float()  # .half()

        # 
        with torch.no_grad():  # 
            cls_feat = base_model.module.forward_rgb(rgb, res)
        all_features.append(cls_feat)
        all_labels.append(label)
        all_sample_id.append(sample_id)
        all_pos.append(pos)
    #if config.fwd_type == 'expr' or config.fwd_type == 'benchmark':
    for idx, (rgb, res, label, sample_id, pos) in enumerate(test_dataloader):
        torch.autograd.set_detect_anomaly(True)
        if idx %100 ==0:
            print("test features ",idx)

        if args.use_gpu:
            rgb, res,label= rgb.to(args.local_rank), res.to(args.local_rank), label.to(args.local_rank)

        # 
        rgb = rgb  # .half() 
        res = res  # .half()
        label = label.float()  # .half()

        # 
        with torch.no_grad():  # 
            cls_feat = base_model.module.forward_rgb(rgb, res)
            all_features_expr_test.append(cls_feat)
            all_labels_expr_test.append(label)
            all_sample_id_expr_test.append(sample_id)
            all_pos_expr_test.append(pos)






    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        for idx in range(len(all_features)):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            data_time.update(time.time() - batch_start_time)
            cls_feat = all_features[idx]
            label = all_labels[idx]
            #print("cls_feat.shape",cls_feat.shape)
            #print("label",label.shape)
            ret = base_model.module.classifier(cls_feat)
            loss, acc = base_model.module.get_loss_acc(ret, label)

            loss.backward()
            if "0" in str(base_model.device) and idx%100 == 0:
                    #print("ret:",ret)
                    #print("label:",label[0])
                    print("loss:",loss,"acc:",acc,"lr:",optimizer.param_groups[0]['lr'],"size",cls_feat.shape," epoch ",epoch)

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                #acc = dist_utils.reduce_tensor(acc[0][0], args)
                losses.update([loss.item()])
            else:
                losses.update([loss.item()])
            

            if args.distributed:
                torch.cuda.synchronize()
            '''
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                #train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)
            '''

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()
        ''''''
        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)
        

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [losses.avg(0)]],
                   optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate_freeze(base_model, all_features_expr_test,all_labels_expr_test,all_sample_id_expr_test,all_pos_expr_test, epoch, val_writer, args, config, best_metrics, logger=logger)

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




def validate_freeze(base_model, all_features_expr_test,all_labels_expr_test,all_sample_id_expr_test,all_pos_expr_test, epoch, val_writer, args, config, best_metrics, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    with torch.no_grad():
        for idx in range(len(all_features_expr_test)):#// batch_size 
            cls_feat = all_features_expr_test[idx]
            label = all_labels_expr_test[idx]
            logits = base_model.module.classifier(cls_feat)

            target = label.view(-1)
            pred = logits.argmax(-1).view(-1)

            test_pred.append(logits)
            test_label.append(label)

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)
        #test_pred = torch.stack(test_pred, dim=1)
        #test_label = torch.stack(test_label, dim=1)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        #acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        loss, acc = base_model.module.get_loss_acc(test_pred, test_label)
        print_log('[Test] EPOCH: %d  TOP1acc = %.4f,TOP3acc = %.4f,TOP5acc = %.4f,Weighted F1 = %.4f,AUC = %.4f, best_acc = %.4f' % (epoch, acc[0][0],acc[0][1],acc[0][2],acc[1],acc[2], max(best_metrics.acc, acc[0][0])),
                  logger=logger)
        acc = acc[0][0]
        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return AccMetric(acc)





def validate(base_model, test_dataloader, epoch, val_writer, args, config, best_metrics, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    with torch.no_grad():
        for idx, (rgb, res, label, _) in enumerate(test_dataloader):
            if args.use_gpu:
                rgb, res, label = rgb.to(args.local_rank), res.to(args.local_rank), label.to(args.local_rank)
            rgb = rgb.permute(0, 3, 1, 2)

            cls_feat = base_model.module.forward_rgb(rgb, res)
            logits = base_model.module.classifier(cls_feat)

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
        print_log('[Validation] EPOCH: %d  acc = %.4f, best_acc = %.4f' % (epoch, acc, max(best_metrics.acc, acc)),
                  logger=logger)

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
        for idx, (rgb, res, label, _) in enumerate(test_dataloader):
            if args.use_gpu:
                rgb, res, label = rgb.to(args.local_rank), res.to(args.local_rank), label.to(args.local_rank)
            rgb = rgb.permute(0, 3, 1, 2)

            cls_feat = base_model.module.forward_rgb(rgb, res)
            logits = base_model.module.classifier(cls_feat)

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


def string_to_tensor(s, max_len):
    byte_array = list(s.encode('utf-8'))
    byte_array += [0] * (max_len - len(byte_array))
    return torch.tensor(byte_array, dtype=torch.uint8)


def tensor_to_string(t):
    byte_array = t.cpu().numpy().tolist()
    return bytes([b for b in byte_array if b != 0]).decode('utf-8')


def vote_slide(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.val)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

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

    pred_list = []
    label_list = []
    sample_id_list = []
    base_model.eval()
    with torch.no_grad():
        for idx, (rgb, res, label, sample_id) in enumerate(test_dataloader):
            sample_id = torch.stack([string_to_tensor(s, 32) for s in sample_id])
            if args.use_gpu:
                rgb, res, label, sample_id = rgb.to(args.local_rank), res.to(args.local_rank), label.to(args.local_rank), sample_id.to(args.local_rank)
            rgb = rgb.permute(0, 3, 1, 2)

            cls_feat = base_model.module.forward_rgb(rgb, res)
            logits = base_model.module.classifier(cls_feat)

            target = label.view(-1)
            pred = logits.argmax(-1).view(-1)

            pred_list.append(pred.detach())
            label_list.append(target.detach())
            sample_id_list.append(sample_id.detach())

        pred_list = torch.cat(pred_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        sample_id_list = torch.cat(sample_id_list, dim=0)

        if args.distributed:
            pred_list = dist_utils.gather_tensor(pred_list, args)
            label_list = dist_utils.gather_tensor(label_list, args)
            sample_id_list = dist_utils.gather_tensor(sample_id_list, args)

        sample_id_list = [tensor_to_string(t) for t in sample_id_list]

        pred_dict = {}
        label_dict = {}
        for i, sample_id in enumerate(sample_id_list):
            if sample_id not in pred_dict:
                pred_dict[sample_id] = []
                label_dict[sample_id] = label_list[i].item()
            pred_dict[sample_id].append(pred_list[i].item())

    # vote
    acc = sum(max(set(v), key=v.count) == label_dict[k] for k, v in pred_dict.items()) / len(pred_dict) * 100
    print_log('[Validation] acc = %.4f' % acc, logger=logger)


def get_embedding(args, config):
    logger = get_logger(args.log_name)
    print_log('Get embedding ... ', logger=logger)
    (_, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    base_model = builder.model_builder(config.model)
    #print("base_model",base_model)
    #print(print("args.ckpts",args.ckpts))

    # load checkpoints bug version
    #builder.load_model(base_model, args.ckpts, logger=logger)
    #load checkpoint work version
    if args.ckpts is not None:
        base_model.load_model_from_ckpt(args.ckpts)

    if args.use_gpu:
        base_model.to(args.local_rank)

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

    training_split = {
        'train': train_dataloader
        #'test': test_dataloader
    }
    start_time = time.time()
    with torch.no_grad():
        for split, dataloader in training_split.items():
            embedding_list = []
            label_list = []
            sample_id_list = []
            pos_list = []
            base_model.eval()
            for idx, (rgb ,res, label, sample_id,pos) in enumerate(dataloader):#if batchsize = 1 turn to                 
                #sample_id = torch.stack([string_to_tensor(s, 32) for s in sample_id])
                if args.use_gpu:
                    rgb,res, label, sample_id = rgb.to(args.local_rank), res.to(args.local_rank), label.to(args.local_rank), sample_id
                rgb = rgb.permute(0, 3, 1, 2)
                cls_feat = base_model.module.forward_rgb(rgb, res)
                #print("cls_feat",cls_feat)
                if idx % 100 == 0:
                    current_time = time.time()
                    elapsed_time = current_time - start_time  # 
                    time_per_100 = elapsed_time / (idx // 100 + 1)  #  100 
                    remaining_batches = (len(dataloader) - idx) // 100
                    estimated_remaining_time = remaining_batches * time_per_100
                    
                    print(f"Processing batch {idx}, time for last 100: {time_per_100:.2f}s, "
                          f"estimated remaining time: {estimated_remaining_time:.2f}s")
                    
                    #  100 
                    start_time = current_time
                embedding_list.append(cls_feat.detach().cpu())
                for poses in pos:
                    label_list.append(poses)
                #label_list.append(pos)
                for sample_ids in sample_id:
                    sample_id_list.append(sample_ids)#

            embedding_list = torch.cat(embedding_list, dim=0)
            #label_list = torch.cat(label_list, dim=0)
            #sample_id_list = torch.cat(sample_id_list, dim=0)
            print("embedding_list",len(embedding_list))
            print("label_list",len(label_list))
            print("sample_id_list",len(sample_id_list))
            if args.distributed:
                embedding_list = dist_utils.gather_tensor(embedding_list, args)
                label_list = dist_utils.gather_tensor(label_list, args)
                sample_id_list = dist_utils.gather_tensor(sample_id_list, args)

            #sample_id_list = [tensor_to_string(t) for t in sample_id_list]

            # save embedding in args.save_dir
            if args.local_rank == 0:
                embedding_dict = {}
                label_dict = {}

                ''''''
                for i, sample_id in enumerate(sample_id_list):
                    if sample_id not in embedding_dict:
                        embedding_dict[sample_id] = []
                    embedding_dict[sample_id].append(embedding_list[i])
                

                for sample_id, embedding in embedding_dict.items():
                    split_path = os.path.join(args.save_dir, split)
                    sample_path = os.path.join(split_path, sample_id)
                    os.makedirs(sample_path, exist_ok=True)
                    torch.save(torch.stack(embedding), sample_path+".pt")
                    print(sample_path+".pt")
                    pkl_file_path = sample_path+"_pos.pkl"
                    print(pkl_file_path)
                    #  label_list  .pkl 
                    with open(pkl_file_path, 'wb') as f:
                        pickle.dump(label_list, f)
