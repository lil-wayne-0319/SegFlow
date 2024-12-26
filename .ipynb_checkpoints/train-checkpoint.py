import os
import time
import datetime
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from datasets import MVTecDataset, VisADataset, MIADDataset
from models.extractors import build_extractor
from models.flow_models import build_msflow_model, build_flow_model_wtconv
from post_process import post_process
from utils import Score_Observer, t2np, positionalencoding2d, save_weights, load_weights
from evaluations import eval_det_loc

from torch.utils.data import DataLoader, SubsetRandomSampler


from PIL import Image


def save_img(output_tensor):
    save_dir = './save_test'
    img_num = output_tensor.shape[0]
    for i in range(img_num):
        img = output_tensor[i, :, :]
        # img = np.squeeze(img)

        # 计算数组的最小值和最大值
        min_val = np.min(img)
        max_val = np.max(img)

        # 归一化到 [0, 1]
        normalized_arr = (img - min_val) / (max_val - min_val)
        normalized_data = (normalized_arr * 255).astype(np.uint8)
        pil_image = Image.fromarray(normalized_data)
        # 保存图像
        pil_image.save(os.path.join(save_dir, 'output_image_{}.png'.format(i)))


def model_forward(c, extractor, parallel_flows, fusion_flow, image):
    h_list = extractor(image)
    if c.pool_type == 'avg':
        pool_layer = nn.AvgPool2d(3, 2, 1)
    elif c.pool_type == 'max':
        pool_layer = nn.MaxPool2d(3, 2, 1)
    else:
        pool_layer = nn.Identity()

    z_list = []
    parallel_jac_list = []
    for idx, (h, parallel_flow, c_cond) in enumerate(zip(h_list, parallel_flows, c.c_conds)):
        y = pool_layer(h)
        B, _, H, W = y.shape
        cond = positionalencoding2d(c_cond, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
        z, jac = parallel_flow(y, [cond, ])
        z_list.append(z)
        parallel_jac_list.append(jac)

    z_list, fuse_jac = fusion_flow(z_list)
    jac = fuse_jac + sum(parallel_jac_list)

    return z_list, jac


def model_forward_with_seg(c, extractor, parallel_flows, fusion_flow, image, segment):
    h_list = extractor(image)
    if c.pool_type == 'avg':
        pool_layer = nn.AvgPool2d(3, 2, 1)
    elif c.pool_type == 'max':
        pool_layer = nn.MaxPool2d(3, 2, 1)
    else:
        pool_layer = nn.Identity()

    z_list = []
    parallel_jac_list = []
    for idx, (h, parallel_flow, c_cond) in enumerate(zip(h_list, parallel_flows, c.c_conds)):
        y = pool_layer(h)
        B, _, H, W = y.shape
        if c.seg_enable and idx == 0:
            cond = segment + positionalencoding2d(c_cond, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
        else:
            cond = positionalencoding2d(c_cond, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
        z, jac = parallel_flow(y, [cond, ])
        z_list.append(z)
        parallel_jac_list.append(jac)

    z_list, fuse_jac = fusion_flow(z_list)
    jac = fuse_jac + sum(parallel_jac_list)

    return z_list, jac


def train_meta_epoch(c, epoch, loader, extractor, parallel_flows, fusion_flow, params, optimizer, warmup_scheduler, decay_scheduler, scaler=None):
    parallel_flows = [parallel_flow.train() for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.train()

    for sub_epoch in range(c.sub_epochs):
        epoch_loss = 0.
        image_count = 0
        if c.seg_enable:
            for idx, (image, _, _, segment) in enumerate(loader):
                optimizer.zero_grad()
                image = image.to(c.device)
                segment = segment.to(c.device)
                if scaler:
                    with autocast():
                        z_list, jac = model_forward_with_seg(c, extractor, parallel_flows, fusion_flow, image, segment)
                        loss = 0.
                        for z in z_list:
                            loss += 0.5 * torch.sum(z ** 2, (1, 2, 3))
                        loss = loss - jac
                        loss = loss.mean()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, 2)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    z_list, jac = model_forward_with_seg(c, extractor, parallel_flows, fusion_flow, image, segment)
                    loss = 0.
                    for z in z_list:
                        loss += 0.5 * torch.sum(z ** 2, (1, 2, 3))
                    loss = loss - jac
                    loss = loss.mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 2)
                    optimizer.step()
                epoch_loss += t2np(loss)
                image_count += image.shape[0]
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            if warmup_scheduler:
                warmup_scheduler.step()
            if decay_scheduler:
                decay_scheduler.step()

            mean_epoch_loss = epoch_loss / image_count
            print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
                  'Epoch {:d}.{:d} train loss: {:.3e}\tlr={:.2e}'.format(
                      epoch, sub_epoch, mean_epoch_loss, lr))
        else:
            for idx, (image, _, _) in enumerate(loader):
                optimizer.zero_grad()
                image = image.to(c.device)
                if scaler:
                    with autocast():
                        z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)
                        loss = 0.
                        for z in z_list:
                            loss += 0.5 * torch.sum(z**2, (1, 2, 3))
                        loss = loss - jac
                        loss = loss.mean()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, 2)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)
                    loss = 0.
                    for z in z_list:
                        loss += 0.5 * torch.sum(z**2, (1, 2, 3))
                    loss = loss - jac
                    loss = loss.mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 2)
                    optimizer.step()
                epoch_loss += t2np(loss)
                image_count += image.shape[0]
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            if warmup_scheduler:
                warmup_scheduler.step()
            if decay_scheduler:
                decay_scheduler.step()

            mean_epoch_loss = epoch_loss / image_count
            print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
                'Epoch {:d}.{:d} train loss: {:.3e}\tlr={:.2e}'.format(
                    epoch, sub_epoch, mean_epoch_loss, lr))
        

def inference_meta_epoch(c, epoch, loader, extractor, parallel_flows, fusion_flow):
    parallel_flows = [parallel_flow.eval() for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.eval()
    epoch_loss = 0.
    image_count = 0
    gt_label_list = list()
    gt_mask_list = list()
    outputs_list = [list() for _ in parallel_flows]
    size_list = []
    start = time.time()
    with torch.no_grad():
        if c.seg_enable:
            for idx, (image, label, mask, segment) in enumerate(loader):
                image = image.to(c.device)
                segment = segment.to(c.device)
                gt_label_list.extend(t2np(label))
                gt_mask_list.extend(t2np(mask))

                z_list, jac = model_forward_with_seg(c, extractor, parallel_flows, fusion_flow, image, segment)

                loss = 0.
                for lvl, z in enumerate(z_list):
                    if idx == 0:
                        size_list.append(list(z.shape[-2:]))
                    logp = - 0.5 * torch.mean(z ** 2, 1)
                    outputs_list[lvl].append(logp)
                    loss += 0.5 * torch.sum(z ** 2, (1, 2, 3))

                loss = loss - jac
                loss = loss.mean()
                epoch_loss += t2np(loss)
                image_count += image.shape[0]

            mean_epoch_loss = epoch_loss / image_count
            fps = len(loader.dataset) / (time.time() - start)
            print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
                  'Epoch {:d}   test loss: {:.3e}\tFPS: {:.1f}'.format(
                      epoch, mean_epoch_loss, fps))
        else:
            for idx, (image, label, mask) in enumerate(loader):
                image = image.to(c.device)
                gt_label_list.extend(t2np(label))
                gt_mask_list.extend(t2np(mask))

                z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)

                loss = 0.
                for lvl, z in enumerate(z_list):
                    if idx == 0:
                        size_list.append(list(z.shape[-2:]))
                    logp = - 0.5 * torch.mean(z**2, 1)
                    outputs_list[lvl].append(logp)
                    loss += 0.5 * torch.sum(z**2, (1, 2, 3))

                loss = loss - jac
                loss = loss.mean()
                epoch_loss += t2np(loss)
                image_count += image.shape[0]

            mean_epoch_loss = epoch_loss / image_count
            fps = len(loader.dataset) / (time.time() - start)
            print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
                'Epoch {:d}   test loss: {:.3e}\tFPS: {:.1f}'.format(
                    epoch, mean_epoch_loss, fps))

    return gt_label_list, gt_mask_list, outputs_list, size_list


def train(c):
    
    if c.wandb_enable:
        wandb.finish()
        wandb_run = wandb.init(
            project='65001-msflow', 
            group=c.version_name,
            name=c.class_name)
    # llw 11/25
    if c.dataset == 'mvtec':
        Dataset = MVTecDataset
    elif c.dataset == 'visa':
        Dataset = VisADataset
    elif c.dataset == 'miad':
        Dataset = MIADDataset

    # Dataset = MVTecDataset if c.dataset == 'mvtec' else VisADataset

    train_dataset = Dataset(c, is_train=True)
    test_dataset  = Dataset(c, is_train=False)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=c.workers, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, num_workers=c.workers, pin_memory=True)

    # llw 11/28: train --> 随机采样50%
    if c.dataset == 'miad':
        Coff_sample_tr = 0.5  # 采样比例
        N_sample_tr = int(len(train_dataset) * Coff_sample_tr)

        random_indices_tr = torch.randperm(len(train_dataset))[:N_sample_tr]  # 从索引列表中随机选择 N 个索引
        sampler_tr = SubsetRandomSampler(random_indices_tr)  # 创建一个子集采样器，使用随机选择的索引
        # 创建 DataLoader，使用采样器来限制读取的样本数
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, sampler=sampler_tr, shuffle=False,
                                                  num_workers=c.workers, pin_memory=False)

        # llw 11/26: 在MIAD上采样全部5000张进行后处理会CUDA OOM --> 随机采样500
        N_sample = 500  # 想要使用的样本数
        indices = list(range(len(test_dataset)))  # 创建一个索引列表，包含从 0 到 len(test_dataset)-1 的所有索引
        random_indices = torch.randperm(len(test_dataset))[:N_sample]  # 从索引列表中随机选择 N 个索引
        sampler = SubsetRandomSampler(random_indices)  # 创建一个子集采样器，使用随机选择的索引
        # 创建 DataLoader，使用采样器来限制读取的样本数
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, sampler=sampler, shuffle=False, num_workers=c.workers, pin_memory=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=c.workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, num_workers=c.workers, pin_memory=True)

    extractor, output_channels = build_extractor(c)
    extractor = extractor.to(c.device).eval()
    if c.wtconv_enable:
        parallel_flows, fusion_flow = build_flow_model_wtconv(c, output_channels)
    else:
        parallel_flows, fusion_flow = build_msflow_model(c, output_channels)
    parallel_flows = [parallel_flow.to(c.device) for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.to(c.device)
    # if c.wandb_enable:
    #     for idx, parallel_flow in enumerate(parallel_flows):
    #         wandb.watch(parallel_flow, log='all', log_freq=100, idx=idx)
    #     wandb.watch(fusion_flow, log='all', log_freq=100, idx=len(parallel_flows))
    params = list(fusion_flow.parameters())
    for parallel_flow in parallel_flows:
        params += list(parallel_flow.parameters())
        
    optimizer = torch.optim.Adam(params, lr=c.lr)
    if c.amp_enable:
        scaler = GradScaler()

    det_auroc_obs = Score_Observer('Det.AUROC', c.meta_epochs)
    loc_auroc_obs = Score_Observer('Loc.AUROC', c.meta_epochs)
    loc_pro_obs = Score_Observer('Loc.PRO', c.meta_epochs)

    start_epoch = 0
    if c.mode == 'test':
        start_epoch = load_weights(parallel_flows, fusion_flow, c.eval_ckpt)
        epoch = start_epoch + 1
        gt_label_list, gt_mask_list, outputs_list, size_list = inference_meta_epoch(c, epoch, test_loader, extractor, parallel_flows, fusion_flow)

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)
        # best_det_auroc, best_loc_auroc, best_loc_pro = eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, c.pro_eval)
        # llw 2024.12.3 保存概率图
        if c.save_enable:
            save_img(anomaly_score_map_mul)
        return
    
    if c.resume:
        last_epoch = load_weights(parallel_flows, fusion_flow, os.path.join(c.ckpt_dir, 'last.pt'), optimizer)
        start_epoch = last_epoch + 1
        print('Resume from epoch {}'.format(start_epoch))

    if c.lr_warmup and start_epoch < c.lr_warmup_epochs:
        if start_epoch == 0:
            start_factor = c.lr_warmup_from
            end_factor = 1.0
        else:
            start_factor = 1.0
            end_factor = c.lr / optimizer.state_dict()['param_groups'][0]['lr']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=(c.lr_warmup_epochs - start_epoch)*c.sub_epochs)
    else:
        warmup_scheduler = None

    mile_stones = [milestone - start_epoch for milestone in c.lr_decay_milestones if milestone > start_epoch]

    if mile_stones:
        decay_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, c.lr_decay_gamma)
    else:
        decay_scheduler = None

    for epoch in range(start_epoch, c.meta_epochs):
        print()
        train_meta_epoch(c, epoch, train_loader, extractor, parallel_flows, fusion_flow, params, optimizer, warmup_scheduler, decay_scheduler, scaler if c.amp_enable else None)

        gt_label_list, gt_mask_list, outputs_list, size_list = inference_meta_epoch(c, epoch, test_loader, extractor, parallel_flows, fusion_flow)

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)

        if c.pro_eval and (epoch > 0 and epoch % c.pro_eval_interval == 0):
            pro_eval = True
        else:
            pro_eval = False

        det_auroc, loc_auroc, loc_pro_auc, \
            best_det_auroc, best_loc_auroc, best_loc_pro = \
                eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, pro_eval)

        if c.wandb_enable:
            wandb_run.log(
                {
                    'Det.AUROC': det_auroc,
                    'Loc.AUROC': loc_auroc,
                    'Loc.PRO': loc_pro_auc
                },
                step=epoch
            )

        if c.no_save:
            continue
        else:
            save_weights(epoch, parallel_flows, fusion_flow, 'last', c.ckpt_dir, optimizer)
            if best_det_auroc and c.mode == 'train':
                save_weights(epoch, parallel_flows, fusion_flow, 'best_det', c.ckpt_dir)
            if best_loc_auroc and c.mode == 'train':
                save_weights(epoch, parallel_flows, fusion_flow, 'best_loc_auroc', c.ckpt_dir)
            if best_loc_pro and c.mode == 'train':
                save_weights(epoch, parallel_flows, fusion_flow, 'best_loc_pro', c.ckpt_dir)
        
