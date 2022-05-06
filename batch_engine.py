import math
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tools.distributed import reduce_tensor
from tools.utils import AverageMeter, to_scalar, time_str
import torch.nn as nn

def logits4pred(criterion, logits_list):
    # print(criterion.__class__.__name__.lower())
    if criterion.__class__.__name__.lower() in ['bceloss','focalloss']:
        logits = logits_list[0]
        probs = logits.sigmoid()
    elif criterion.__class__.__name__.lower() in ['fpnbceloss']:
        logits = torch.max(torch.max(torch.max(logits_list[0],logits_list[1]),logits_list[2]),logits_list[3])
        probs = logits.sigmoid()
    elif criterion.__class__.__name__.lower() in ['partbceloss']:
        # logits = torch.cat((logits_list[0],torch.cat((logits_list[2],logits_list[1]),1)),1)
        # probs = logits.sigmoid()
        soft = nn.Softmax(-1)

        # for ele in logits_list:
        #     print('#',ele.shape)

        softmax_1 = soft(logits_list[0])
        index_1 = softmax_1.max(-1,keepdim = True)[1]
        probs_1 = torch.zeros_like(softmax_1).scatter_(-1,index_1,1.0)
        # print(probs_1)

        softmax_2 = soft(logits_list[1])
        index_2 = softmax_2.max(-1,keepdim = True)[1]
        probs_2 = torch.zeros_like(softmax_2).scatter_(-1,index_2,1.0)
        # print(probs_1)
        softmax_3 = soft(logits_list[2])
        index_3 = softmax_3.max(-1,keepdim = True)[1]
        probs_3 = torch.zeros_like(softmax_3).scatter_(-1,index_3,1.0)

        # # print(logits_list[1][:,3:].shape)

        probs_4 = logits_list[3]
        # # print(probs_4.shape)
        probs_4=probs_4.sigmoid()

        probs = torch.cat((probs_1,probs_2,probs_3,probs_4),1)
        # print(probs.shape)
        # probs = probs_4

    else:
        assert False, f"{criterion.__class__.__name__.lower()} not exits"

    return probs, logits_list[0]


def batch_trainer(cfg, args, epoch, model, model_ema, train_loader, criterion, optimizer, loss_w=[1, ], scheduler=None):
    model.train()
    epoch_time = time.time()

    loss_meter = AverageMeter()
    subloss_meters = [AverageMeter() for i in range(len(loss_w))]

    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []
    preds_logits = []
    imgname_list = []
    loss_mtr_list = []

    lr = optimizer.param_groups[0]['lr']

    for step, (imgs, gt_label, imgname,target_softmax1,target_softmax2,target_softmax3,imgs2,imgs3,imgs4,imgs5) in enumerate(train_loader):
        iter_num = epoch * len(train_loader) + step

        batch_time = time.time()
        imgs, gt_label = imgs.cuda(), gt_label.cuda()

        target_softmax1,target_softmax2,target_softmax3 = target_softmax1.cuda(),target_softmax2.cuda(),target_softmax3.cuda()


        # print(target_softmax1,target_softmax2,target_softmax3)
        


        train_logits, feat = model(imgs, None)

        # print(target_softmax)
        loss_list, loss_mtr = criterion(train_logits, gt_label ,[target_softmax1,target_softmax2,target_softmax3])

        train_loss = 0

        for i, l in enumerate(loss_w):
            train_loss += loss_list[i] * l

        optimizer.zero_grad()
        train_loss.backward()

        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print("NO " + name)
        #     else:
        #         print("YES " + name)

        if cfg.TRAIN.CLIP_GRAD:
            clip_grad_norm_(model.parameters(), max_norm=10.0)  # make larger learning rate works


        optimizer.step()

        if cfg.TRAIN.LR_SCHEDULER.TYPE == 'annealing_cosine' and scheduler is not None:
            scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()

        if len(loss_list) > 1:
            for i, meter in enumerate(subloss_meters):
                meter.update(
                    to_scalar(reduce_tensor(loss_list[i], args.world_size)
                              if args.distributed else loss_list[i]))

        loss_meter.update(to_scalar(reduce_tensor(train_loss, args.world_size) if args.distributed else train_loss))
        # print('#',gt_label)
        train_probs, train_logits = logits4pred(criterion, train_logits)


        # print(gt_label)
        # print(target_softmax1.shape)
        # print(target_softmax2.shape)
        # print(target_softmax3.shape)




        gt_list.append(gt_label.cpu().numpy())
        preds_probs.append(train_probs.detach().cpu().numpy())
        preds_logits.append(train_logits.detach().cpu().numpy())

        imgname_list.append(imgname)

        log_interval = 25

        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            if args.local_rank == 0:
                print(f'{time_str()}, '
                      f'Step {step}/{batch_num} in Ep {epoch}, '
                      f'LR: [{optimizer.param_groups[0]["lr"]:.1e}, {optimizer.param_groups[0]["lr"]:.1e}] '
                      f'Time: {time.time() - batch_time:.2f}s , '
                      f'train_loss: {loss_meter.avg:.4f}, ')

                print([f'{meter.avg:.4f}' for meter in subloss_meters])

            # break

    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    if args.local_rank == 0:
        print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')

    return train_loss, gt_label, preds_probs, imgname_list, preds_logits, loss_mtr_list


def valid_trainer(cfg, args, epoch, model, valid_loader, criterion, loss_w=[1, ],multi_scale = False):
    model.eval()
    loss_meter = AverageMeter()
    subloss_meters = [AverageMeter() for i in range(len(loss_w))]

    preds_probs = []
    preds_logits = []
    gt_list = []
    imgname_list = []
    loss_mtr_list = []

    with torch.no_grad():
        for step, (imgs, gt_label, imgname,target_softmax1,target_softmax2,target_softmax3,imgs2,imgs3,imgs4,imgs5) in enumerate(tqdm(valid_loader)):
            
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0

            target_softmax1,target_softmax2,target_softmax3 = target_softmax1.cuda(),target_softmax2.cuda(),target_softmax3.cuda()

            if multi_scale:
                imgs = imgs.cuda()
                imgs2 = imgs2.cuda()
                imgs3 = imgs3.cuda()
                imgs4 = imgs4.cuda()
                imgs5 = imgs5.cuda()


                valid_logits_1, feat = model(imgs, gt_label)
                valid_logits_2, feat = model(imgs2, gt_label)
                valid_logits_3, feat = model(imgs3, gt_label)
                valid_logits_4, feat = model(imgs4, gt_label)
                valid_logits_5, feat = model(imgs5, gt_label)

                valid_probs1 = logits4pred(criterion, valid_logits_1)[0]
                valid_probs2 = logits4pred(criterion, valid_logits_2)[0]
                valid_probs3 = logits4pred(criterion, valid_logits_3)[0]
                valid_probs4 = logits4pred(criterion, valid_logits_4)[0]
                valid_probs5 = logits4pred(criterion, valid_logits_5)[0]

                valid_probs = torch.zeros_like(valid_logits_1[0]).cuda()
                valid_probs[valid_probs1>0.5]+=1
                valid_probs[valid_probs1<=0.5]-=1
                valid_probs[valid_probs2>0.5]+=1
                valid_probs[valid_probs2<=0.5]-=1
                valid_probs[valid_probs3>0.5]+=1
                valid_probs[valid_probs3<=0.5]-=1
                valid_probs[valid_probs4>0.5]+=1
                valid_probs[valid_probs4<=0.5]-=1
                valid_probs[valid_probs5>0.5]+=1
                valid_probs[valid_probs5<=0.5]-=1
                valid_probs[valid_probs<1]=0
                # valid_probs = valid_probs1

                loss_list, loss_mtr = criterion(valid_logits_1, gt_label ,[target_softmax1,target_softmax2,target_softmax3])
                valid_loss = 0
                for i, l in enumerate(loss_list):
                    valid_loss += loss_w[i] * l
                valid_logits = valid_probs

            else:

                valid_logits, feat = model(imgs, gt_label)
                loss_list, loss_mtr = criterion(valid_logits, gt_label ,[target_softmax1,target_softmax2,target_softmax3])

                valid_loss = 0
                for i, l in enumerate(loss_list):
                    valid_loss += loss_w[i] * l

                valid_probs, valid_logits = logits4pred(criterion, valid_logits)

            
            preds_probs.append(valid_probs.cpu().numpy())
            preds_logits.append(valid_logits.cpu().numpy())

            if len(loss_list) > 1:
                for i, meter in enumerate(subloss_meters):
                    meter.update(
                        to_scalar(reduce_tensor(loss_list[i], args.world_size) if args.distributed else loss_list[i]))
            loss_meter.update(to_scalar(reduce_tensor(valid_loss, args.world_size) if args.distributed else valid_loss))

            torch.cuda.synchronize()

            imgname_list.append(imgname)

    valid_loss = loss_meter.avg

    if args.local_rank == 0:
        print([f'{meter.avg:.4f}' for meter in subloss_meters])

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    preds_logits = np.concatenate(preds_logits, axis=0)

    return valid_loss, gt_label, preds_probs, imgname_list, preds_logits, loss_mtr_list
