import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pickle

from dataset.augmentation import get_transform
# from dataset.multi_label.coco import COCO14
from metrics.pedestrian_metrics import get_pedestrian_metrics
# from models.model_factory import build_backbone, build_classifier
from models.model_factory import build_loss, build_classifier, build_backbone

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import cfg, update_config
from dataset.pedes_attr.pedes import PedesAttr
from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
# from models.base_block import FeatClassifier,PoolingSwin
# from models.model_factory import model_dict, classifier_dict
from models.base_block import FeatClassifier,SwinALM,PoolingSwin,FPNNeckSwin,CSRAClassifierFPN,PoolingSwin2
from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool, time_str
from models.backbone import swin_transformer, resnet, bninception
# from models.backbone.tresnet import tresnet
from losses import bceloss, scaledbceloss

from dataset.hehuang.PAR import HehuangPARTest,HehuangPAR
import csv

from models.base_block import FeatClassifier,SwinALM,PoolingSwin,FPNNeckSwin,CSRAClassifierFPN,PoolingSwin2
from batch_engine import valid_trainer

set_seed(605)


def main(cfg, args):

    args.distributed = False
    args.local_rank = 0

    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)

    train_tsfm, valid_tsfm = get_transform(cfg)
    print(valid_tsfm)

    # if cfg.DATASET.TYPE == 'multi_label':
    #     train_set = COCO14(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=train_tsfm,
    #                        target_transform=cfg.DATASET.TARGETTRANSFORM)

    #     valid_set = COCO14(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
    #                        target_transform=cfg.DATASET.TARGETTRANSFORM)
    # else:
    #     train_set = PedesAttr(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=valid_tsfm,
    #                           target_transform=cfg.DATASET.TARGETTRANSFORM)
    #     valid_set = PedesAttr(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
    #                           target_transform=cfg.DATASET.TARGETTRANSFORM)

    rootpath = '/media/ubuntu/data/hehuang/train2'

    multi_scale_test = True

    valid_set = HehuangPAR(root =rootpath,phase=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
                              target_transform=None,multi_scale = multi_scale_test)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # print(f'{cfg.DATASET.TRAIN_SPLIT} set: {len(train_loader.dataset)}, '
    #       f'{cfg.DATASET.TEST_SPLIT} set: {len(valid_loader.dataset)}, '
    #       f'attr_num : {train_set.attr_num}')

    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)


    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=valid_set.attr_num,
        c_in=c_output,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale =cfg.CLASSIFIER.SCALE
    )

    # if 'fpn' in cfg.BACKBONE.TYPE:
    #     model = PoolingSwin(fpn_backbone = backbone , classifier = classifier,num_classes = valid_set.attr_num,bn_wd=cfg.TRAIN.BN_WD)
    # else:
    #     model = FeatClassifier(backbone, classifier, bn_wd=cfg.TRAIN.BN_WD)

    if cfg.PIPELINE == 'Swin_Neck_Part_CSRA':
        model = FPNNeckSwin(backbone, classifier, bn_wd=cfg.TRAIN.BN_WD)
    elif cfg.PIPELINE == 'Swin_Neck_Pooling_CSRA':
        model = PoolingSwin(fpn_backbone = backbone , classifier = classifier,num_classes = train_set.attr_num,bn_wd=cfg.TRAIN.BN_WD)
    elif cfg.PIPELINE == 'Swin_Neck_Pooling2_CSRA':
        model = PoolingSwin2(fpn_backbone = backbone , classifier = classifier,num_classes = train_set.attr_num,bn_wd=cfg.TRAIN.BN_WD)
    elif cfg.PIPELINE == 'Swin_FPN_ALM':
        model = SwinALM(backbone, classifier, bn_wd=cfg.TRAIN.BN_WD)
    else:
        model = FeatClassifier(backbone, classifier, bn_wd=cfg.TRAIN.BN_WD)

    # model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    model = get_reload_weight(model_dir, model, pth='ckpt_max_2022-03-22_20:35:12.pth')

    loss_weight = cfg.LOSS.LOSS_WEIGHT


    criterion = build_loss(cfg.LOSS.TYPE)(
        sample_weight=None, scale=cfg.CLASSIFIER.SCALE, size_sum=cfg.LOSS.SIZESUM, tb_writer=None)
    criterion = criterion.cuda()

    valid_loss, valid_gt, valid_probs, valid_imgs, valid_logits, valid_loss_mtr = valid_trainer(cfg, args, 1, model, valid_loader, criterion, loss_w=cfg.LOSS.LOSS_WEIGHT,multi_scale = multi_scale_test)

    valid_result = get_pedestrian_metrics(valid_gt, valid_probs, index=None, cfg=cfg)

    if args.local_rank == 0:

        print(f'Evaluation on test set, valid losses {valid_loss}\n',
              'ma: {:.4f}, label_f1: {:.4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_f1),
                  np.mean(valid_result.label_pos_recall),
                  np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))
        print(valid_result.label_f1)
        print(f'{time_str()}')
        print('-' * 60)
    # model.eval()
    # preds_probs = []
    # gt_list = []
    # path_list = []

    # attn_list = []


    # f = open('result.csv','w',encoding='utf-8',newline='')
    # csv_writer = csv.writer(f)
    # csv_writer.writerow([
    #     'name',
    #     'upperLength',
    #     'clothesStyles',
    #     'hairStyles',
    #     'upperBlack',
    #     'upperBrown',
    #     'upperBlue',
    #     'upperGreen',
    #     'upperGray',
    #     'upperOrange',
    #     'upperPink',
    #     'upperPurple',
    #     'upperRed',
    #     'upperWhite',
    #     'upperYellow'])

    # multi={
    #     0:'Short',
    #     1:'Long',
    #     2:'middle',
    #     3:'Bald',
    #     4:'ShortSleeve',
    #     5:'LongSleeve',
    #     6:'NoSleeve',
    #     7:'multicolour',
    #     8:'Solidcolor',
    #     9:'lattice',
    # }

    # with torch.no_grad():
    #     for step, (imgs, gt_label, imgname) in enumerate(tqdm(valid_loader)):
            
    #         imgs = imgs.cuda()

    #         gt_label = gt_label.cuda()

    #         valid_logits, attns = model(imgs, gt_label)

    #         # valid_probs = valid_logits[0].cpu().numpy()
    #         if type(valid_logits)==type([]):
    #             valid_probs = torch.max(torch.max(torch.max(valid_logits[0],valid_logits[1]),valid_logits[2]),valid_logits[3])
    #             valid_probs = torch.sigmoid(valid_probs).cpu().numpy()
    #         else:
    #             valid_probs = torch.sigmoid(valid_logits[0]).cpu().numpy()

    #         for i in range(valid_probs.shape[0]):

    #             write_line = []

    #             row = valid_probs[i]
    #             name = imgname[i]

    #             hairStyles = row[:3]
    #             upperLength = row[3:7]
    #             clothesStyles = row[7:10]
    #             color = row[10:]

                
    #             write_hairStyles = multi[np.argmax(hairStyles)]
    #             write_upperLength = multi[np.argmax(upperLength) + 3]
    #             write_clothesStyles = multi[np.argmax(clothesStyles) + 7]

    #             write_line.append(name)
    #             write_line.append(write_upperLength)
    #             write_line.append(write_clothesStyles)
    #             write_line.append(write_hairStyles)
                
                

    #             # t = np.exp(color)
    #             # color = np.exp(color) / np.sum(t)

    #             for ele in color:
    #                 # e = round(float(ele),1)
    #                 if ele>0.5:
    #                     write_line.append(str(1))
    #                 else:
    #                     write_line.append('')

    #             csv_writer.writerow(write_line)
    #             # print(write_hairStyles)
    #             # print(write_upperLength)
    #             # print(write_clothesStyles)
    #             # print(row)
    #             # print(name)
    #             # print(write_line)
                
    # f.close()

    # gt_label = np.concatenate(gt_list, axis=0)
    # preds_probs = np.concatenate(preds_probs, axis=0)



    

def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,
    )
    parser.add_argument("--debug", type=str2bool, default="true")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()
    update_config(cfg, args)

    main(cfg, args)
