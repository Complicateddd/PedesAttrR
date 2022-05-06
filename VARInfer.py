import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pickle

from dataset.augmentation import get_transform
# from dataset.multi_label.coco import COCO14
from metrics.pedestrian_metrics import get_pedestrian_metrics
from models.model_factory import build_backbone, build_classifier

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import cfg, update_config
from dataset.pedes_attr.pedes import PedesAttr
from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
from models.base_block import FeatClassifier
# from models.model_factory import model_dict, classifier_dict

from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool, time_str
from models.backbone import swin_transformer, resnet, bninception
# from models.backbone.tresnet import tresnet
from losses import bceloss, scaledbceloss

from dataset.hehuang.PAR import HehuangVARTest
import csv

set_seed(605)


def main(cfg, args):
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

    rootpath = '/media/ubuntu/data/hehuang/car'

    valid_set = HehuangVARTest(root =rootpath,phase=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
                              target_transform=None)

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

    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    model = get_reload_weight(model_dir, model, pth='ckpt_max_2022-02-25_17:28:22.pth')

    model.eval()
    preds_probs = []
    gt_list = []
    path_list = []

    attn_list = []


    f = open('result.csv','w',encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow([
        'id',
        'type',
        ])

    multi={
    0:'van',
    1:'car',
    2:'truck',
    3:'suv',
    }

    with torch.no_grad():
        for step, (imgs, gt_label, imgname) in enumerate(tqdm(valid_loader)):
            
            imgs = imgs.cuda()

            gt_label = gt_label.cuda()

            valid_logits, attns = model(imgs, gt_label)

            # valid_probs = valid_logits[0].cpu().numpy()
            valid_probs = torch.sigmoid(valid_logits[0]).cpu().numpy()

            for i in range(valid_probs.shape[0]):

                write_line = []

                row = valid_probs[i]
                name = imgname[i]

                type_ = row

                
                write_type_ = multi[np.argmax(type_)]
            

                write_line.append(name)
                write_line.append(write_type_)

                # t = np.exp(color)
                # color = np.exp(color) / np.sum(t)


                csv_writer.writerow(write_line)
    f.close()

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
