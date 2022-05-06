import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import LOSSES
from tools.function import ratio2weight


@LOSSES.register("bceloss")
class BCELoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=None, tb_writer=None):
        super(BCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None

    def forward(self, logits, targets,targets_softmax = [None]):
        logits = logits[0]

        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask, self.sample_weight)

            loss_m = (loss_m * sample_weight.cuda())

        # losses = loss_m.sum(1).mean() if self.size_sum else loss_m.mean()
        loss = loss_m.sum(1).mean() if self.size_sum else loss_m.sum()

        return [loss], [loss_m]


@LOSSES.register("fpn_bceloss")
class FPNBCELoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=None, tb_writer=None):
        super(FPNBCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None

    def forward(self, logits_fpn, targets,targets_softmax = [None]):

        loss_record = []
        # print(logits_fpn[-1])

        for i in range(len(logits_fpn)):

            logits = logits_fpn[i]

            # print(logits.shape)

            if self.smoothing is not None:
                targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

            loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

            targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
            if self.sample_weight is not None:
                sample_weight = ratio2weight(targets_mask, self.sample_weight)

                loss_m = (loss_m * sample_weight.cuda())

            # losses = loss_m.sum(1).mean() if self.size_sum else loss_m.mean()
            loss = loss_m.sum(1).mean() if self.size_sum else loss_m.sum()

            loss_record.append(loss)

        return loss_record, None



from torch.autograd import Variable
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py
@LOSSES.register("FocalLoss")
class FocalLoss(nn.Module):
    def __init__(self, sample_weight=None, size_sum=True, scale=None, tb_writer=None,
        alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.smoothing = None


    def forward(self, logits, targets,targets_softmax = [None]):

        logits = logits[0]

        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        pt = torch.exp(-BCE_loss)
        
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        F_loss = F_loss.sum(1).mean()

        # print(F_loss)

        return [F_loss],None
       

@LOSSES.register("partbceloss")
class PartBCELoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=None, tb_writer=None):
        super(PartBCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits_part_fpn, targets ,targets_softmax = [None]):

        logit1,logit2,logit3,logit4 = logits_part_fpn
        
        loss_1 = self.cross_entropy(logit1,targets_softmax[0])
        loss_2 = self.cross_entropy(logit2,targets_softmax[1])
        loss_3 = self.cross_entropy(logit3,targets_softmax[2])

        # print(logit1)
        # print(logit2)
        # print(logit3)

        # logits = torch.cat((logits_part_fpn[0],torch.cat((logits_part_fpn[2],logits_part_fpn[1]),1)),1)

        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        loss_m = F.binary_cross_entropy_with_logits(logit4, targets[:,10:], reduction='none')

        # targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        # if self.sample_weight is not None:
        #     sample_weight = ratio2weight(targets_mask, self.sample_weight)

        #     loss_m = (loss_m * sample_weight.cuda())

        # losses = loss_m.sum(1).mean() if self.size_sum else loss_m.mean()
        loss_m = loss_m.sum(1).mean() if self.size_sum else loss_m.sum() 


        # loss = loss_m+loss_1+loss_2
        # print(loss_m,loss_1,loss_2)

        return [loss_m,loss_1,loss_2,loss_3], None
