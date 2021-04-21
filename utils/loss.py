import torch.nn as nn
import torch.nn.functional as F
import torch

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


def one_hot(index, classes):
    # index is not flattened (pypass ignore) ############
    # size = index.size()[:1] + (classes,) + index.size()[1:]
    # view = index.size()[:1] + (1,) + index.size()[1:]
    #####################################################
    # index is flatten (during ignore) ##################
    size = index.size()[:1] + (classes,)
    view = index.size()[:1] + (1,)
    #####################################################

    # mask = torch.Tensor(size).fill_(0).to(device)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7, size_average=True, nclass=7, one_hot=True, ignore=None, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        self.one_hot = one_hot
        self.ignore = ignore

        if weight is not None:
            self.weight = torch.zeros(1, nclass).cuda()
            if ignore is not None:
                self.weight[:, 1:] =  weight[:]
            else:
                self.weight[:, :] = weight[:]
        else:
            self.weight = None

    def forward(self, input, target, sp_weight=None, lamda=0.15):
        '''
        only support ignore at 0
        '''
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        target = target.view(-1)
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]

        if self.one_hot: target = one_hot(target, input.size(1))
        if self.one_hot:
            probs = F.softmax(input, dim=1)
            if self.weight is not None:
                probs = (probs * target * self.weight).sum(1)
            else:
                probs = (probs * target).sum(1)
        else:
            probs = F.sigmoid(input)
            probs = (probs * target.float().unsqueeze(dim=-1)).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        # sample_loss = batch_loss.view(B, -1).sum(-1)
        # if sp_weight is not None:
        #     a = torch.exp(torch.Tensor([-1 / lamda])).cuda()
        #     batch_loss = sample_loss * sp_weight + torch.log((1 + a - sp_weight) ** (1 + a - sp_weight)) + \
        #                  torch.log(sp_weight ** sp_weight) - sp_weight / lamda
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss, 0


class SoftCrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss2d, self).__init__()

    def forward(self, inputs, targets):
        loss = 0
        inputs = -F.log_softmax(inputs, dim=1)
        for index in range(inputs.size()[0]):
            loss += F.conv2d(inputs[range(index, index+1)], targets[range(index, index+1)])/(targets.size()[2] *
                                                                                             targets.size()[3])
        return loss