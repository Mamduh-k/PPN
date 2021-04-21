#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from models.PPN import adaLG
from utils.metrics import ConfusionMatrix
from utils.handle_data import get_sub_batch,get_local,global2patch,masks_transform,images_transform,resize, crop_rein_patch
torch.backends.cudnn.deterministic = True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def collate(batch):
    image_origin = [b['image_origin'] for b in batch]  # w, h
    label_origin = [b['label_origin'] for b in batch]
    id = [b['id'] for b in batch]
    index = [b['index'] for b in batch]
    res = [b['res'] for b in batch]
    image = [b['image'] for b in batch]  # w, h
    label = [b['label'] for b in batch]
    return {'image_origin': image_origin, 'label_origin': label_origin,
            'image': image, 'label': label, 'res': res, 'id': id, 'index': index}

def imwrite_tensor_RGB(tensor, filepath):
    import matplotlib.pyplot as plt
    unloader = transforms.ToPILImage()
    tensor = tensor*255
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    # plt.imshow(image)
    # plt.savefig(filepath)
    # image = image.convert("RGB")
    # cv2.imwrite(filepath, image)
    image.save(filepath)

def mIoU(label, predict, classes):
    smooth = 0.00000001
    confusion_matrix = torch.zeros(label.size()[0], classes, classes)
    b_index = 0
    for lt, lp in zip(label, predict):
        lt = lt.view(-1)
        lp = lp.view(-1)
        mask = (lt >= 0) & (lt < classes)
        hist = np.bincount(classes * lt[mask].int() + lp[mask].int(), minlength=classes**2).reshape(classes, classes)
        confusion_matrix[b_index] = torch.from_numpy(hist)
        b_index += 1

    b_hist = confusion_matrix
    b_intersect = b_hist[:, [i for i in range(classes)], [i for i in range(classes)]]
    b_union = b_hist.sum(dim=2) + b_hist.sum(dim=1) - b_intersect

    b_iou = (b_intersect + smooth) / (b_union + smooth)
    m_iou = torch.zeros(b_iou.size()[0])
    for i in range(b_iou.size()[0]):
        if len(b_iou[i][b_iou[i] < 1]) != 0:
            m_iou[i] = b_iou[i][b_iou[i] < 1].mean()
        else:
            m_iou[i] = 1.
    return m_iou #b_iou[:, 1:].mean(dim=-1) # b_mean_iou

def create_model_load_weights(n_class, model_path=None, path_weight=None, criterion_gscnn=None):
    model = adaLG(n_class)
    model = model.cuda()
    model = nn.DataParallel(model)
    if path_weight:
        path_weight = os.path.join(model_path, path_weight)
        print("load pre-trained weights from %s" % path_weight)
        weights = torch.load(path_weight)
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in weights.items() if k in state}
        state.update(pretrained_dict)
        model.load_state_dict(state, strict=True)
        print("load successfully!")

    return model
def get_optimizer(model, mode=1, learning_rate=2e-5):
    if mode == 0:
        # train local
        optimizer = torch.optim.Adam([
            {'params': model.module.resnet_l.parameters(), 'lr': learning_rate},
            {'params': model.module.fpn_l.parameters(), 'lr': learning_rate},
        ], weight_decay=5e-4)
    elif mode == 1:
        # train global
        optimizer = torch.optim.Adam([
            {'params': model.module.resnet_g.parameters(), 'lr': learning_rate},
            {'params': model.module.fpn_g.parameters(), 'lr': learning_rate},
        ], weight_decay=5e-4)
    elif mode == 2:
        # train rein classifier
        optimizer = torch.optim.Adam([
            {'params': model.module.rein_classifier.parameters(), 'lr': learning_rate},
            {'params': model.module.saliency_classifier.parameters(), 'lr': learning_rate},
            ],  weight_decay=5e-4)
    elif mode == 3:
        optimizer = torch.optim.Adam([
            {'params': model.module.resnet_g.parameters(), 'lr': learning_rate},
            {'params': model.module.fpn_g.parameters(), 'lr': learning_rate},
            {'params': model.module.resnet_l.parameters(), 'lr': learning_rate},
            {'params': model.module.fpn_l.parameters(), 'lr': learning_rate},
            {'params': model.module.saliency.parameters(), 'lr': learning_rate},
            {'params': model.module.rein_output_layer.parameters(), 'lr': learning_rate},
        ], weight_decay=5e-4)
    return optimizer

class Trainer(object):
    def __init__(self, criterion, n_class, size_g, size_p, size_lp, batch_size, sub_batch_size=6, margin=0, mode=1,
                 weight_l=None, weight_rein=None, lamda_l=0.15, lamda_rein=0.15, mu=0.9):
        self.criterion = criterion
        self.optimizer = None
        self.metrics_tea = ConfusionMatrix(n_class, batch_size)
        self.metrics_fuse = ConfusionMatrix(n_class, batch_size)
        self.n_class = n_class
        self.margin = margin
        self.size_g = size_g
        self.size_p = size_p
        self.size_lp = size_lp

        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.max_patch_side = 3000
        self.coordinates = None
        self.patch_n = None
        self.num = 16
        self.ratio = None
        self.mse = nn.MSELoss()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_mode(self, mode):
        self.mode = mode

    def set_loss(self, criterion):
        self.criterion = criterion

    def set_train(self, model):
        if self.mode == 1:
            model.module.resnet_g.train()
            model.module.fpn_g.train()
        if self.mode == 2:
            model.module.resnet_g.eval()
            model.module.fpn_g.eval()
            model.module.rein_classifier.train()
            model.module.saliency_classifier.train()
        if self.mode == 3:
            model.module.resnet_l.train()
            model.module.fpn_l.train()
            model.module.saliency.train()
            model.module.rein_output_layer.train()
            model.module.resnet_g.train()
            model.module.fpn_g.train()
            model.module.rein_classifier.eval()
            model.module.saliency_classifier.eval()
        if self.mode == 4:
            model.train()

    def get_scores(self):
        score_tea = self.metrics_tea.get_scores()
        score_fuse = self.metrics_fuse.get_scores()
        return np.mean(np.nan_to_num(score_tea["iou"][0:])), np.mean(np.nan_to_num(score_fuse["iou"][0:]))

    def reset_metrics(self):
        self.metrics_tea.reset()
        self.metrics_fuse.reset()

    def train(self, sample, model):
        images_glb, labels_glb, images_origin, labels_origin = sample['image'], sample['label'], sample['image_origin'], sample['label_origin']
        images_glb = torch.stack(images_glb, dim=0).cuda()
        labels_glb = torch.stack(labels_glb, dim=0).squeeze(1).cuda()

        if self.mode == 1:
            outputs_global = model.forward(images_glb)
            loss_g, _ = self.criterion(outputs_global, labels_glb)
            loss_g.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            outputs_global = outputs_global.argmax(1).detach()
            self.metrics_tea.update(labels_glb.cpu(), outputs_global.cpu())
            return loss_g

        if self.mode == 2:
            with torch.no_grad():
                outputs_global = model.forward(images_glb, patch_n=None, mode=1) # mode = 1 for fistly propagate to global
            b, c, h, w = outputs_global.size()
            batch_mIoU = mIoU(labels_glb.cpu(), outputs_global.clone().argmax(1).cpu(), c)
            # get patches coordinates
            if self.coordinates is None:
                self.coordinates, self.patch_n, self.ratio = global2patch(outputs_global.clone(), self.size_p) # patch_n number of patches in h and w
            # gt of classifier
            gt = torch.zeros((b, self.patch_n[0] * self.patch_n[1])).cuda()
            for i in range(len(self.coordinates)):
                top, left = self.coordinates[i][0], self.coordinates[i][1]
                top = int(top * h)
                left = int(left * w)
                down, right = top + self.size_p[0], left + self.size_p[1]
                patch_batch_mIoU = mIoU(labels_glb[:, top:down, left:right].cpu().contiguous(),
                                            outputs_global[:, :, top:down, left:right].clone().argmax(1).cpu().contiguous(), c)
                l = (patch_batch_mIoU < batch_mIoU) & (batch_mIoU - patch_batch_mIoU > self.margin)
                gt[:, i] = l[0:b]

            pred = model.forward(images_glb, patch_n=self.patch_n, mode=self.mode, glb_res=outputs_global.clone())
            loss = self.criterion(pred, gt.cuda())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            pred = (pred > 0.5).float()
            score = torch.sum(gt == pred).float() / (self.batch_size * self.patch_n[0] * self.patch_n[1])
            return loss, score

        if self.mode == 3:
            outputs_global = model.forward(images_glb, mode=1)
            loss_g, _ = self.criterion(outputs_global, labels_glb)  # total_loss += loss

            if self.coordinates is None:
                self.coordinates, self.patch_n, self.ratio = global2patch(outputs_global.clone(), self.size_p)

            with torch.no_grad():
                pred_rein = model.forward(images_glb, patch_n=self.patch_n, mode=2, glb_res=outputs_global.clone())
                pred_rein = (pred_rein > 0.5).int().cpu()

            selected_patch, selected_label, index = crop_rein_patch(images_origin, labels_origin, pred_rein, ratio=self.ratio,coords=self.coordinates)
            # torch.cuda.empty_cache()
            b, c, h, w = outputs_global.size()
            template = torch.zeros_like(outputs_global)
            for i in range(len(images_origin)):
                patches_var = images_transform(selected_patch[i][:])
                labels_patch_var = selected_label[i][:]
                _, _, h_p, w_p = patches_var.size()
                # if True:
                if h_p > self.max_patch_side or w_p > self.max_patch_side:
                    if h_p > w_p:
                        patches_var = F.interpolate(patches_var,
                                                    size=(
                                                        self.max_patch_side, int(self.max_patch_side / h_p * w_p)),
                                                    mode='bilinear', align_corners=True)
                    else:
                        patches_var = F.interpolate(patches_var,
                                                    size=(
                                                        int(self.max_patch_side / w_p * h_p), self.max_patch_side),
                                                    mode='bilinear', align_corners=True)
                labels_patch_var = masks_transform(resize(labels_patch_var, (patches_var.size()[3] // 4, patches_var.size()[2] // 4), label=True))  # train reinforce process

                # train reinforce process
                torch.cuda.empty_cache()
                global_need_rein = []
                # former_pmIoU = []
                for j in range(len(index[i])):
                    top, left = self.coordinates[index[i][j]]
                    top = int(np.round(top * h))
                    left = int(np.round(left * w))
                    down, right = top + self.size_p[0], left + self.size_p[1]
                    rein_patch = outputs_global[i:i + 1, :, top:down, left:right].detach()
                    global_need_rein.append(
                        F.interpolate(rein_patch, size=patches_var.size()[2:], mode='bilinear', align_corners=True))
                global_need_rein = F.softmax(torch.cat(global_need_rein, dim=0), dim=1)
                # print(global_need_rein.shape)
                global_rein = model.forward(global_need_rein, patch_inp=patches_var, mode=self.mode)

                if global_rein.size()[2] == labels_patch_var.size()[1] and global_rein.size()[3] == labels_patch_var.size()[2]:
                    loss_rein_p, _ = self.criterion(global_rein, labels_patch_var)
                else:
                    loss_rein_p, _ = self.criterion(F.interpolate(global_rein, labels_patch_var.size()[1:]), labels_patch_var)

                loss_rein_p.backward()
                global_rein_s = F.interpolate(global_rein, size=self.size_p, mode='bilinear', align_corners=True).detach_()

                for j in range(len(index[i])):
                    top, left = self.coordinates[index[i][j]]
                    top = int(np.round(top * h))
                    left = int(np.round(left * w))
                    down, right = top + self.size_p[0], left + self.size_p[1]
                    template[i:i + 1, :, top:down, left:right] = global_rein_s[j:j + 1]

            rein_global = model.forward(outputs_global, template=template, mode=self.mode)
            loss_rein_g, _ = self.criterion(rein_global, labels_glb)
            loss_g = 0.5 * loss_g + loss_rein_g
            loss_g.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.metrics_fuse.update(labels_glb.cpu(), rein_global.argmax(1).cpu())
            self.metrics_tea.update(labels_glb.cpu(), outputs_global.argmax(1).cpu())
            loss = loss_rein_p + loss_g
            return loss


class Evaluator(object):
    def __init__(self, n_class, size_g, size_p, batch_size, sub_batch_size=6, mode=1, test=False):
        self.metrics = ConfusionMatrix(n_class, batch_size)
        self.n_class = n_class
        self.size_g = size_g
        self.size_p = size_p
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.coordinates = None
        self.patch_n = None
        self.num = 16
        self.ratio = None
        self.max_patch_side = 3000

        self.metrics_tea = ConfusionMatrix(n_class, batch_size)
        self.metrics_fuse = ConfusionMatrix(n_class, batch_size)

    def set_mode(self, mode):
        self.mode = mode

    def get_scores(self):
        score_tea = self.metrics_tea.get_scores()
        score_fuse = self.metrics_fuse.get_scores()
        return np.mean(np.nan_to_num(score_tea["iou"][1:])), np.mean(np.nan_to_num(score_fuse["iou"][1:]))

    def reset_metrics(self):
        self.metrics_tea.reset()
        self.metrics_fuse.reset()

    def eval_test(self, sample, model):
        torch.cuda.empty_cache()
        with torch.no_grad():
            images_glb, labels_glb, images_origin, labels_origin = sample['image'], sample['label'], sample[
                'image_origin'], sample['label_origin']  # PIL images
            images_glb = torch.stack(images_glb, dim=0).cuda()
            labels_glb = torch.stack(labels_glb, dim=0).squeeze(1).cuda()
            if self.mode == 1:
                torch.cuda.empty_cache()
                outputs_global = model.forward(images_glb, mode=self.mode)
                outputs_global = outputs_global.argmax(1).detach()
                self.metrics_tea.update(labels_glb.cpu(), outputs_global.cpu())
            elif self.mode == 2:
                torch.cuda.empty_cache()
                outputs_global = model.forward(images_glb, mode=1)
                b, c, h, w = outputs_global.size()
                if self.coordinates is None:
                    self.coordinates, self.patch_n, self.ratio = global2patch(outputs_global.clone(), self.size_p)
                pred_rein = model.forward(images_glb, patch_n=self.patch_n, mode=2, glb_res=outputs_global.clone())
                pred_rein = (pred_rein > 0.5).int()
                # justify classifier
                batch_mIoU = mIoU(labels_glb.cpu(), outputs_global.argmax(1).cpu(), c)
                # gt of classifier
                gt = torch.zeros((b, self.patch_n[0] * self.patch_n[1])).cuda()
                for i in range(len(self.coordinates)):
                    top, left = self.coordinates[i][0], self.coordinates[i][1]
                    top = int(top * h)
                    left = int(left * w)
                    down, right = top + self.size_p[0], left + self.size_p[1]

                    patch_res = outputs_global[:, :, top:down, left:right].argmax(1)
                    patch_batch_mIoU = mIoU(labels_glb[:, top:down, left:right].cpu().contiguous(),
                                            patch_res.cpu().contiguous(), c)

                    l = (patch_batch_mIoU < batch_mIoU)
                    gt[:, i] = l[0:b]
                score = torch.sum(gt.cpu() == pred_rein.cpu().float()).float() / (self.sub_batch_size * self.patch_n[0] * self.patch_n[1])
                return score
            elif self.mode == 3:
                torch.cuda.empty_cache()
                outputs_global = model.forward(images_glb, mode=1)
                if self.coordinates is None:
                    self.coordinates, self.patch_n, self.ratio = global2patch(outputs_global.clone(), self.size_p)
                pred_rein = model.forward(images_glb, patch_n=self.patch_n, mode=2, glb_res=outputs_global.clone())
                pred_rein = (pred_rein > 0.5).int()

                # All patches
                # pred_rein = torch.ones(1, 16).int()
                # pred_rein[...] = 1
                selected_patch, _, index = crop_rein_patch(images_origin, None, pred_rein, ratio=self.ratio,
                                                           coords=self.coordinates)
                b, c, h, w = outputs_global.size()
                score = 0

                # whole global rein version
                # template = outputs_global.clone()
                template = torch.zeros([b, 7, 612, 612])
                template_select = torch.zeros([b, 7, 612, 612])
                for i in range(len(images_origin)):
                    patches_var = images_transform(selected_patch[i][:])
                    _, _, h_p, w_p = patches_var.size()
                    # if True:
                    if h_p > self.max_patch_side or w_p > self.max_patch_side:
                        if h_p > w_p:
                            patches_var = F.interpolate(patches_var, size=(self.max_patch_side, int(self.max_patch_side / h_p * w_p)),mode='bilinear', align_corners=True)
                        else:
                            patches_var = F.interpolate(patches_var, size=(int(self.max_patch_side / w_p * h_p), self.max_patch_side),mode='bilinear', align_corners=True)
                    # reinforce process
                    # torch.cuda.empty_cache()

                    global_rein_s = []
                    j = 0
                    while j < len(index[i]):
                        top, left = self.coordinates[index[i][j]]
                        top = int(np.round(top * h))
                        left = int(np.round(left * w))
                        down, right = top + self.size_p[0], left + self.size_p[1]
                        rein_patch = outputs_global[i:i + 1, :, top:down, left:right]
                        rein_patch = F.softmax(F.interpolate(rein_patch, size=patches_var.size()[2:], mode='bilinear'), dim=1)
                        global_rein_s.append(model.forward(rein_patch, patch_inp=patches_var[j:j + 1], mode=self.mode))
                        j += 1
                        # torch.cuda.empty_cache()
                    global_rein_s = torch.cat(global_rein_s, dim=0)


                    for j in range(len(index[i])):
                        top, left = self.coordinates[index[i][j]]
                        top = int(np.round(top * 612))
                        left = int(np.round(left * 612))
                        down, right = top + 153, left + 153
                        template[i:i + 1, :, top:down, left:right] = global_rein_s[j:j + 1]
                        template_select[i:i + 1, :, top:down, left:right] = torch.ones_like(template_select[i:i + 1, :, top:down, left:right])

                template = F.interpolate(template,size=(outputs_global.shape[2],outputs_global.shape[3] ),mode='bilinear', align_corners=True)
                rein_global = model.forward(outputs_global, template=template, mode=self.mode)
                self.metrics_fuse.update(labels_glb.cpu(), rein_global.argmax(1).cpu().detach())
                self.metrics_tea.update(labels_glb.cpu(), outputs_global.argmax(1).cpu().detach())
                return outputs_global
