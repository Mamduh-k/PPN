

import torch
import numpy as np
import torch.nn.functional as F
from dataset.deep_globe import classToRGB
import os
from torchvision import transforms

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
    # b_mean_iou = torch.mean(b_iou, dim=-1)
    # t = b_iou[:, 1:].mean(dim=-1)
    # t = b_iou[b_iou < 1]
    m_iou = torch.zeros(b_iou.size()[0])
    for i in range(b_iou.size()[0]):
        if len(b_iou[i][b_iou[i] < 1]) != 0:
            m_iou[i] = b_iou[i][1:][b_iou[i][1:] < 1].mean()
        else:
            m_iou[i] = 1.
    return m_iou #b_iou[:, 1:].mean(dim=-1) # b_mean_iou

def select_best_mIoU(local,Global,fuse,label,classes):
    target = []
    for i in range(label.shape[0]):
        local_miou = mIoU(label[i:i+1], local[i:i+1].argmax(1).cpu(), classes)
        global_miou = mIoU(label[i:i+1], Global[i:i+1].argmax(1).cpu(), classes)
        fuse_miou = mIoU(label[i:i+1], fuse[i:i+1].argmax(1).cpu(), classes)
        if local_miou >= global_miou and local_miou >= fuse_miou:
            target.append(local[i:i+1])
        elif global_miou >= local_miou and global_miou >= fuse_miou:
            target.append(Global[i:i+1])
        elif fuse_miou >= local_miou and fuse_miou >= global_miou:
            target.append(fuse[i:i+1])
        else:
            print("NOne")
    return torch.cat(target,dim=0)

def select_G_L_best_mIoU(local,Global,label,classes):
    target = []
    for i in range(label.shape[0]):
        local_miou = mIoU(label[i:i+1], local[i:i+1].argmax(1).cpu(), classes)
        global_miou = mIoU(label[i:i+1], Global[i:i+1].argmax(1).cpu(), classes)
        if local_miou >= global_miou :
            target.append(local[i:i+1])
        elif global_miou >= local_miou :
            target.append(Global[i:i+1])
    return torch.cat(target,dim=0)

# def get_classification_label(label_seg):
#     label_class = to_categorical(np.unique(label_seg[0]))
#     for i in range(1,label_seg.shape[0]):
#         label_class = np.concatenate((label_class,np.unique(label_seg[i])),axis=0)
#
#     return label_class



def save_pred_img(img, root='./predict_mask', subdir='', filename=''):
    img = classToRGB(img.argmax(1)[0].detach().cpu().numpy()) * 255.
    save_path = os.path.join(root, subdir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    transforms.functional.to_pil_image(img).save(os.path.join(save_path, filename+'.png'))

def save_gt_img(img, root='./predictions', subdir='', filename=''):
    img_save = classToRGB(img[0].detach().cpu().numpy())*255
    save_path = os.path.join(root, subdir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    transforms.functional.to_pil_image(img_save).save(os.path.join(save_path, filename+'.png'))

#
def global2local_(Global,coordinates,size_p,target_size,index):
    '''
    :param Global:
    :param coordinates:
    :param size_p:
    :param target_size:
    :return: g2l
    the order is global2local1 global2local2.etc
    '''
    b, c, h, w = Global.size()
    g2l = []
    for i in range(b):
        for j in range(len(index[i])):
            top, left = coordinates[index[i][j]]
            top = int(np.round(top * h))
            left = int(np.round(left * w))
            down, right = top + size_p[0], left + size_p[1]

            rein_feat = Global[i:i + 1, :, top:down, left:right]
            if(size_p == target_size):
                g2l.append(rein_feat)
            else:
                g2l.append(F.interpolate(rein_feat, size=target_size, mode='bilinear', align_corners=True))
    return g2l

def global2local_label(Global,coordinates,size_p,target_size,index):
    '''
    :param Global:
    :param coordinates:
    :param size_p:
    :param target_size:
    :return: g2l
    the order is global2local1 global2local2.etc
    '''
    b, c, h, w = Global.size()
    g2l = []
    for i in range(b):
        for j in range(len(index[i])):
            top, left = coordinates[index[i][j]]
            top = int(np.round(top * h))
            left = int(np.round(left * w))
            down, right = top + size_p[0], left + size_p[1]

            rein_feat = Global[i:i + 1, :, top:down, left:right]
            g2l.append(F.interpolate(rein_feat, size=target_size, mode='nearest'))
    return g2l

def global2local(Global, coordinates, size_p, index):
    b, c, h, w = Global.size()
    g2l = []
    for i in range(b):
        for j in range(len(index[i])):
            top, left = coordinates[index[i][j]]
            top = int(np.round(top * h))
            left = int(np.round(left * w))
            down, right = top + size_p[0], left + size_p[1]

            rein_feat = Global[i:i + 1, :, top:down, left:right]
            g2l.append(rein_feat)
    return g2l
def local2global(template,local, coordinates, size_p, index):
    h,w = template.shape[2:]
    for i in range(template.shape[0]):
        for j in range(len(index[i])):
            top, left = coordinates[index[i][j]]
            top = int(np.round(top * h))
            left = int(np.round(left * w))
            down, right = top + size_p[0], left + size_p[1]
            template[i:i + 1, :, top:down, left:right] = local[i*16 + j: i*16 + j + 1, :,:, :]
    return template

    # b, c, h, w = Global.size()
    # g2l = []
    # for i in range(b):
    #     for j in range(len(index[i])):
    #         top, left = coordinates[index[i][j]]
    #         top = int(np.round(top * h))
    #         left = int(np.round(left * w))
    #         down, right = top + size_p[0], left + size_p[1]
    #
    #         rein_feat = Global[i:i + 1, :, top:down, left:right]
    #         g2l.append(rein_feat)
    # return g2l



