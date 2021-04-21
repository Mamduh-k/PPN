import torch
import numpy as np
from torchvision import transforms
from dataset.deep_globe import RGB_mapping_to_class
from PIL import Image
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
def get_sub_batch(B, order, list_image, list_label, g2l_temp):
    patch = []
    label = []
    g2l = []
    for i in range(B):
        patch.append(list_image[order[i]])
        label.append(list_label[order[i]])
        g2l.append(g2l_temp[order[i]])
        return torch.stack(patch), torch.stack(label), torch.cat(g2l, dim=0)
def get_local(images_origin, labels_origin, ratio, coordinates):
    pred = torch.ones(len(images_origin), 16).int()
    selected_patch, selected_label, all_selected_index = crop_rein_patch(images_origin, labels_origin, pred,
                                                                         ratio=ratio, coords=coordinates)
    all_selected_index = [i.cpu().numpy().tolist() for i in all_selected_index]
    all_selected_patch = []
    all_selected_label = []
    all_selected_patch_num = 0
    for i in range(len(selected_patch)):
        all_selected_patch = all_selected_patch + selected_patch[i]
        all_selected_label = all_selected_label + selected_label[i]
        all_selected_patch_num = all_selected_patch_num + len(selected_patch[i])
    return all_selected_patch, all_selected_label, all_selected_index


def crop_rein_patch(images, labels=None, pred_patch=None, ratio=None, coords=None):
    rein_index = []
    select_patch = []
    select_label = []
    for i in range(len(pred_patch)):
        no_zero = pred_patch[i].nonzero()[:, 0]
        rein_index.append(no_zero)
        w, h = images[i].size
        size = (h, w)
        select_patch.append([0] * len(no_zero))
        select_label.append([0] * len(no_zero))
        for j in range(len(no_zero)):
            coord = coords[rein_index[i][j]]
            patch = transforms.functional.crop(images[i], int(np.round(coord[0] * size[0])) , int(np.round(coord[1] * size[1])),
                                               int(np.round(ratio[0] * size[0])), int(np.round(ratio[1] * size[1])))
            select_patch[i][j] = patch
            if labels is not None:
                label = transforms.functional.crop(labels[i], int(np.round(coord[0] * size[0])) , int(np.round(coord[1] * size[1])),
                                                   int(np.round(ratio[0] * size[0])), int(np.round(ratio[1] * size[1])))
                select_label[i][j] = label
    return select_patch, select_label, rein_index





def get_patch_info(shape, p_size):
    '''
    shape: origin image size, (x, y)
    p_size: patch size (square)
    return: n_x, n_y, step_x, step_y
    '''
    x = shape[0]
    y = shape[1]
    n = m = 1
    while x > n * p_size:
        n += 1
    # while p_size - 1.0 * (x - p_size) / (n - 1) < 50:
    #     n += 1
    while y > m * p_size:
        m += 1
    # while p_size - 1.0 * (y - p_size) / (m - 1) < 50:
    #     m += 1
    return n, m, p_size, p_size  # (x - p_size) * 1.0 / (n - 1), (y - p_size) * 1.0 / (m - 1)
def get_patch(images, ratio=(0.25, 0.25)):
    '''
    image/label => patches
    p_size: patch size
    return: list of PIL patch images; coordinates: images->patches; ratios: (h, w)
    '''
    patches = [];
    coordinates = [];
    p_sizes = [];
    n_x, n_y = int(1 / ratio[0]), int(1 / ratio[1])
    for i in range(len(images)):
        w, h = images[i].size
        size = (h, w)

        step_x, step_y = ratio[0] * size[0], ratio[1] * size[1]
        p_size = (int(step_x), int(step_y))
        p_sizes.append(p_size)
        # ratios[i] = (float(p_size[0]) / size[0], float(p_size[1]) / size[1])
        patches.append([images[i]] * (n_x * n_y))
        coordinates.append([(0, 0)] * (n_x * n_y))
        for x in range(n_x):
            if x < n_x - 1:
                top = int(np.round(x * step_x))
            else:
                top = size[0] - p_size[0]
            for y in range(n_y):
                if y < n_y - 1:
                    left = int(np.round(y * step_y))
                else:
                    left = size[1] - p_size[1]
                coordinates[i][x * n_y + y] = (1.0 * top / size[0], 1.0 * left / size[1])
                patches[i][x * n_y + y] = transforms.functional.crop(images[i], top, left, p_size[0], p_size[1])
    return patches, coordinates, p_sizes
def global2patch(globals, p_size):
    '''
    image/label => patches
    p_size: patch size
    return: list of PIL patch images; coordinates: images->patches; ratios: (h, w)
    '''

    h, w = globals.size()[-2:]
    size = (h, w)
    ratio = (float(p_size[0]) / size[0], float(p_size[1]) / size[1])
    n_x, n_y, step_x, step_y = get_patch_info(size, p_size[0])
    coordinates = [(0, 0)] * (n_x * n_y)
    for x in range(n_x):
        if x < n_x - 1:
            top = int(np.round(x * step_x))
        else:
            top = size[0] - p_size[0]
        for y in range(n_y):
            if y < n_y - 1:
                left = int(np.round(y * step_y))
            else:
                left = size[1] - p_size[1]

            coordinates[x * n_y + y] = (1.0 * top / size[0], 1.0 * left / size[1])

    return coordinates, (n_x, n_y), ratio
def crop_rein_patch(images, labels=None, pred_patch=None, ratio=None, coords=None):
    rein_index = []
    select_patch = []
    select_label = []
    for i in range(len(pred_patch)):
        no_zero = pred_patch[i].nonzero()[:, 0]
        rein_index.append(no_zero)
        w, h = images[i].size
        size = (h, w)
        select_patch.append([0] * len(no_zero))
        select_label.append([0] * len(no_zero))
        for j in range(len(no_zero)):
            coord = coords[rein_index[i][j]]
            patch = transforms.functional.crop(images[i], int(np.round(coord[0] * size[0])),
                                               int(np.round(coord[1] * size[1])),
                                               int(np.round(ratio[0] * size[0])), int(np.round(ratio[1] * size[1])))
            select_patch[i][j] = patch
            if labels is not None:
                label = transforms.functional.crop(labels[i], int(np.round(coord[0] * size[0])),
                                                   int(np.round(coord[1] * size[1])),
                                                   int(np.round(ratio[0] * size[0])), int(np.round(ratio[1] * size[1])))
                select_label[i][j] = label
    return select_patch, select_label, rein_index
def resize(images, shape, label=False):
    '''
    resize PIL images
    shape: (w, h)
    '''
    resized = list(images)

    for i in range(len(images)):
        if label:
            resized[i] = images[i].resize(shape, Image.NEAREST)
            # resized[i] = images[i].resize(shape, Image.BILINEAR)
        else:
            resized[i] = images[i].resize(shape, Image.BILINEAR)
    return resized

def _mask_transform(mask):
    # isic
    # target = np.array(mask).astype('int32')
    # target[target == 255] = 1
    # deepglobe
    target = np.array(mask).astype('int32')
    target = RGB_mapping_to_class(target)
    # crag
    # target = np.array(mask).astype('int32')
    # target[target != 0] = 1
    return target
def masks_transform(masks, numpy=False):
    '''
    masks: list of PIL images
    '''
    targets = []
    for m in masks:
        targets.append(_mask_transform(m))
    targets = np.array(targets)
    if numpy:
        return targets
    else:
        return torch.from_numpy(targets).long().cuda()
def images_transform(images):
    '''
    images: list of PIL images
    '''
    inputs = []
    for img in images:
        img = transformer(img)
        inputs.append(img)
    inputs = torch.stack(inputs, dim=0).cuda()
    return inputs
