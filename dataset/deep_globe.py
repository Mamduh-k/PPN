import os
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
from torchvision import transforms
import cv2
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def fliplr(x):
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(
                np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return x
def rot90(arr,num):
    num = 4 - num
    c,h,w = arr.shape
    arr = arr.reshape(c, 1, h, w)
    for i in range(num):
        arr = arr.swapaxes(-2,-1)[...,::-1]
    arr = arr.reshape(c,h,w)
    return arr

def imshow_tensor(tensor, title=None):
    import matplotlib.pyplot as plt
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def imshow_2d(tensor,name):
    import matplotlib.pyplot as plt
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = unloader(image)
    plt.figure("dog")
    plt.imshow(image)
    # plt.show()

    plt.savefig(name+'.jpg')
    plt.pause(0.001)

def imshow_array(array):
    from PIL import Image
    import matplotlib.pyplot as plt
    image = Image.fromarray(array)
    plt.imshow(image)
    plt.pause(0.001)

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

def masks_transform_single(masks, numpy=False):
    '''
    masks: list of PIL images
    '''
    targets = []

    targets.append(_mask_transform(masks))
    targets = np.array(targets)
    if numpy:
        return targets
    else:
        return torch.from_numpy(targets).long()

def images_transform_single(images):
    '''
    images: list of PIL images
    '''
    images = transformer(images)
    return images



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def find_label_map_name(img_filenames, labelExtension=".png"):
    img_filenames = img_filenames.replace('_sat.jpg', '_mask')
    return img_filenames + labelExtension

def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr

def flip90_right(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.transpose(new_arr)[::-1]
    return new_arr

def RGB_mapping_to_class(label):
    l, w = label.shape[0], label.shape[1]
    classmap = np.zeros(shape=(l, w))
    indices = np.where(np.all(label == (0, 255, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 1
    indices = np.where(np.all(label == (255, 255, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 2
    indices = np.where(np.all(label == (255, 0, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 3
    indices = np.where(np.all(label == (0, 255, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 4
    indices = np.where(np.all(label == (0, 0, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 5
    indices = np.where(np.all(label == (255, 255, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 6
    indices = np.where(np.all(label == (0, 0, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 0
    #     plt.imshow(colmap)
    #     plt.show()
    return classmap


def classToRGB(label):
    l, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(l, w, 3)).astype(np.float32)
    indices = np.where(label == 1)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 255, 255]
    indices = np.where(label == 2)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 255, 0]
    indices = np.where(label == 3)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 0, 255]
    indices = np.where(label == 4)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 255, 0]
    indices = np.where(label == 5)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 255]
    indices = np.where(label == 6)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 255, 255]
    indices = np.where(label == 10)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 0, 0]
    indices = np.where(label == 0)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 0]
    transform = ToTensor();
    #     plt.imshow(colmap)
    #     plt.show()
    return transform(colmap)


def class_to_target(inputs, numClass):
    batchSize, l, w = inputs.shape[0], inputs.shape[1], inputs.shape[2]
    target = np.zeros(shape=(batchSize, l, w, numClass), dtype=np.float32)
    for index in range(7):
        indices = np.where(inputs == index)
        temp = np.zeros(shape=7, dtype=np.float32)
        temp[index] = 1
        target[indices[0].tolist(), indices[1].tolist(), indices[2].tolist(), :] = temp
    return target.transpose(0, 3, 1, 2)


def label_bluring(inputs):
    batchSize, numClass, height, width = inputs.shape
    outputs = np.ones((batchSize, numClass, height, width), dtype=np.float)
    for batchCnt in range(batchSize):
        for index in range(numClass):
            outputs[batchCnt, index, ...] = cv2.GaussianBlur(inputs[batchCnt, index, ...].astype(np.float), (7, 7), 0)
    return outputs


class DeepGlobe(data.Dataset):
    """input and label image dataset"""

    def __init__(self, root, ids, label=False, transform=False,size_g = (512,512)):
        super(DeepGlobe, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.label = label
        self.transform = transform
        self.ids = ids
        self.classdict = {1: "urban", 2: "agriculture", 3: "rangeland", 4: "forest", 5: "water", 6: "barren", 0: "unknown"}

        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04)
        self.resizer = transforms.Resize((2448, 2448))
        self.size_g = size_g

    def __getitem__(self, index):
        sample = {}
        # sample['id'] = self.ids[index][:-8]
        sample['id'] = self.ids[index]
        image = Image.open(os.path.join(self.root, "Sat/" + self.ids[index])) # w, h
        sample['image_origin'] = image
        sample['index'] = index
        # print(index)
        if self.label:
            label = Image.open(os.path.join(self.root, 'Label/' + self.ids[index].replace('_sat.jpg', '_mask.png')))
            sample['label_origin'] = label

        if self.transform and self.label:

            image, label = self._transform(image, label)
            sample['image_origin'] = image
            sample['label_origin'] = label



        images_glb = image.resize(self.size_g, Image.BILINEAR)
        images_glb = images_transform_single(images_glb)
        labels_glb = label.resize((self.size_g[0]//4, self.size_g[1]//4), Image.NEAREST)

        labels_glb = masks_transform_single(labels_glb)



        sample['image'] = images_glb
        sample['label'] = labels_glb


        sample['res'] = label


        return sample

    def _transform(self, image, label):
        if np.random.random() > 0.5:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)

        if np.random.random() > 0.5:
            degree = random.choice([90, 180, 270])
            image = transforms.functional.rotate(image, degree)
            label = transforms.functional.rotate(label, degree)

        return image, label


    def __len__(self):
        return len(self.ids)