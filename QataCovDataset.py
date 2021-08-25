import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader,Dataset
import os
import random
import cv2
from PIL import Image

from skimage import io


# torch.utils.data.Dataset is an abstract class representing a dataset
class QataCovDataset(Dataset): # inherit from torch.utils.data.Dataset
    "Lung sengmentation dataset."
    def __init__(self,root_dir = os.path.join(os.getcwd(),"data/Lung Segmentation"),split=None, transforms = None , shuffle = True):
        """
        Args:
        :param root_dir (str):
        :param split (str):
        :param transforms (callable, optional) :
        """
        self.root_dir = root_dir
        self.split = split # train / val / test
        self.transforms = transforms

        self.image_path = self.root_dir + '/Images/'

        # target
        self.mask_path = os.path.join(self.root_dir,'Ground-truths')
        

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        img_name = self.split[idx]
        # set index
        #for fName in self.data_file[self.split]["image"]:
        #    file_idx = int(fName.split('_')[1])
        #    if idx == file_idx:
        #        img_fName = fName
        img_path = os.path.join(self.image_path, img_name)
        img = Image.open(img_path).convert('L')  # open as PIL Image and set Channel = 1
        # img = cv2.imread(img_path)

        #for fName in self.data_file[self.split]["mask"]:
        #    file_idx = int(fName.split('_')[1])
        #    if idx == file_idx:
        #        mask_fName = fName
        mask_path = os.path.join(self.mask_path, 'mask_'+img_name)

        img_size = np.array(img).shape

        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')  # PIL Image
        else:
            mask = Image.fromarray(np.zeros(img_size,dtype=np.uint8))

        sample = {'image': img, 'mask': mask}

        if self.transforms:
            sample = self.transforms(sample)

        if isinstance(img,torch.Tensor) and isinstance(mask, torch.Tensor):
            assert img.size == mask.size
        return sample

if __name__ == "__main__":
    """Visualization"""
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision.transforms as transforms
    from my_transforms import Resize, ToTensor, GrayScale
    from torch.utils.data import DataLoader
    # set img size
    img_size = 512

    scale = Resize(512)
    composed = transforms.Compose([Resize(600),
                                   GrayScale()])

    data = LungSegDataset(split='val')

    def show(img, mask):
        combined = np.hstack((img, mask))
        plt.imshow(combined)
    fig = plt.figure(figsize=(30,30))
    cnt = 0
    for i in range(len(data)):
        sample = data[i]

    # for i in range(len(data)):
    #     if i == 3:
    #         break
    #     sample = data[i]
    #     for tsfrm in [composed]:
    #         transformed_sample = tsfrm(sample)
    #
    #         ax = plt.subplot(1, 3, i + 1)
    #         plt.tight_layout()
    #         ax.set_title(type(tsfrm).__name__)
    #         show(transformed_sample['image'],transformed_sample['mask'])
    #
    #
    #
    # plt.show()
    # print(data.train_image_idx)
    # print(data.train_mask_idx)
    # print(data.train_idx)
    # # print(data.train_mask_file[1].split("_"))
    # sample = []
    # for i in range(6):
    #     i += 100
    #     combined = np.hstack((data[i]['image'],data[i]['mask']))
    #     sample.append(combined)
    #
    # plt.figure(figsize=(25, 10))
    #
    # for i in [0,3]:
    #     for j in [1,2,3]:
    #         plt.subplot(2, 3, i + j)
    #         plt.imshow(sample[i + j - 1])
    # plt.show()

