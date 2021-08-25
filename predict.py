#!/usr/bin/env python3
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from QataCovDataset import QataCovDataset
import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt

from tqdm import tqdm
from model.unet import UNet


def mask_to_image(mask):
    #return Image.fromarray((mask*255)).astype(np.uint8)
    return (mask*255).astype(np.uint8)


def predict(imgs,net,device):

    imgs = imgs.to(device=device, dtype=torch.float32)

        # zero the parameter gradients
        # optimizer.zero_grad()

    with torch.set_grad_enabled(False):
        masks_pred = net(imgs)
        pred = torch.sigmoid(masks_pred) > 0.5

        print(f'Torch prediction shape:{pred.shape}')

        return pred.detach().cpu().numpy()

#def transform():


def get_args():

    parser = argparse.ArgumentParser(description = "Qata_Covid19 Segmentation" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--path',type=str,default='./covid_1.png')
    parser.add_argument('--gpu', type=str, default = '0')
    # arguments for training
    parser.add_argument('--img_size', type = int , default = 224)

    parser.add_argument('--load_model', type=str, default='best_checkpoint.pt', help='.pth file path to load model')
    return parser.parse_args()

def main():

    args = get_args()

    img_path= args.path

    img = Image.open(img_path).convert('L').resize((args.img_size, args.img_size), Image.LANCZOS)

    mask = Image.fromarray(np.zeros((args.img_size,args.img_size),dtype=np.uint8))

    real_mask_image = Image.open('./mask_covid_1.png').convert('L')

   
    # set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # default: '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set model
    model = UNet(n_channels=1, n_classes=1).to(device)

    checkpoint = torch.load(args.load_model)
    model.load_state_dict(checkpoint['model_state_dict'])


    """set img size
        - UNet type architecture require input img size be divisible by 2^N,
        - Where N is the number of the Max Pooling layers (in the Vanila UNet N = 5)
    """

    img_size = args.img_size #default: 224


    # set transforms for dataset
    import torchvision.transforms as transforms
    from my_transforms import RandomHorizontalFlip,RandomVerticalFlip,ColorJitter,GrayScale,Resize,ToTensor
    eval_transforms = transforms.Compose([
        GrayScale(),
        Resize(img_size),
        ToTensor()
    ])


    sample = {'image': img, 'mask': mask}

    sample = eval_transforms(sample)

    img_batch = sample['image'].unsqueeze(0)

    predicted_masks = predict(img_batch,model,device)

    print(f'Numpy prediction shape:{predicted_masks.shape}')

    predicted_mask_images = mask_to_image(predicted_masks)

    first_image = predicted_mask_images[0][0]

    images = [np.array(img),np.array(real_mask_image),first_image]

    imgs_comb = np.hstack(images)

    fig = plt.figure()

    plt.imshow(imgs_comb)

    fig.suptitle('X_Ray Image - True Mask - Predicted Mask',fontsize=10)

    #fig.show()
    
    plt.show()

    croped_image = np.where(predicted_masks[0][0],img,0)

    plt.imshow(croped_image,cmap='gray')

    plt.show()

if __name__ == '__main__':

    main()
