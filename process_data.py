#!/usr/bin/env python3
import os
import scipy

import numpy as np
import pandas as pd
from PIL import Image

#from skimage.filters import gabor_kernel
import math
import argparse
from tqdm import tqdm

from scipy.stats import rankdata

import matplotlib.pyplot as plt
import cv2

def process_image(img,kernels):

    # decode image

    footprint = np.array([[1,1,1],[1,1,1],[1,1,1]])

    decode_img = scipy.ndimage.generic_filter(img,convolve,footprint=footprint)

    decode_img = decode_img.reshape(-1)

    # calculate frequencies

    _, freqs = np.unique(decode_img, return_counts=True)

    freqs = np.sort(freqs)[::-1]

    # calculate rank
    
    rank = abs(rankdata(freqs,method='max')-freqs.shape[0]-1)  

    # calculate ferquencies >1

    freqs_deleted_ones = np.delete(freqs,np.where(freqs == 1))

    # remove redandate frequencies
    
    unique_freqs,count_freqs = np.unique(freqs,return_counts=True)

    
    unique_freqs,count_freqs = unique_freqs[::-1],count_freqs[::-1]

    match = np.concatenate([np.expand_dims(unique_freqs,axis=1),np.expand_dims(count_freqs,axis=1)],axis=1)

    nbr_freqs=np.zeros(freqs.shape)

    for i in match:
        nbr_freqs[np.where(freqs == i[0])] = i[1]

    # entropy calculation

    p_e1 = freqs / np.sum(freqs)

    entropy_1 = -np.sum(p_e1 * np.log(p_e1))/math.log(freqs.shape[0])

    p_e2 = count_freqs / freqs.shape[0]

    entropy_2 = -np.sum(p_e2 * np.log(p_e2))/math.log(count_freqs.shape[0])

    # calculate slope

    u = np.log(freqs_deleted_ones)
    v= np.log(np.arange(1,freqs_deleted_ones.shape[0]+1))
    pente, constante= np.polyfit(u,v,1)

    # calculate air under zipf

    oao_zipf = math.log10(freqs[0]) 


    rank_deleted_ones = rank[:freqs_deleted_ones.shape[0]]
    
    air_zipf = np.sum((freqs_deleted_ones[:-1]+freqs_deleted_ones[1:])*(rank_deleted_ones[1:]-rank_deleted_ones[:-1])/2)

    # calculate zipf inverse
    
    u = np.log(freqs)
    v = np.log(nbr_freqs)

    zi_pente,_ = np.polyfit(u,v,1)

    oao_zipf_inv = math.log10(nbr_freqs[-1])

    # all zipf and zipf inverse features

    zipf_features = np.array([pente, constante, entropy_1, entropy_2, oao_zipf, air_zipf, oao_zipf_inv, zi_pente],dtype=np.float32)

    # calculate gabor features
    
    gabor_features_data = gabor_features(img,kernels,32,32)

    return np.concatenate([zipf_features, gabor_features_data])


    
def convolve(window):

    flat_window = window.reshape(-1)

    window_history = flat_window.copy()

    attempt=1

    flat_window= np.where(flat_window == window_history[0],0,flat_window)

    for i,x in enumerate(window_history):
        if i ==0: continue
        if x != window_history[i-1]:
            flat_window= np.where(flat_window == x,attempt,flat_window)
            attempt+=1


    cum = flat_window[8]+flat_window[7]*10+flat_window[6]*100+flat_window[5]*1000+flat_window[4]*10000+flat_window[3]*100000+flat_window[2]*1000000+flat_window[1]*10000000+flat_window[0]*100000000

    return cum


def gabor_kernels(u,v,m,n):

    filters = []
    fmax = 0.25
    gama = math.sqrt(2)
    eta = math.sqrt(2)
    for i in range(1,u+1):

        fu = fmax/((math.sqrt(2))**(i-1))
        alpha = fu/gama
        beta = fu/eta

        for j in range(1,v+1):
            tetav = ((j-1)/v)*math.pi
            g_filter = np.zeros((m,n),dtype=np.complex128)

            for x in range(1,m+1):
                for y in range(1,n+1):
                    xprime = (x-((m+1)/2))*np.cos(tetav)+(y-((n+1)/2))*np.sin(tetav);
                    yprime = -(x-((m+1)/2))*np.sin(tetav)+(y-((n+1)/2))*np.cos(tetav);
                    g_filter[x-1,y-1] = (fu**2/(math.pi*gama*eta))*np.exp(-((alpha**2)*(xprime**2)+(beta**2)*(yprime**2)))*np.exp(1j*2*math.pi*fu*xprime);

            filters.append(g_filter)


    return filters

def gabor_features(img, kernels, d1, d2):

    features = []

    for kernel in kernels:

        filtred_img_complex = scipy.ndimage.convolve(img,kernel)

        #filtred_img_complex = cv2.filter2D(img,-1,kernel)

        filtred_img = np.abs(filtred_img_complex)

        down_fi = filtred_img[::d1,::d2]

        flat_fi = down_fi.reshape(-1)

        flat_fi = (flat_fi-np.mean(flat_fi))/np.std(flat_fi)

        features.append(flat_fi)


    return np.concatenate(features)

def get_args():

    parser = argparse.ArgumentParser(description = "Qata_Covid19 Segmentation" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--path',type=str,default='./data/Qata_COV')
    # arguments for training
    parser.add_argument('--img_size', type = int , default = 224)

    parser.add_argument('--out', type=str, default='./dataset')
    return parser.parse_args()

def main():

    args = get_args()

    images_path = os.path.join(args.path,'predict_crop_images')

    df = pd.read_csv(os.path.join(args.path,'target.csv'),nrows=10)

    kernels = gabor_kernels(5,8,39,39)

    data = []

    for row in tqdm(df['img'].values):

        img = np.array(Image.open(os.path.join(images_path,'croped_'+row)).convert('L'))

        features = process_image(img,kernels)

        data.append(features)


    np_data = np.concatenate(data)

    feature_df = pd.DataFrame(data)

    final_df = pd.concat([df,feature_df],axis=1)
    
    final_df.to_csv(os.path.join(args.out,'data.csv'),index=False)


if __name__ == '__main__':

    main()

    #img = np.random.randint(0,256,(224,224),dtype=np.uint8)

    #img= np.zeros((224,224),dtype=np.uint8)

    #plt.imshow(img)

    #kernels = gabor_kernels(5,8,39,39)

    #features = process_image(img,kernels)

    #print(features)
    
    #print(np.sum(np.isnan(features)))
    #print(features.shape[0])
    #plt.show()
