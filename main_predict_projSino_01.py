#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:53:22 2020

@author: zilingwu
"""


import numpy as np
import h5py, argparse, os
from util import save2img

parser = argparse.ArgumentParser(description='Generate testing datasets')
parser.add_argument('-dose', type=int, default=10,  help='Dose value')
parser.add_argument('-proj_modelName', type=str, required=True, help='Trained model name')
parser.add_argument('-sino_modelName', type=str, required=True, help='Trained model name')

args, unparsed = parser.parse_known_args()


# load original noisy image
print('Loading original noisy image')
hf_1 = h5py.File('./H5_Files/sino_00058_noisy_%d.h5'%(args.dose), 'r')
data_noisy = hf_1['dataset'][:].astype(np.float32)
print(data_noisy.shape)
print(data_noisy.min())
print(data_noisy.max())

# load original denoised image in projection domain
print('Loading original denoised image in projection domain')
hf_2 = h5py.File('./result_58/proj/H5_File/sino_recon_%s_proj.h5'%(args.proj_modelName), 'r')
# data_recon_proj = hf_2.get('dataset')
data_recon_proj = hf_2['dataset'][:].astype(np.float32)
print(data_recon_proj.shape)
print(data_recon_proj.min())
print(data_recon_proj.max())


# load original denoised image in sinogram domain
print('Loading original denoised image in sinogram domain')
hf_3 = h5py.File('./result_58/sino/H5_File/sino_recon_%s_sino.h5'%(args.sino_modelName), 'r')
# data_recon_sino = hf_3.get('dataset')
data_recon_sino = hf_3['dataset'][:].astype(np.float32)
print(data_recon_sino.shape)
print(data_recon_sino.min())
print(data_recon_sino.max())



# Combine
print('Combining to get final results')
arrays = []
for i in range(data_recon_sino.shape[0]):
    print('Slice number %d'%(i))
    arrays.append(data_noisy[i])
    arrays.append(data_recon_proj[i])
    arrays.append(data_recon_sino[i])
arrays = np.asarray(arrays, dtype=np.float32)
# arrays = np.concatenate((data_noisy[0:5], data_recon_proj[0:5], data_recon_sino[0:5]), axis=0)
print(arrays.shape)


# save2img(arrays[0,:,:], './projSino/noisy_1.png')
# save2img(arrays[1,:,:], './projSino/noisy_2.png')
# save2img(arrays[2,:,:], './projSino/noisy_3.png')



hf_1.close()
hf_2.close()
hf_3.close()

print('Saving noisy dataset into projSino/dataset')
hf = h5py.File('./projSino/H5_File/sino_00058_combine_%d_%s_%s.h5'%(args.dose,\
     args.proj_modelName,args.sino_modelName), 'w')
hf.create_dataset('images', data = arrays)
hf.close()







