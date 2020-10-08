# Deep-Learning-based-Low-dose-Tomography-Reconstruction-with-Hybrid-dose-Measurements
This repository includes the functions for [deep learning-based low-dose tomography tomography reconstruction with hybrid-dose measurements](https://arxiv.org/abs/2009.13589).

## Trainning
There are two different networks included, one is traditional Unet structure and the other is Unet with residual blocks. For each network, three different loss functions could be chosen by setting the correponding parameters. Here is the example to run by using mean absolute error loss only:

*python ./HDrec-scripts/main_ResUnet2.py -expName nor-proj-ResUnet2-random-128-dose10-l1 -xtrain ./projection_sino/dataset/noisy_train_128_10.h5 -ytrain ./projection_sino/dataset/clean_train_128.h5 -xtest ./projection_sino/dataset/noisy_test_128_10.h5 -ytest ./projection_sino/dataset/clean_test_128.h5 -lmse 10 -lperc 0 -ladv 0 -lnpcc 0 -itg 1 -itd 1 -gpus 1*

Required trainning datasets are included in the 'Datasets' folder. 

In the script, there also provides the opportunity to run with GAN-based training. 

## Prediction

The function "main_predict_proj.py" is included to test the trainned networks. Here is the example to run the script:

*python ./projection_sino/main_predict_proj.py -gpus 1 -modelName nor-proj-ResUnet2-random-2-dose100-l1-it20000 -xtest sino_00058_noisy_100 -tomo 58*

Corresponding datasets and trained models are also included in the 'Datasets' folder. 

If you find this work helpful, please consider cite:

Wu, Ziling, Tekin Bicer, Zhengchun Liu, Vincent De Andrade, Yunhui Zhu, and Ian T. Foster. "Deep Learning-based Low-dose Tomography Reconstruction with Hybrid-dose Measurements." arXiv preprint arXiv:2009.13589 (2020).





