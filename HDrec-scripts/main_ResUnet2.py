#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 23:22:34 2020

@author: zilingwu
"""



#! /home/zliu0/usr/anaconda3/envs/tfgpu/bin/python
import tensorflow as tf 
tf.enable_eager_execution()
import numpy as np 
from util import save2img
import sys, os, time, argparse, shutil, scipy, h5py, glob
from models import tomogan_disc as make_discriminator_model  # import a disc model
from models_zw import Res_Unet as make_generator_model           # import a generator model
from data_processor import bkgdGen, gen_train_batch_bg_all, get1batch4test
from keras import backend as K
from scipy.io import savemat


tf.logging.set_verbosity(tf.logging.ERROR)
print('Reading parameters')
parser = argparse.ArgumentParser(description='encode sinogram image.')
parser.add_argument('-gpus',  type=str, default="0", help='list of visiable GPUs')
parser.add_argument('-expName', type=str, required=True, help='Experiment name')
parser.add_argument('-lmse', type=float, default=0.5, help='lambda mse')#0.5
parser.add_argument('-lperc', type=float, default=2.0, help='lambda perceptual')
parser.add_argument('-ladv', type=float, default=0, help='lambda adv')
parser.add_argument('-lnpcc', type=float, default=0, help='lambda adv')
parser.add_argument('-lunet', type=int, default=3, help='Unet layers')
parser.add_argument('-depth', type=int, default=1, help='input depth')
parser.add_argument('-itg', type=int, default=1, help='iterations for G')
parser.add_argument('-itd', type=int, default=1, help='iterations for D')
parser.add_argument('-xtrain', type=str, required=True, help='file name of X for training')
parser.add_argument('-ytrain', type=str, required=True, help='file name of Y for training')
parser.add_argument('-xtest', type=str, required=True, help='file name of X for testing')
parser.add_argument('-ytest', type=str, required=True, help='file name of Y for testing')

args, unparsed = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.Session(config = config)
tf.keras.backend.set_session(sess)


mb_size = 64
img_size = 128
in_depth = args.depth
disc_iters, gene_iters = args.itd, args.itg
lambda_mse, lambda_adv, lambda_perc, lambda_npcc = args.lmse, args.ladv, args.lperc, args.lnpcc

print('Generate datasets')

itr_out_dir = './projection/' + args.expName + '-itrOut'
if os.path.isdir(itr_out_dir): 
    shutil.rmtree(itr_out_dir)
os.mkdir(itr_out_dir) # to save temp output

print(' redirect print to a file')
#sys.stdout = open('%s/%s' % (itr_out_dir, 'iter-prints.log'), 'w') 
print('datasets size')
print('X train: {}\nY train: {}\nX test: {}\nY test: {}'.format(args.xtrain, args.ytrain, args.xtest, args.ytest))

# build minibatch data generator with prefetch for training
mb_data_iter = bkgdGen(data_generator=gen_train_batch_bg_all(x_fn=args.xtrain, \
                                      y_fn=args.ytrain, mb_size=mb_size, \
                                      in_depth=in_depth, img_size=img_size), \
                       max_prefetch=16)  
    
    
# build minibatch data generator with prefetch for validation
mb_data_iter_val = bkgdGen(data_generator=gen_train_batch_bg_all(x_fn=args.xtest, \
                                      y_fn=args.ytest, mb_size=mb_size, \
                                      in_depth=in_depth, img_size=img_size), \
                       max_prefetch=16)   

generator = make_generator_model(input_shape=(None, None, in_depth),filter_num = 64)
#, nlayers=args.lunet
discriminator = make_discriminator_model(input_shape=(img_size, img_size, 1))
print(generator.summary())
# input range should be [0, 255]
feature_extractor_vgg = tf.keras.applications.VGG19(\
                        weights='./vgg19_weights_notop.h5', \
                        include_top=False)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def adversarial_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def npcc_loss(generated_image,true_image):
    
    fsp=generated_image-K.mean(K.constant(generated_image),axis=(0,1,2,3),keepdims=True)
    fst=true_image-K.mean(K.constant(true_image),axis=(0,1,2,3),keepdims=True)
    
    devP=K.std(K.constant(generated_image),axis=(0,1,2,3))
    devT=K.std(K.constant(true_image),axis=(0,1,2,3))
    
    npcc_loss=(-1)*K.mean(K.constant(fsp*fst),axis=(0,1,2,3))/K.clip(devP*devT,K.epsilon(),None)    ## (BL,1)
    return npcc_loss



gen_optimizer  = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-08)
disc_optimizer = tf.train.AdamOptimizer(1e-4)



ckpt = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                            discriminator_optimizer=disc_optimizer,
                            generator=generator,
                            discriminator=discriminator)

loss_train = []
loss_val = []

print('Start training')
for epoch in range(20001):
    time_git_st = time.time()
    for _ge in range(gene_iters):
        X_mb, y_mb = mb_data_iter.next() # with prefetch
        #print(X_mb.shape)
        with tf.GradientTape() as gen_tape:
            gen_tape.watch(generator.trainable_variables)

            gen_imgs = generator(X_mb, training=True)
            disc_fake_o = discriminator(gen_imgs, training=False)

            # loss_mse = tf.losses.mean_squared_error(gen_imgs, y_mb)
            loss_mse = tf.losses.absolute_difference(gen_imgs, y_mb)
            loss_adv = adversarial_loss(disc_fake_o)
            loss_npcc = npcc_loss(gen_imgs,y_mb)

            vggf_gt  = feature_extractor_vgg.predict(tf.concat([y_mb, y_mb, y_mb], 3).numpy())
            vggf_gen = feature_extractor_vgg.predict(tf.concat([gen_imgs, gen_imgs, gen_imgs], 3).numpy())
            perc_loss= tf.losses.mean_squared_error(vggf_gt.reshape(-1), vggf_gen.reshape(-1))

            gen_loss = lambda_adv * loss_adv + lambda_mse * loss_mse + lambda_perc * perc_loss + \
                lambda_npcc * loss_npcc
            loss_train.append(gen_loss)
            
            
            gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

    itr_prints_gen = '[Info] Epoch: %05d, gloss: %.2f (mse%.3f, adv%.3f, perc:%.3f, npcc: %.3f), gen_elapse: %.2fs/itr' % (\
                     epoch, gen_loss, loss_mse*lambda_mse, loss_adv*lambda_adv, perc_loss*lambda_perc, lambda_npcc * loss_npcc,\
                     (time.time() - time_git_st)/gene_iters)

    
    time_dit_st = time.time()

    for _de in range(disc_iters):
        X_mb, y_mb = mb_data_iter.next() # with prefetch        
        with tf.GradientTape() as disc_tape:
            disc_tape.watch(discriminator.trainable_variables)

            gen_imgs = generator(X_mb, training=False)

            disc_real_o = discriminator(y_mb, training=True)
            disc_fake_o = discriminator(gen_imgs, training=True)

            disc_loss = discriminator_loss(disc_real_o, disc_fake_o)

            disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    print('%s; dloss: %.2f (r%.3f, f%.3f), disc_elapse: %.2fs/itr, gan_elapse: %.2fs/itr' % (itr_prints_gen,\
          disc_loss, disc_real_o.numpy().mean(), disc_fake_o.numpy().mean(), \
          (time.time() - time_dit_st)/disc_iters, time.time()-time_git_st))
        
    # validation   
    # if epoch % (200//gene_iters) == 0:
    X_val, Y_val = mb_data_iter_val.next() # with prefetch
    Y_pred = generator.predict(X_val)
        
    # loss_mse = tf.losses.mean_squared_error(Y_pred, Y_val)
    loss_mse = tf.losses.absolute_difference(Y_val, Y_pred)
    vggf_gt  = feature_extractor_vgg.predict(tf.concat([Y_val, Y_val, Y_val], 3).numpy())
    vggf_gen = feature_extractor_vgg.predict(tf.concat([Y_pred, Y_pred, Y_pred], 3).numpy())
    perc_loss= tf.losses.mean_squared_error(vggf_gt.reshape(-1), vggf_gen.reshape(-1))
    loss_npcc = npcc_loss(Y_pred,Y_val)

        # vggf_gt  = feature_extractor_vgg.predict(tf.concat([y_mb, y_mb, y_mb], 3).numpy())
        # vggf_gen = feature_extractor_vgg.predict(tf.concat([gen_imgs, gen_imgs, gen_imgs], 3).numpy())
        # perc_loss= tf.losses.mean_squared_error(vggf_gt.reshape(-1), vggf_gen.reshape(-1))

        # gen_loss = lambda_adv * loss_adv + lambda_mse * loss_mse + lambda_perc * perc_loss + \
        #         lambda_npcc * loss_npcc
    val_error = lambda_mse * loss_mse + lambda_perc * perc_loss + lambda_npcc * loss_npcc
    loss_val.append(val_error)
    print('Validation error %.2f'%(val_error))

    if epoch % (200//gene_iters) == 0:
        X222, y222 = get1batch4test(x_fn=args.xtest, y_fn=args.ytest, in_depth=in_depth)
        pred_img = generator.predict(X222[:1])
        print('Prediction')
        print(pred_img.shape)
        
        
        save2img(pred_img[0,:,:,0], '%s/it%05d.png' % (itr_out_dir, epoch))
        if epoch == 0: 
            save2img(y222[0,:,:,0], '%s/gtruth.png' % (itr_out_dir))
            save2img(X222[0,:,:,in_depth//2], '%s/noisy.png' % (itr_out_dir))

        generator.save("%s/%s-it%05d.h5" % (itr_out_dir, args.expName, epoch), \
                       include_optimizer=False)

        # discriminator.save("%s/disc-it%05d.h5" % (itr_out_dir, epoch), \
        #                include_optimizer=False)

 #   sys.stdout.flush()
loss_val = np.asarray(loss_val, dtype=np.float32)
savemat("./projection/loss_val_%s.mat"%(args.expName), {'loss_val':loss_val})
loss_train = np.asarray(loss_train, dtype=np.float32)
savemat("./projection/loss_train_%s.mat"%(args.expName), {'loss_train':loss_train})
