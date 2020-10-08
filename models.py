import tensorflow as tf 
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Conv1D, BatchNormalization, \
    Activation, Input, UpSampling2D, MaxPooling2D, MaxPooling1D, SpatialDropout2D, Lambda, Add,\
        Dropout,Concatenate
import numpy as np 
from tensorflow.keras import layers

def tomogan_disc(input_shape):
    inputs = Input(shape=input_shape)
    _tmp = Conv2D(filters=64, kernel_size=3, padding='same', \
                  activation='relu')(inputs)
    _tmp = Conv2D(filters=64, kernel_size=3, padding='same', strides=(2,2),\
                  activation='relu')(_tmp)
    
    _tmp = Conv2D(filters=128, kernel_size=3, padding='same', \
                  activation='relu')(_tmp)
    _tmp = Conv2D(filters=128, kernel_size=3, padding='same', strides=(2,2),\
                  activation='relu')(_tmp)
    
    _tmp = Conv2D(filters=128, kernel_size=3, padding='same', \
                  activation='relu')(_tmp)
    _tmp = Conv2D(filters=4, kernel_size=3, padding='same', strides=(2,2),\
                  activation='relu')(_tmp)

    _tmp = layers.Flatten()(_tmp)
    _tmp = layers.Dense(units=64, activation='relu')(_tmp)
    _tmp = layers.Dense(units=1)(_tmp)
    
    return tf.keras.models.Model(inputs, _tmp)

def unet_conv_block(inputs, nch):
    _tmp = Conv2D(filters=nch, kernel_size=3, padding='same', activation='relu')(inputs)
    _tmp = Conv2D(filters=nch, kernel_size=3, padding='same', activation='relu')(_tmp)
    return _tmp

def unet(input_shape, use_cnnt=False, nlayers=3):
    inputs = Input(shape=input_shape)
    ly_outs= [inputs, ]
    label2idx = {'input': 0}
    
    _tmp = Conv2D(filters=8, kernel_size=1, padding='valid', activation='relu')(ly_outs[-1])
    ly_outs.append(_tmp)
#     label2idx['ch_stack'] = len(ly_outs)-1

    _tmp = unet_conv_block(ly_outs[-1], 32)
    ly_outs.append(_tmp)
    label2idx['box1_out'] = len(ly_outs)-1
    for ly in range(2, nlayers+1):
        _tmp = MaxPooling2D(pool_size=2, strides=2, padding='same')(ly_outs[-1])        
        _tmp = unet_conv_block(_tmp, 2*ly_outs[-1].shape[-1].value)
        ly_outs.append(_tmp)
        label2idx['box%d_out' % (ly)] = len(ly_outs)-1
        
    # intermediate layers
    _tmp = MaxPooling2D(pool_size=2, strides=2, padding='same')(ly_outs[-1])
    _tmp = Conv2D(filters=ly_outs[-1].shape[-1].value, kernel_size=3, \
                  padding='same', activation='relu')(_tmp)
    ly_outs.append(_tmp)
    
    for ly, nch in zip(range(1, nlayers+1), (64, 32, 32)):
        if use_cnnt:
            _tmp = Conv2DTranspose(filters=ly_outs[-1].shape[-1].value, activation='relu', \
                                   kernel_size=4, strides=(2, 2), padding='same')(ly_outs[-1]) 
        else: 
            _tmp = UpSampling2D(size=(2, 2), interpolation='bilinear')(ly_outs[-1]) 
        _tmp = tf.keras.layers.concatenate([ly_outs[label2idx['box%d_out' % (nlayers-ly+1)]], _tmp])
        _tmp = unet_conv_block(_tmp, nch)
        ly_outs.append(_tmp)
    
    _tmp = Conv2D(filters=16, kernel_size=1, padding='valid', 
                  activation='relu')(ly_outs[-1])

    _tmp = Conv2D(filters=1, kernel_size=1, padding='valid', \
                  activation=None)(_tmp)
    
    return tf.keras.models.Model(inputs, _tmp)

def re_unet(input_shape, use_cnnt=False, nlayers=3):
    inputs = Input(shape=input_shape)
    ly_outs= [inputs, ]
    label2idx = {'input': 0}
    
    _tmp = Conv2D(filters=8, kernel_size=1, padding='valid', activation='relu')(ly_outs[-1])
    ly_outs.append(_tmp)
#     label2idx['ch_stack'] = len(ly_outs)-1

    _tmp = unet_conv_block(ly_outs[-1], 32)
    ly_outs.append(_tmp)
    label2idx['box1_out'] = len(ly_outs)-1
    for ly in range(2, nlayers+1):
        _tmp = MaxPooling2D(pool_size=2, strides=2, padding='same')(ly_outs[-1])        
        _tmp = unet_conv_block(_tmp, 2*ly_outs[-1].shape[-1].value)
        ly_outs.append(_tmp)
        label2idx['box%d_out' % (ly)] = len(ly_outs)-1
        
    # intermediate layers
    _tmp = MaxPooling2D(pool_size=2, strides=2, padding='same')(ly_outs[-1])
    _tmp = Conv2D(filters=ly_outs[-1].shape[-1].value, kernel_size=3, \
                  padding='same', activation='relu')(_tmp)
    ly_outs.append(_tmp)
    
    for ly, nch in zip(range(1, nlayers+1), (64, 32, 32)):
        if use_cnnt:
            _tmp = Conv2DTranspose(filters=ly_outs[-1].shape[-1].value, activation='relu', \
                                   kernel_size=4, strides=(2, 2), padding='same')(ly_outs[-1]) 
        else: 
            _tmp = UpSampling2D(size=(2, 2), interpolation='bilinear')(ly_outs[-1]) 
        _tmp = tf.keras.layers.concatenate([ly_outs[label2idx['box%d_out' % (nlayers-ly+1)]], _tmp])
        _tmp = unet_conv_block(_tmp, nch)
        ly_outs.append(_tmp)
    
    _tmp = Conv2D(filters=16, kernel_size=1, padding='valid', 
                  activation = None)(ly_outs[-1])
    
    _tmp = Activation('relu')(_tmp)
    _tmp = Add()([_tmp,inputs])
    _tmp = Conv2D(filters=1, kernel_size=1, padding='valid', \
                  activation=None)(_tmp)
    
    return tf.keras.models.Model(inputs, _tmp)

def newUnet(input_shape):
    G_in=Input(shape=input_shape)
    G_1_1_bn=BatchNormalization()(G_in)
    G_1_1_relu=Activation('relu')(G_1_1_bn)
    G_1_1_c=Conv2D(8,kernel_size=(5,5),strides=(2,2), padding='same')(G_1_1_relu)
    G_1_c_bn= BatchNormalization()(G_1_1_c)
    G_1_c_bn_relu= Activation('relu')(G_1_c_bn)
    G_1_01=Conv2D(8,kernel_size=(5,5),strides=(1,1), padding='same')(G_1_c_bn_relu)
    G_1_02=Conv2D(8,kernel_size=(5,5),strides=(2,2), activation='relu', padding='same')(G_in)
    G_1_0=Add()([G_1_01,G_1_02])
    G_1_0_out=Dropout(0.02)(G_1_0)                        

    G_1r_bn=BatchNormalization()(G_1_0_out)
    G_1r_relu=Activation('relu')(G_1r_bn)
    G_1r_c=Conv2D(8,kernel_size=(5,5),strides=(1,1), padding='same')(G_1r_relu)
    G_1r_c_bn=BatchNormalization()(G_1r_c)
    G_1r_c_bn_relu=Activation('relu')(G_1r_c_bn)
    G_1r_c_out=Conv2D(8,(5,5),strides=(1,1),padding='same')(G_1r_c_bn_relu)

    G_1_out=Add()([G_1r_c_out,G_1_0_out])
    G_1_out=Dropout(0.02)(G_1_out)                        ## 128x128

    G_2_1_bn=BatchNormalization()(G_1_out)
    G_2_1_relu=Activation('relu')(G_2_1_bn)
    G_2_1_c=Conv2D(16,kernel_size=(5,5),strides=(2,2), padding='same')(G_2_1_relu)
    G_2_c_bn=BatchNormalization()(G_2_1_c)
    G_2_c_bn_relu= Activation('relu')(G_2_c_bn)
    G_2_01=Conv2D(16,kernel_size=(5,5),strides=(1,1), dilation_rate=(2,2),padding='same')(G_2_c_bn_relu)
    G_2_02=Conv2D(16,kernel_size=(5,5),strides=(2,2), padding='same')(G_1_out)
    G_2_0_out=Add()([G_2_01,G_2_02])

    G_2r_bn=BatchNormalization()(G_2_0_out)
    G_2r_relu=Activation('relu')(G_2r_bn)
    G_2r_c=Conv2D(16,kernel_size=(5,5),strides=(1,1), padding='same')(G_2r_relu)
    G_2r_c_bn=BatchNormalization()(G_2r_c)
    G_2r_c_bn_relu=Activation('relu')(G_2r_c_bn)
    G_2r_c_out=Conv2D(16,(5,5),strides=(1,1),padding='same')(G_2r_c_bn_relu)
    G_2_out=Add()([G_2r_c_out,G_2_0_out])
    G_2_out=Dropout(0.02)(G_2_out)                      ## 64x64
    
    G_3_1_bn=BatchNormalization()(G_2_out)
    G_3_1_relu=Activation('relu')(G_3_1_bn)
    G_3_1_c=Conv2D(32,kernel_size=(5,5),strides=(2,2), padding='same')(G_3_1_relu)
    G_3_c_bn=BatchNormalization()(G_3_1_c)
    G_3_c_bn_relu= Activation('relu')(G_3_c_bn)
    G_3_01=Conv2D(32,kernel_size=(5,5),strides=(1,1), padding='same')(G_3_c_bn_relu)
    G_3_02=Conv2D(32,kernel_size=(5,5),strides=(2,2), activation='relu', padding='same')(G_2_out)
    G_3_0_out=Add()([G_3_01,G_3_02])

    G_3r_bn=BatchNormalization()(G_3_0_out)
    G_3r_relu=Activation('relu')(G_3r_bn)
    G_3r_c=Conv2D(32,kernel_size=(5,5),strides=(1,1), padding='same')(G_3r_relu)
    G_3r_c_bn=BatchNormalization()(G_3r_c)
    G_3r_c_bn_relu=Activation('relu')(G_3r_c_bn)
    G_3r_c_out=Conv2D(32,(5,5),strides=(1,1),padding='same')(G_3r_c_bn_relu)
    G_3_out=Add()([G_3r_c_out,G_3_0_out])
    G_3_out=Dropout(0.02)(G_3_out)                         ## 32x32
    
    
    G_4_1_bn=BatchNormalization()(G_3_out)
    G_4_1_relu=Activation('relu')(G_4_1_bn)
    G_4_1_c=Conv2D(50,kernel_size=(5,5),strides=(2,2), activation='relu', padding='same')(G_4_1_relu)
    G_4_c_bn=BatchNormalization()(G_4_1_c)
    G_4_c_bn_relu= Activation('relu')(G_4_c_bn)
    G_4_01=Conv2D(64,kernel_size=(5,5),strides=(1,1), padding='same')(G_4_c_bn_relu)
    G_4_02=Conv2D(64,kernel_size=(5,5),strides=(2,2), activation='relu', padding='same')(G_3_out)
    G_4_0_out=Add()([G_4_01,G_4_02])
    
    G_4r_bn=BatchNormalization()(G_4_0_out)
    G_4r_relu=Activation('relu')(G_4r_bn)
    G_4r_c=Conv2D(64,kernel_size=(5,5),strides=(1,1), padding='same')(G_4r_relu)
    G_4r_c_bn=BatchNormalization()(G_4r_c)
    G_4r_c_bn_relu=Activation('relu')(G_4r_c_bn)
    G_4r_c_out=Conv2D(64,(5,5),strides=(1,1),padding='same')(G_4r_c_bn_relu)
    G_4_out=Add()([G_4r_c_out,G_4_0_out])
    G_4_out=Dropout(0.02)(G_4_out)                         ## 16x16
    
    G_5_1_bn=BatchNormalization()(G_4_out)
    G_5_1_relu=Activation('relu')(G_5_1_bn)
    G_5_1_c=Conv2D(90,kernel_size=(5,5),strides=(2,2), activation='relu', padding='same')(G_5_1_relu)
    G_5_c_bn=BatchNormalization()(G_5_1_c)
    G_5_c_bn_relu= Activation('relu')(G_5_c_bn)
    G_5_01=Conv2D(128,kernel_size=(5,5),strides=(1,1), activation='relu', padding='same')(G_5_c_bn_relu)
    G_5_02=Conv2D(128,kernel_size=(5,5),strides=(2,2), activation='relu', padding='same')(G_4_out)
    G_5_0_out=Add()([G_5_01,G_5_02])
    
    G_5r_bn=BatchNormalization()(G_5_0_out)
    G_5r_relu=Activation('relu')(G_5r_bn)
    G_5r_c=Conv2D(128,kernel_size=(5,5),strides=(1,1), padding='same')(G_5r_relu)
    G_5r_c_bn=BatchNormalization()(G_5r_c)
    G_5r_c_bn_relu=Activation('relu')(G_5r_c_bn)
    G_5r_c_out=Conv2D(128,(5,5),strides=(1,1),padding='same')(G_5r_c_bn_relu)
    G_5_out=Add()([G_5r_c_out,G_5_0_out])
    G_5_out=Dropout(0.02)(G_5_out)                          ## 8x8 
    
    
    
    G_6_up_bn=BatchNormalization()(G_5_out)
    G_6_up_relu= Activation('relu')(G_6_up_bn)
    G_6_up_ct= Conv2DTranspose(128, kernel_size=(5,5), strides=(2,2),padding='same')(G_6_up_relu)
    G_6_up_ct_bn=BatchNormalization()(G_6_up_ct)
    G_6_up_ct_relu= Activation('relu')(G_6_up_ct_bn)
    G_6_up_c_1_out=Conv2D(256,kernel_size=(5,5),strides=(1,1), padding='same')(G_6_up_ct_relu)
    G_6_up_c_2_out=Conv2DTranspose(256,kernel_size=(5,5),strides=(2,2), activation='relu', padding='same')(G_5_out)
    G_6_up_0=Add()([G_6_up_c_1_out,G_6_up_c_2_out])
    G_6_up_out=Concatenate()([G_6_up_0,G_4_out])
    G_6_up_out=Dropout(0.02)(G_6_up_out)                 ## 16x16 
    
    G_5_up_bn=BatchNormalization()(G_6_up_out)
    G_5_up_relu= Activation('relu')(G_5_up_bn)
    G_5_up_ct= Conv2DTranspose(100, kernel_size=(5,5), strides=(2,2),padding='same')(G_5_up_relu)
    G_5_up_ct_bn=BatchNormalization()(G_5_up_ct)
    G_5_up_ct_relu= Activation('relu')(G_5_up_ct_bn)
    G_5_up_c_1_out=Conv2D(192,kernel_size=(5,5),strides=(1,1), padding='same')(G_5_up_ct_relu)
    G_5_up_c_2_out=Conv2DTranspose(192,kernel_size=(5,5),strides=(2,2), activation='relu', padding='same')(G_6_up_out)
    G_5_up_0=Add()([G_5_up_c_1_out,G_5_up_c_2_out])
    G_5_up_out=Concatenate()([G_5_up_0,G_3_out])
    G_5_up_out=Dropout(0.02)(G_5_up_out)                ## 32x32 
    
    
    G_4_up_bn=BatchNormalization()(G_5_up_out)
    G_4_up_relu= Activation('relu')(G_4_up_bn)
    G_4_up_ct= Conv2DTranspose(80, kernel_size=(5,5), strides=(2,2),padding='same')(G_4_up_relu)
    G_4_up_ct_bn=BatchNormalization()(G_4_up_ct)
    G_4_up_ct_relu= Activation('relu')(G_4_up_ct_bn)
    G_4_up_c_1_out=Conv2D(128,kernel_size=(5,5),strides=(1,1), padding='same')(G_4_up_ct_relu)
    G_4_up_c_2_out=Conv2DTranspose(128,kernel_size=(3,3),strides=(2,2), activation='relu', padding='same')(G_5_up_out)
    G_4_up_0=Add()([G_4_up_c_1_out,G_4_up_c_2_out])
    G_4_up_out=Concatenate()([G_4_up_0,G_2_out])
    G_4_up_out=Dropout(0.02)(G_4_up_out)                 ## 64x64 
    
    G_3_up_bn=BatchNormalization()(G_4_up_out)
    G_3_up_relu= Activation('relu')(G_3_up_bn)
    G_3_up_ct= Conv2DTranspose(60, kernel_size=(5,5), strides=(2,2),padding='same')(G_3_up_relu)
    G_3_up_ct_bn=BatchNormalization()(G_3_up_ct)
    G_3_up_ct_relu= Activation('relu')(G_3_up_ct_bn)
    G_3_up_c_1_out=Conv2D(64,kernel_size=(5,5),strides=(1,1), padding='same')(G_3_up_ct_relu)
    G_3_up_c_2_out=Conv2DTranspose(64,kernel_size=(5,5),strides=(2,2), activation='relu', padding='same')(G_4_up_out)
    G_3_up_0=Add()([G_3_up_c_1_out,G_3_up_c_2_out])
    G_3_up_out=Concatenate()([G_3_up_0,G_1_out])
    G_3_up_out=Dropout(0.02)(G_3_up_out)                ## 128x128 
    
    
    G_2_up_bn=BatchNormalization()(G_3_up_out)
    G_2_up_relu= Activation('relu')(G_2_up_bn)
    G_2_up_ct= Conv2DTranspose(40, kernel_size=(5,5), strides=(2,2),padding='same')(G_2_up_relu)
    G_2_up_ct_bn=BatchNormalization()(G_2_up_ct)
    G_2_up_ct_relu= Activation('relu')(G_2_up_ct_bn)
    G_2_up_c_1_out=Conv2D(32,kernel_size=(5,5),strides=(1,1), padding='same')(G_2_up_ct_relu)
    G_2_up_c_2_out=Conv2DTranspose(32,kernel_size=(5,5),strides=(2,2), activation='relu', padding='same')(G_3_up_out)
    G_2_up_0=Add()([G_2_up_c_1_out,G_2_up_c_2_out])
    G_2_up_out=Concatenate()([G_2_up_0,G_in])
    G_2_up_out=Dropout(0.02)(G_2_up_out)                ## 256x256 
    
    
    G_out_0_1_1=Conv2D(16,(5,5),strides=(1,1), activation='relu',padding='same')(G_2_up_out)


    G_out = Conv2D(filters=1, kernel_size=1, padding='valid', \
                  activation=None)(G_out_0_1_1)
    #G_out = Reshape((256,256,100,1))(G_out_0)
    #G_out = permute((256,256,100,1))(G_out_0)
    G = tf.keras.models.Model(inputs=G_in, outputs=G_out)
    G.output_shape
    return G
            