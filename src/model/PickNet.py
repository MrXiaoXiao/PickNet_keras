import tensorflow as tf
from keras import Model, layers, optimizers, losses
import keras.backend as K
import numpy as np
from keras.optimizers import Adam
from keras.layers import Input, InputLayer, Lambda, Dense, Flatten, Conv1D, BatchNormalization, SpatialDropout1D, MaxPooling1D
from keras.layers import UpSampling1D, Cropping1D, Conv2DTranspose, Concatenate, Activation, add, Lambda, BatchNormalization
from src.loss.custom_losses import cross_entropy_balanced, BinaryFocalLossCustom
from src.model.custom_layers import Conv1DTranspose

def build_side_layer(x):
    side_out = Conv1D(1,1,padding='same',activation=None)(x)
    return side_out

def RU_layer(x, x_res, upscale_out, upscale_res):
    res_inputs_deconv = Conv1DTranspose(x_res,1,upscale_res*2,upscale_res,padding='same')
    concate = Concatenate(axis=-1)([x, res_inputs_deconv])

    res_outputs = Conv1D(1,3,padding='same')(concate)
    side_outputs = Conv1DTranspose(res_outputs,1,upscale_out*2,upscale_out,activation='sigmoid')

    return res_outputs, side_outputs

def PickNet_keras(cfgs=None):
    """
    Stacked Unet
    """
    input_data = layers.Input(shape=(cfgs['PickNet']['length'],cfgs['PickNet']['channel_num']),name='input')

    # block 1
    conv1_1 = Conv1D(64, cfgs['PickNet']['b1_convw'],padding='same',activation='relu')(input_data)
    conv1_2 = Conv1D(64, cfgs['PickNet']['b1_convw'],padding='same',activation='relu')(conv1_1)
    pool1 = MaxPooling1D(2,2)(conv1_2)

    # block 2
    conv2_1 = Conv1D(128, cfgs['PickNet']['b2_convw'],padding='same',activation='relu')(pool1)
    conv2_2 = Conv1D(128, cfgs['PickNet']['b2_convw'],padding='same',activation='relu')(conv2_1)
    pool2 = MaxPooling1D(2,2)(conv2_2)

    # block 3
    conv3_1 = Conv1D(256, cfgs['PickNet']['b3_convw'],padding='same',activation='relu')(pool2)
    conv3_2 = Conv1D(256, cfgs['PickNet']['b3_convw'],padding='same',activation='relu')(conv3_1)
    conv3_3 = Conv1D(256, cfgs['PickNet']['b3_convw'],padding='same',activation='relu')(conv3_2)
    pool3 = MaxPooling1D(2,2)(conv3_3)

    # block 4
    conv4_1 = Conv1D(512, cfgs['PickNet']['b3_convw'],padding='same',activation='relu')(pool3)
    conv4_2 = Conv1D(512, cfgs['PickNet']['b3_convw'],padding='same',activation='relu')(conv4_1)
    conv4_3 = Conv1D(512, cfgs['PickNet']['b3_convw'],padding='same',activation='relu')(conv4_2)
    pool4 = MaxPooling1D(2,2)(conv4_3)

    # block 5
    conv5_1 = Conv1D(512, cfgs['PickNet']['b3_convw'],padding='same',activation='relu')(pool4)
    conv5_2 = Conv1D(512, cfgs['PickNet']['b3_convw'],padding='same',activation='relu')(conv5_1)
    conv5_3 = Conv1D(512, cfgs['PickNet']['b3_convw'],padding='same',activation='relu')(conv5_2)
    
    # side layers
    side_1_1 = build_side_layer(conv1_1)
    side_1_2 = build_side_layer(conv1_2)

    side_2_1 = build_side_layer(conv2_1)
    side_2_2 = build_side_layer(conv2_2)

    side_3_1 = build_side_layer(conv3_1)
    side_3_2 = build_side_layer(conv3_2)
    side_3_3 = build_side_layer(conv3_3)

    side_4_1 = build_side_layer(conv4_1)
    side_4_2 = build_side_layer(conv4_2)
    side_4_3 = build_side_layer(conv4_3)

    side_5_1 = build_side_layer(conv5_1)
    side_5_2 = build_side_layer(conv5_2)
    side_5_3 = build_side_layer(conv5_3)

    # Deep-to-shallow-fashion RSRN
    dsnout_5_3 = Conv1DTranspose(side_5_3,1,16*2,16,padding='same')
    residual_5_2, dsnout_5_2  = RU_layer(side_5_2, side_5_3, 16, 1)
    residual_5_1, dsnout_5_1  = RU_layer(side_5_1, residual_5_2, 16, 1)

    residual_4_3, dsnout_4_3  = RU_layer(side_4_3, residual_5_1, 8, 2)
    residual_4_2, dsnout_4_2  = RU_layer(side_4_2, residual_4_3, 8, 1)
    residual_4_1, dsnout_4_1  = RU_layer(side_4_1, residual_4_2, 8, 1)

    residual_3_3, dsnout_3_3  = RU_layer(side_3_3, residual_4_1, 4, 2)
    residual_3_2, dsnout_3_2  = RU_layer(side_3_2, residual_3_3, 4, 1)
    residual_3_1, dsnout_3_1  = RU_layer(side_3_1, residual_3_2, 4, 1)

    residual_2_2, dsnout_2_2  = RU_layer(side_2_2, residual_3_1, 2, 2)
    residual_2_1, dsnout_2_1  = RU_layer(side_2_1, residual_2_2, 2, 1)

    residual_1_2, dsnout_1_2  = RU_layer(side_1_2, residual_2_1, 1, 2)
    _, dsnout_1_1  = RU_layer(side_1_1, residual_1_2, 1, 1)
    
    side_outputs = [dsnout_1_1,dsnout_1_2,
                dsnout_2_1,dsnout_2_2,
                dsnout_3_1,dsnout_3_2,dsnout_3_3,
                dsnout_4_1,dsnout_4_2,dsnout_4_3,
                dsnout_5_1,dsnout_5_2,dsnout_5_3]
    
    fuse_output_concat = Concatenate(axis=-1)(side_outputs)
    fuse_output = Conv1D(1,1,padding='same',activation='sigmoid')(fuse_output_concat)
    
    all_outputs = side_outputs + [fuse_output]

    model = Model(inputs=input_data,outputs=all_outputs)    

    if cfgs['Training']['opt'] == 'adam':
        opt = optimizers.Adam(cfgs['Training']['lr'])
    else:
        pass

    loss_list = list()
    if cfgs['Training']['loss'] == 'BCE':
        for _ in range(len(all_outputs)):
            loss_list.append('binary_crossentropy')
    elif cfgs['Training']['loss'] == 'CB_BCE':
        for _ in range(len(all_outputs)):
            loss_list.append(cross_entropy_balanced)
    elif cfgs['Training']['loss'] == 'Focal_CB_BCE':
        for _ in range(len(all_outputs)):
            loss_list.append(BinaryFocalLossCustom(gamma=2))

    model.compile(loss=loss_list,optimizer=opt)
    
    model.summary()

    return model