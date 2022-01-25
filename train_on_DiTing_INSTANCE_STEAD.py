from src.data.data_generator import SimpleTrainGenerator_DIS
from keras.callbacks import ModelCheckpoint,  ReduceLROnPlateau, CSVLogger
import os
import yaml
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from src.model.PickNet import PickNet_keras

def train_PickNet(cfgs=None):
    if cfgs==None:
        print('Empty Config')
        return
    
    model = PickNet_keras(cfgs)

    # if use previous model
    if cfgs['Training']['use_previous_model_as_start']:
        model.load_weights(cfgs['Training']['previous_model_path'])

    init_dict = dict()
    # fill dict here
    # load DiTing Params
    init_dict['DiTing_hdf5_path'] = cfgs['Training']['DiTing_hdf5_path']
    init_dict['DiTing_csv_path'] = cfgs['Training']['DiTing_csv_path_train']

    # load STEAD Params
    init_dict['STEAD_hdf5_path'] = cfgs['Training']['STEAD_hdf5_path_train'] 
    init_dict['STEAD_ev_csv_path'] = cfgs['Training']['STEAD_ev_csv_path_train'] 
    init_dict['STEAD_noise_csv_path'] = cfgs['Training']['STEAD_noise_csv_path_train'] 
    # load INSTANCE Params
    init_dict['INSTANCE_ev_hdf5_path'] = cfgs['Training']['INSTANCE_ev_hdf5_path_train']
    init_dict['INSTANCE_ev_csv_path'] = cfgs['Training']['INSTANCE_ev_csv_path_train']

    init_dict['DiTing_batch_size'] = cfgs['Training']['DiTing_batch_size']
    init_dict['STEAD_batch_size'] = cfgs['Training']['STEAD_batch_size']
    init_dict['INSTANCE_batch_size'] = cfgs['Training']['INSTANCE_batch_size']
    init_dict['Noise_batch_size'] = cfgs['Training']['Noise_batch_size']
    init_dict['Empty_batch_size'] = cfgs['Training']['Empty_batch_size']

    train_data_gen = SimpleTrainGenerator_DIS(init_dict = init_dict,
                                    miniepoch = cfgs['Training']['epochs'],
                                    batch_size = cfgs['Training']['batch_size'],
                                    duplicate_num = cfgs['PickNet']['duplicate_num'],
                                    dim = cfgs['PickNet']['length'],
                                    dim_y = cfgs['PickNet']['length'],
                                    n_channels= cfgs['PickNet']['channel_num'],
                                    shift_max = cfgs['Training']['shift_max'],
                                    wave_type = cfgs['PickNet']['wave_type'])
    
    init_dict['DiTing_csv_path'] = cfgs['Training']['DiTing_csv_path_val']
    init_dict['STEAD_ev_csv_path'] = cfgs['Training']['STEAD_ev_csv_path_val'] 
    init_dict['INSTANCE_ev_csv_path'] = cfgs['Training']['INSTANCE_ev_csv_path_val']
    
    val_data_gen = SimpleTrainGenerator_DIS(init_dict = init_dict,
                                    miniepoch = cfgs['Training']['validation_steps'],
                                    batch_size = cfgs['Training']['batch_size'],
                                    duplicate_num = cfgs['PickNet']['duplicate_num'],
                                    dim = cfgs['PickNet']['length'],
                                    dim_y = cfgs['PickNet']['length'],
                                    n_channels= cfgs['PickNet']['channel_num'],
                                    shift_max = cfgs['Training']['shift_max'],
                                    wave_type = cfgs['PickNet']['wave_type'])

    print('Done Loading CSV')

    TASK_ID = cfgs['Training']['TASK_ID']

    filepath = cfgs['Training']['filepath'] + TASK_ID + '_{epoch:04d}.hdf5'
    if os.path.exists(cfgs['Training']['filepath']):
        pass
    else:
        os.makedirs(cfgs['Training']['filepath'])
    
    checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=False, mode='auto', period=1)
    print('Done Creating Generator')

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                    cooldown=0,
                                    patience=10,
                                    min_lr = 0.1e-7)
    # write traning loss log
    if os.path.exists(cfgs['Training']['train_log_dir']):
        pass
    else:
        os.makedirs(cfgs['Training']['train_log_dir'])
    csv_logger = CSVLogger(cfgs['Training']['train_log_dir'] + cfgs['Training']['TASK_ID'] + 'train_log.csv')

    hist = model.fit(train_data_gen,
                        workers=cfgs['Training']['num_works'],
                        max_queue_size=cfgs['Training']['max_queue'],
                        use_multiprocessing=False,
                        callbacks=[checkpoint,lr_reducer,csv_logger],
                        epochs=cfgs['Training']['epochs'],
                        steps_per_epoch=cfgs['Training']['steps_per_epoch'],
                        validation_data=val_data_gen,
                        validation_steps=cfgs['Training']['validation_steps']
                        )

    histpath = cfgs['Training']['histpath']  + TASK_ID + '_hist.hdf5'
    with open(histpath, 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)
    model.save(cfgs['Training']['histpath'] + TASK_ID + '_last.hdf5')

    print('Training done')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility of Training PickNet keras Beta 0.01')
    parser.add_argument('--config-file', dest='config_file', type=str, help='Path to Configuration file')

    args = parser.parse_args()
    cfgs = yaml.load(open(args.config_file), Loader=yaml.SafeLoader)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfgs['Training']['gpu_id']
    
    train_PickNet(cfgs)
    