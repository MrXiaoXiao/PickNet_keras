import sys
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from random import shuffle
from src.data.data_load_lib import get_from_STEAD, get_from_INSTANCE
from src.data.data_load_lib import get_instance_for_training

class SimpleTrainGenerator(tf.keras.utils.Sequence):
    """
    generator data for PickNet Keras Traning
    """
    def __init__(self, 
                init_dict=None,
                batch_size=64, 
                dim=1200,
                dim_y=1200,
                n_channels=1,
                n_classes=1, 
                miniepoch=2000,
                duplicate_num=14,
                wave_type = 'P',
                shift_max = 300):
        """
        Init
        """
        # load STEAD Params
        self.STEAD_hdf5_path = init_dict['STEAD_hdf5_path']
        self.STEAD_ev_csv_path = init_dict['STEAD_ev_csv_path']
        self.STEAD_ev_csv = pd.read_csv(self.STEAD_ev_csv_path)
        self.STEAD_ev_keys = list(self.STEAD_ev_csv['trace_name'])
        
        self.STEAD_noise_csv_path = init_dict['STEAD_noise_csv_path']
        self.STEAD_noise_csv = pd.read_csv(self.STEAD_noise_csv_path)
        self.STEAD_no_keys = list(self.STEAD_noise_csv['trace_name'])
        
        # load INSTANCE Params
        self.INSTANCE_ev_hdf5_path = init_dict['INSTANCE_ev_hdf5_path']
        self.INSTANCE_ev_csv_path = init_dict['INSTANCE_ev_csv_path']
        self.INSTANCE_ev_csv = pd.read_csv(self.INSTANCE_ev_csv_path)

        self.STEAD_batch_size = init_dict['STEAD_batch_size']
        self.INSTANCE_batch_size = init_dict['INSTANCE_batch_size']
        self.Noise_batch_size = init_dict['Noise_batch_size']
        self.Empty_batch_size = init_dict['Empty_batch_size']
        self.dim = dim
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.miniepoch = miniepoch
        self.duplicate_num = duplicate_num
        self.wave_type = wave_type
        self.on_epoch_end()
        self.shift_max = shift_max

    def __len__(self):
        """
        number of batches per epoch
        """
        return self.miniepoch

    def __getitem__(self, index):
        """
        create batch for input
        """
        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        y = np.zeros((self.batch_size, self.dim, self.n_classes))
        start_dx = 0
        # get from STEAD dataset
        total_choice_keys = np.random.choice(self.STEAD_ev_keys, self.STEAD_batch_size)
        for batch_dx in range(start_dx, start_dx + self.STEAD_batch_size):
            choice_key = total_choice_keys[int(batch_dx-start_dx)]
            temp_data_X, temp_data_Y = get_instance_for_training(dataset='STEAD',
                                                                dataset_path=self.STEAD_hdf5_path,
                                                                data_length = self.dim,
                                                                data_channel_num = self.n_channels,
                                                                key = choice_key,
                                                                wave_type=self.wave_type,
                                                                shift_max = self.shift_max)
            X[batch_dx,:,:] = temp_data_X[:,:]
            y[batch_dx,:,:] = temp_data_Y[:,:]
        
        start_dx += self.STEAD_batch_size

        # get from INSTANCE dataset
        total_lines = len(self.INSTANCE_ev_csv) - 1
        total_choice_ids = np.random.choice(total_lines, 5*self.INSTANCE_batch_size)
        
        for batch_dx in range(start_dx, start_dx + self.INSTANCE_batch_size):
            choice_id = total_choice_ids[int(batch_dx-start_dx)]
            choice_line = self.INSTANCE_ev_csv.iloc[choice_id]
            key = choice_line['trace_name']
            p_t = choice_line['trace_P_arrival_sample']
            s_t = choice_line['trace_S_arrival_sample']

            temp_data_X, temp_data_Y = get_instance_for_training(dataset='INSTANCE',
                                                                dataset_path=self.INSTANCE_ev_hdf5_path,
                                                                data_length = self.dim,
                                                                data_channel_num = self.n_channels,
                                                                key = key,
                                                                wave_type = self.wave_type,
                                                                shift_max = self.shift_max,
                                                                p_t = p_t,
                                                                s_t = s_t)
            X[batch_dx,:,:] = temp_data_X[:,:]
            y[batch_dx,:,:] = temp_data_Y[:,:]

        start_dx += self.INSTANCE_batch_size
        
        # get from noise dataset
        total_choice_keys = np.random.choice(self.STEAD_no_keys, self.STEAD_batch_size)
        for batch_dx in range(start_dx, start_dx + self.Noise_batch_size):
            choice_key = total_choice_keys[batch_dx - start_dx]
            data, _ , _ = get_from_STEAD(choice_key)
            data_length = np.shape(data)[0]
            noise_start_dx = int(np.random.randint(0,data_length - self.dim))
            if self.wave_type == 'P':
                X[batch_dx,:,:] = data[noise_start_dx:noise_start_dx + self.dim,2:3]
            else:
                X[batch_dx,:,:] = data[noise_start_dx:noise_start_dx + self.dim,0:2]
            
            for chn_dx in range(self.n_channels):
                X[batch_dx,:,chn_dx] -= np.mean(X[batch_dx,:,chn_dx] )
                norm_factor = np.max(np.abs(X[batch_dx,:,chn_dx] ))
                if norm_factor == 0:
                    pass
                else:
                    X[batch_dx,:,chn_dx]  /= norm_factor
        
        Y = list()
        for _ in range(self.duplicate_num):
            Y.append(y[:,:,0:1])
        return (X, Y)
    
    def on_epoch_end(self):
        """
        epoch end
        """
        return

class SimpleTrainGeneratorINSTANCE(tf.keras.utils.Sequence):
    """
    generator data for PickNet Keras Traning on INSTANCE
    """
    def __init__(self, 
                init_dict=None,
                batch_size=64, 
                dim=1200,
                dim_y=1200,
                n_channels=1,
                n_classes=1, 
                miniepoch=2000,
                duplicate_num=14,
                wave_type = 'P',
                shift_max = 300):
        """
        Init
        """        
        # load INSTANCE Params
        self.INSTANCE_ev_hdf5_path = init_dict['INSTANCE_ev_hdf5_path']
        self.INSTANCE_ev_csv_path = init_dict['INSTANCE_ev_csv_path']
        self.INSTANCE_ev_csv = pd.read_csv(self.INSTANCE_ev_csv_path)
        self.INSTANCE_ev_csv = self.INSTANCE_ev_csv.dropna(subset=['trace_S_arrival_sample'])


        self.INSTANCE_batch_size = init_dict['INSTANCE_batch_size']
        self.INSTANCE_ev_indexs = np.arange(len(self.INSTANCE_ev_csv),dtype=np.int)

        self.dim = dim
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.miniepoch = miniepoch
        self.duplicate_num = duplicate_num
        self.wave_type = wave_type
        self.on_epoch_end()
        self.shift_max = shift_max

    def __len__(self):
        """
        number of batches per epoch
        """
        return int(len(self.INSTANCE_ev_csv)/self.batch_size)

    def __getitem__(self, index):
        """
        create batch for input
        """
        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        y = np.zeros((self.batch_size, self.dim, self.n_classes))
        start_dx = 0
        # get from INSTANCE dataset
        total_choice_ids = self.INSTANCE_ev_indexs[index*self.INSTANCE_batch_size:(index+1)*self.INSTANCE_batch_size]
        for batch_dx in range(start_dx, start_dx + self.INSTANCE_batch_size):
            choice_id = total_choice_ids[int(batch_dx-start_dx)]
            choice_line = self.INSTANCE_ev_csv.iloc[choice_id]
            key = choice_line['trace_name']
            p_t = choice_line['trace_P_arrival_sample']
            s_t = choice_line['trace_S_arrival_sample']
            
            temp_data_X, temp_data_Y = get_instance_for_training(dataset='INSTANCE',
                                                                dataset_path=self.INSTANCE_ev_hdf5_path,
                                                                data_length = self.dim,
                                                                data_channel_num = self.n_channels,
                                                                key = key,
                                                                wave_type = self.wave_type,
                                                                shift_max = self.shift_max,
                                                                p_t = p_t,
                                                                s_t = s_t)
            X[batch_dx,:,:] = temp_data_X[:,:]
            y[batch_dx,:,:] = temp_data_Y[:,:]

        start_dx += self.INSTANCE_batch_size
                
        Y = list()
        for _ in range(self.duplicate_num):
            Y.append(y[:,:,0:1])
        return (X, Y)
    
    def on_epoch_end(self):
        """
        epoch end
        """
        shuffle(self.INSTANCE_ev_indexs)
        return