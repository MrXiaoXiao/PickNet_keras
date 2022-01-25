import h5py
import obspy
import numpy as np
import pandas as pd
# for debug
import matplotlib.pyplot as plt
from numpy.fft import irfft, rfftfreq
from numpy import sqrt, newaxis
from numpy.random import normal

"""
functions for loading data from various data set
"""

def train_instance_plot(temp_data_X, temp_data_Y, save_name):
    plt.figure(figsize=(12,8))
    plot_dx = 0

    for chdx in range(np.shape(temp_data_X)[1]):
        plt.plot(temp_data_X[:, chdx]/np.max(np.abs(temp_data_X[:, chdx])) + plot_dx*2,color='k')
        plot_dx += 1
    
    plt.plot(temp_data_Y[:,0]-2,color='g')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_name ,dpi=300)
    plt.show()
    plt.close()

    return

def get_instance_for_training(dataset='STEAD',
                                dataset_path = '/mnt/GPT_disk/DL_datasets/STEAD/',
                                data_length = 1200,
                                data_channel_num = 1,
                                wave_type = 'P',
                                part = None,
                                key = None,
                                shift_max = 300,
                                p_t = None,
                                s_t = None
                                ):
    
    temp_data_X = np.zeros([data_length,data_channel_num])
    temp_data_Y = np.zeros([data_length,1])

    # data array E N Z 0 1 2
    if dataset == 'DiTing':
        data_t = get_from_DiTing(part = part, key = key, h5file_path = dataset_path)
        data = np.zeros([20000,3])
        data[:,0] = data_t[:,2]
        data[:,1] = data_t[:,1]
        data[:,2] = data_t[:,0]
        p_t = int(3000 + p_t*100)
        s_t = int(3000 + s_t*100)

    elif dataset == 'STEAD':
        data, p_t, s_t = get_from_STEAD(key = key, h5file_path = dataset_path)

    elif dataset == 'INSTANCE':
        data_t = get_from_INSTANCE(key = key, h5file_path = dataset_path)
        #print(np.shape(data))
        data = np.zeros([12000,3])
        data[:,0] = data_t[0,:]
        data[:,1] = data_t[1,:]
        data[:,2] = data_t[2,:]

    shift = np.random.randint(high=shift_max,low=shift_max*(-1))

    if wave_type == 'P':
        center = p_t
    else:
        center = s_t

    half_len = data_length//2
    start_dx = center + shift - half_len
    end_dx = center + shift + half_len
    
    temp_start = 0
    temp_end = data_length

    if start_dx < 0:
        temp_start = (-1)*start_dx
        start_dx = 0

    elif end_dx > len(data):
        temp_end = (end_dx - len(data))
        end_dx = len(data)
    else:
        pass

    # check if start_dx or end_dx or temp_start or temp_end is incorrect
    temp_start = int(temp_start)
    temp_end = int(temp_end)
    start_dx = int(start_dx)
    end_dx = int(end_dx)
    
    if temp_start < 0 or temp_end > len(temp_data_X) or start_dx < 0 or end_dx > len(data):
        return np.zeros([data_length,data_channel_num]), np.zeros([data_length,1]) 

    if wave_type == 'P':
        temp_data_X[temp_start:temp_end,:] = data[start_dx:end_dx,2:3]
    else:
        temp_data_X[temp_start:temp_end,:] = data[start_dx:end_dx,0:2]

    temp_data_Y[int(half_len - shift)] = 1.0

    reverse_factor = np.random.choice([-1,1])
    rescale_factor = np.random.uniform(low=0.5,high=1.5)

    # normalize
    for chn_dx in range(data_channel_num):
        temp_data_X[:,chn_dx] -= np.mean(temp_data_X[:,chn_dx])
        norm_factor = np.max(np.abs(temp_data_X[:,chn_dx]))
        if norm_factor == 0:
            pass
        else:
            temp_data_X[:,chn_dx] /= norm_factor
        
        temp_data_X[:,chn_dx] *= rescale_factor
        temp_data_X[:,chn_dx] *= reverse_factor
    
    return temp_data_X, temp_data_Y

def get_from_STEAD(key=None, 
                    h5file_path='/mnt/GPT_disk/DL_datasets/STEAD/waveforms.hdf5'):
    """
    Input:
    key, h5file_path
    
    Output:
    data, p_t, s_t
    """
    
    HDF5 = h5py.File(h5file_path, 'r')
    
    if key.split('_')[-1] == 'EV':
        dataset = HDF5.get('earthquake/local/'+str(key))
        p_t = int(dataset.attrs['p_arrival_sample'])
        s_t = int(dataset.attrs['s_arrival_sample'])
    elif key.split('_')[-1] == 'NO':
        dataset = HDF5.get('non_earthquake/noise/'+str(key))
        p_t = None
        s_t = None
    
    data = np.array(dataset).astype(np.float32)

    return data, p_t, s_t

def get_from_INSTANCE(key=None, 
                    h5file_path='/mnt/GPT_disk/DL_datasets/INSTANCE/Instance_events_counts.hdf5'):
    HDF5 = h5py.File(h5file_path, 'r')
    dataset = HDF5.get('data/'+str(key))
    data = np.array(dataset).astype(np.float32)
    return data

def get_from_DiTing(part=None,
                    key=None,
                    h5file_path='/mnt/GPT_disk/DL_datasets/DiTingFinalSet/hdf5_files/'):

    """
    Input:
    part, key, h5file_folder

    Output:
    data
    """
    with h5py.File(h5file_path + 'DiTing330km_part_{}.hdf5'.format(part), 'r') as f:
        dataset = f.get('earthquake/'+str(key))    
        data = np.array(dataset).astype(np.float32)

    return data