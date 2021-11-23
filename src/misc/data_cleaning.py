import h5py
import obspy
import numpy as np
import pandas as pd
# for debug
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """
    # choose instance with both P and S in INSTANCE
    INSTANCE_csv_file_name = '/public/data1/data_for_xiao/INSTANCE/metadata_Instance_events.csv'
    INSTANCE_csv = pd.read_csv(INSTANCE_csv_file_name)
    INSTANCE_both_p_s_csv = INSTANCE_csv.loc[INSTANCE_csv.trace_P_arrival_sample != np.nan & INSTANCE_csv.trace_S_arrival_sample != np.nan]
    
    # seperate noise and ev in STEAD
    STEAD_csv_file_name = '/public/data1/data_for_xiao/STEAD/metadata_10_29_19.csv'
    
    STEAD_csv = pd.read_csv(STEAD_csv_file_name)
    STEAD_csv_eq = STEAD_csv.loc[STEAD_csv.trace_name.str.contains('_EV')]
    STEAD_csv_eq.to_csv('/public/data1/data_for_xiao/STEAD/metadata_eq.csv')

    STEAD_csv_no = STEAD_csv.loc[STEAD_csv.trace_name.str.contains('_NO')]
    STEAD_csv_no.to_csv('/public/data1/data_for_xiao/STEAD/metadata_no.csv')
    """