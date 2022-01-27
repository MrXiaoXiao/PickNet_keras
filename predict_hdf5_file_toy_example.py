import os
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py

def predict_example_hdf5_file(cfgs):
    dup_num = 14
    csv_data = pd.read_csv(cfgs['Testing']['pred_csv'], dtype = {'key': str})
    csv_output_data = pd.DataFrame(columns=['key', 'pred_idx', 'prob_idx'])
    picker = tf.keras.models.load_model(cfgs['Testing']['model_path'], compile=False)
    batch_size = cfgs['Testing']['batchsize']
    # can change to generator to speed up predicting
    total_batch_num = int( len(csv_data) / batch_size ) + 1
    
    for pred_dx in range(total_batch_num):
        input_batch = np.zeros([batch_size, 1200, 1])
        for ins_dx in range(batch_size):
            line_dx = ins_dx + pred_dx * batch_size
            if line_dx >= len(csv_data):
                continue
            else:
                cur_line = csv_data.iloc[line_dx]
                
                # get theoretical arrival time or reference time to set time window
                theo_time = cur_line['P'] + np.random.randint(low=-300,high=300)
                # get data
                with h5py.File(cfgs['Testing']['pred_hdf5'], 'r') as h5_file:
                    data = h5_file.get('earthquake/' + cur_line['key'])
                    data = np.asarray(data)
                input_batch[ins_dx,:,:] = data[theo_time-600:theo_time+600,0:1]
                input_batch[ins_dx,:,:] -= np.mean(input_batch[ins_dx,:,:])
                input_batch[ins_dx,:,:] /= np.max(np.abs(input_batch[ins_dx,:,:]))
        res = picker.predict(input_batch)
        pred_sum = np.sum(res,axis=0)/dup_num
        # write to csv
        for ins_dx in range(batch_size):
            line_dx = ins_dx + pred_dx * batch_size
            if line_dx >= len(csv_data):
                continue
            else:
                key = csv_data.iloc[line_dx]['key']
                pick_idx = np.argmax(pred_sum[ins_dx,:,0])
                prob_idx = pred_sum[ins_dx,pick_idx,0]
                csv_output_data.loc[len(csv_output_data.index)] = [key, pick_idx, prob_idx]

        if cfgs['Testing']['if_plot']:
            if os.path.exists(cfgs['Testing']['imgs_save_path']):
                pass
            else:
                os.mkdir(cfgs['Testing']['imgs_save_path'])
            # check results
            check_id = np.random.randint(batch_size)
            plt.figure(figsize=(12,5))
            plt.plot(input_batch[check_id,:,0], color='k')
            
            for res_dx in range(14):
                plt.plot(res[res_dx][check_id,:,0]-res_dx*1.0 - 2.0,color='b')
            plt.plot([np.argmax(pred_sum[check_id,:,0]),np.argmax(pred_sum[check_id,:,0])], [-1,1],color='b',linewidth=3,linestyle='--')
            plt.savefig(cfgs['Testing']['imgs_save_path']+'Batch_{}_check_id_{}.png'.format(pred_dx,check_id),dpi=300)
            plt.close()
    csv_output_data.to_csv(cfgs['Testing']['output_csv'])
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility of Testing PickNet keras Beta 0.01')
    parser.add_argument('--config-file', dest='config_file', type=str, help='Path to Configuration file')

    args = parser.parse_args()
    cfgs = yaml.load(open(args.config_file), Loader=yaml.SafeLoader)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfgs['Testing']['gpu_id']
    predict_example_hdf5_file(cfgs)
    print('Done predicting')