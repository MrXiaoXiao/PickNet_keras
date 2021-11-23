import os

if __name__ == '__main__':
    for idx in range(31,60):
        os.system('cp /public/data1/data_for_xiao/DiTingFinalSet/hdf5_files/CN_total_part_{}.hdf5 /mnt/GPT_disk/DL_datasets/DiTingFinalSet/hdf5_files/'.format(idx))

        os.system('cp /public/data1/data_for_xiao/DiTingFinalSet/csv_files/CN_total_keys_part_{}.csv /mnt/GPT_disk/DL_datasets/DiTingFinalSet/csv_files/'.format(idx))
