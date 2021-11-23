import pandas as pd
from pathlib import Path

def shuffle_and_split_csv(ori_csv_name, train_p=0.9, test_p=0.05, val_p=0.05):
    # load
    df = pd.read_csv(ori_csv_name)
    # shuffle
    df = df.sample(frac = 1).reset_index(drop=True)
    # split
    total_len = len(df)
    train_len = int(total_len*train_p)
    test_len = int(total_len*test_p)
    val_len = int(total_len*val_p)
    
    df_train = df[:train_len]
    df_train.to_csv(ori_csv_name + '.train.csv')

    df_test = df[train_len:train_len + test_len]
    df_test.to_csv(ori_csv_name + '.test.csv')

    df_val = df[train_len + test_len:]
    df_val.to_csv(ori_csv_name + '.val.csv')

    return

if __name__ == '__main__':
    shuffle_and_split_csv('/mnt/GPT_disk/DL_datasets/DiTingFinalSet/csv_files/merge_csv_pn_pg_sn_sg_330km_within_clear.csv')
    #shuffle_and_split_csv('/mnt/GPT_disk/DL_datasets/STEAD/metadata_eq.csv')
    #shuffle_and_split_csv('/mnt/GPT_disk/DL_datasets/INSTANCE/metadata_Instance_events_both_p_s.csv')