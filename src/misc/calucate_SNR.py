import sys
sys.path.append('/mnt/GPT_disk/zhaoming/xzw/DiTIngModelZoo_V2/')
import numpy as np
import pandas as pd
from DiTingSrc.data.data_load_lib import get_from_DiTing


def Ross_SNR(data_before, data_after):
    return np.max(np.abs(data_after))/np.max(np.abs(data_before))

def common_SNR(data_before, data_after):
    len_data_before = len(data_before)
    power_before = 0
    for idx in range(len_data_before):
        power_before += data_before[idx]**2
    power_before /= len_data_before

    len_data_after = len(data_after)
    power_after = 0
    for idx in range(len_data_after):
        power_after += data_after[idx]**2
    power_after /= len_data_after

    return 10.0*np.log10(power_after/power_before)

if __name__ == '__main__':
    DiTing_ev_csv = pd.read_csv('/mnt/GPT_disk/DL_datasets/DiTingFinalSet/csv_files/merge_csv_pn_pg_sn_sg_330km_within.csv', dtype = {'key': str})
    
    total_lines = len(DiTing_ev_csv)
    print(total_lines)

    dest_snr_file = open('/mnt/GPT_disk/DL_datasets/DiTingFinalSet/csv_files/merge_csv_pn_pg_sn_sg_330km_within_SNR.csv', 'w')
    dest_snr_file.write('key,part,Z_P_ross_snr,Z_P_common_snr,Z_S_ross_snr,Z_S_common_snr,N_P_ross_snr, N_P_common_snr,N_S_ross_snr,N_S_common_snr,E_P_ross_snr,E_P_common_snr,E_S_ross_snr,E_S_common_snr')
    for ldx in range(total_lines):
        choice_line = DiTing_ev_csv.iloc[ldx]
        part = choice_line['part']
        key = choice_line['key']
        key_correct = key.split('.')
        key = key_correct[0].rjust(6,'0') + '.' + key_correct[1].ljust(4,'0')

        p_t = int((choice_line['p_pick'] + 30)*100)
        s_t = int((choice_line['s_pick'] + 30)*100)
        try:
            data = get_from_DiTing(part = part, key = key)

            Z_P_before = data[p_t-50:p_t,0]
            Z_P_after = data[p_t:p_t+50,0]
            Z_P_ross_snr = Ross_SNR(Z_P_before, Z_P_after)
            Z_P_common_snr = common_SNR(Z_P_before, Z_P_after)

            Z_S_before =  data[s_t-150:s_t,0]
            Z_S_after =  data[s_t:s_t+150,0]
            Z_S_ross_snr = Ross_SNR(Z_S_before, Z_S_after)
            Z_S_common_snr = common_SNR(Z_S_before, Z_S_after)

            N_P_before = data[p_t-50:p_t,0]
            N_P_after = data[p_t:p_t+50,0]
            N_P_ross_snr = Ross_SNR(N_P_before, N_P_after)
            N_P_common_snr = common_SNR(N_P_before, N_P_after)

            N_S_before =  data[s_t-150:s_t,0]
            N_S_after =  data[s_t:s_t+150,0]
            N_S_ross_snr = Ross_SNR(N_S_before, N_S_after)
            N_S_common_snr = common_SNR(N_S_before, N_S_after)        

            E_P_before = data[p_t-50:p_t,0]
            E_P_after = data[p_t:p_t+50,0]
            E_P_ross_snr = Ross_SNR(E_P_before, E_P_after)
            E_P_common_snr = common_SNR(E_P_before, E_P_after)

            E_S_before =  data[s_t-150:s_t,0]
            E_S_after =  data[s_t:s_t+150,0]
            E_S_ross_snr = Ross_SNR(E_S_before, E_S_after)
            E_S_common_snr = common_SNR(E_S_before, E_S_after) 
        except:
            print(key)
            print(part)
            continue
        
        t_line = '{:},{:},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}'.format(key, part, Z_P_ross_snr, Z_P_common_snr, Z_S_ross_snr, Z_S_common_snr, N_P_ross_snr, N_P_common_snr, N_S_ross_snr, N_S_common_snr,E_P_ross_snr, E_P_common_snr, E_S_ross_snr, E_S_common_snr) + '\n'
        
        dest_snr_file.write(t_line)
        #break

    dest_snr_file.close()

