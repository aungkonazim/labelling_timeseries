import os
import pandas as pd
path = '/home/mullah/Downloads/mperf'
participants = os.listdir(path)
left_right = ['left_final_data_decoded.csv', 'right_final_data_decoded.csv']
data_col = []
for participant in participants:
    file_list = os.listdir(path+'/'+participant)
    for file in file_list:
        final_path = path+'/'+participant+'/'+file+'/'
        files = os.listdir(final_path)
        for r in range(2):
            try:
                wrist_data = pd.read_csv(final_path+left_right[r],compression='gzip',sep=',',header=None).values
                data_col.append([wrist_data,final_path,r])
            except Exception as e:
                print(e)
import pickle
pickle.dump(data_col,open('./data_saved/data_from_mperf.p','wb'))