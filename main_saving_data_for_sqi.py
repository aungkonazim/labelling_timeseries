import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as pat
from sklearn.preprocessing import RobustScaler
from scipy import signal
path = '/home/mullah/Downloads/mperf'
participants = os.listdir(path)
left_right = ['left_final_data_decoded.csv', 'right_final_data_decoded.csv']
window_col = []
label_col = []
for participant in participants:
    file_list = os.listdir(path+'/'+participant)
    for file in file_list:
        final_path = path+'/'+participant+'/'+file+'/'
        files = os.listdir(final_path)
        for r in range(2):
            try:
                data = pickle.load(open(final_path+str(r)+'event_good.p','rb'))
                ppg_data = pd.read_csv(final_path+left_right[r],compression='gzip',sep=',',header=None).values
                data = data[data[:,0].argsort()]
                ts_array = np.arange(ppg_data[0,0],ppg_data[-1,0],2.5*1000)
                data_labelled = []
                for i,t in enumerate(ts_array[:-1]):
                    index = np.where((ppg_data[:,0]>=t)&(ppg_data[:,0]<ts_array[i+1]))[0]
                    if len(index) < .66*25*2.5:
                        continue
                    ppg_window = ppg_data[index,:]
                    ppg_window[:,2:5] = RobustScaler().fit_transform(ppg_window[:,2:5])
                    check = 0
                    for a in data:
                        if ppg_window[0,0]>=a[0] and ppg_window[-1,0]<=a[1]:
                            check = 1
                            window_col.append(ppg_window)
                            label_col.append(check)
                            break

            except Exception as e:
                print(e)
                a=1
            # try:
            #     data = pickle.load(open(final_path+str(r)+'event_good.p','rb'))
            #     ppg_data = pd.read_csv(final_path+left_right[r],compression='gzip',sep=',',header=None).values
            #     data = data[data[:,0].argsort()]
            #     ts_array = np.arange(ppg_data[0,0],ppg_data[-1,0],2.5*1000)
            #     data_labelled = []
            #     for i,t in enumerate(ts_array[:-1]):
            #         index = np.where((ppg_data[:,0]>=t)&(ppg_data[:,0]<ts_array[i+1]))[0]
            #         if len(index) < .66*25*2.5:
            #             continue
            #         ppg_window = ppg_data[index,:]
            #         ppg_window[:,2:5] = RobustScaler().fit_transform(ppg_window[:,2:5])
            #         check = 0
            #         for a in data:
            #             if ppg_window[0,0]>=a[0] and ppg_window[-1,0]<=a[1]:
            #                 check = 0
            #                 window_col.append(ppg_window)
            #                 label_col.append(check)
            #                 break
            #
            # except Exception as e:
            #     print(e)

print(len(window_col))
import pickle
pickle.dump([window_col,label_col],open('./data_saved/window_col_sqi_good.p','wb'))