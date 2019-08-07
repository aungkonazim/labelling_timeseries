import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as pat
from sklearn.preprocessing import RobustScaler
from scipy import signal
# %matplotlib inline
from scipy.stats import skew,kurtosis
import numpy as np
from scipy import signal
from scipy.stats import iqr
import pickle
def get_metric(x):
    f,pxx = signal.welch(x,fs=25,nperseg=len(x),nfft=10000)
    pxx = np.abs(pxx)
    pxx = pxx/max(pxx)
    peaks_loc1,_ = signal.find_peaks(pxx[np.where((f>.8)&(f<2.5))[0]],height=.01)
    if len(peaks_loc1)==0:
        return False
    else:
        return True
path = '/home/mullah/Downloads/mperf'
destination_dir = './data_images/'
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
                ppg_data = pd.read_csv(final_path+left_right[r],compression='gzip',sep=',',header=None).values
                ts_array = np.arange(ppg_data[0,0],ppg_data[-1,0],2.5*1000*100)
                plt.plot(ppg_data[:,0],ppg_data[:,4])
                plt.show()
                # if not os.path.isdir(destination_dir+participant+'/'+file+'/'+str(r)):
                #     os.makedirs(destination_dir+participant+'/'+file+'/'+str(r))
                # final_path_1 = destination_dir+participant+'/'+file+'/'+str(r)+'/'
                # data_col = {}
                # count = 0
                # for i,t in enumerate(ts_array[:-1]):
                #     index = np.where((ppg_data[:,0]>=t)&(ppg_data[:,0]<t+2.5*1000))[0]
                #     if len(index) < .66*25*2.5:
                #         continue
                #     ppg_window = ppg_data[index,:]
                #     ppg_window[:,2:5] = RobustScaler().fit_transform(ppg_window[:,2:5])
                #     window = np.concatenate([ppg_window[:,0].reshape(-1,1),signal.detrend(ppg_window[:,4]).reshape(-1,1)],
                #                             axis=1)
                #     if not get_metric(window[:,1]):
                #         continue
                #     plt.figure()
                #     plt.plot((window[:,0]-window[0,0])/1000,window[:,1])
                #     plt.savefig(final_path_1+str(count)+'.png')
                #     plt.close('all')
                #     data_col[str(count)] = window
                #     count+=1
                # pickle.dump(data_col,open(final_path_1+'window_col.p','wb'))
            except Exception as e:
                print(e)
                continue