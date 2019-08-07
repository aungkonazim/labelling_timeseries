import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew,kurtosis
import numpy as np
from scipy import signal
from scipy.stats import iqr

def get_metric(x):
    f,pxx = signal.welch(x,fs=25,nperseg=len(x),nfft=1000)
    pxx = np.abs(pxx)
    pxx = pxx/max(pxx)
    peaks_loc1,_ = signal.find_peaks(pxx[np.where((f>=.8)&(f<=2.5))[0]],height=.1)
    peaks_loc2,_ = signal.find_peaks(pxx,height=.1)
    if len(peaks_loc1)==0:
        return -1
    elif len(peaks_loc1)==1 and len(peaks_loc2)==1:
        return 1
    else:
        return 0
path = '/home/mullah/Downloads/mperf'
destination_dir = './data_images_all/'
participants = os.listdir(path)
left_right = ['left_final_data_decoded.csv', 'right_final_data_decoded.csv']
window_col = {}
label_col = []
count=0
final_data = pickle.load(open('./data_saved/final_data.p','rb'))
for i,a in enumerate(final_data[::-1]):
    ppg_data,ecg_rr,r,participant,file = a
    try:
        ts_array = np.arange(ecg_rr[0,0],ecg_rr[0,0]+7200*1000,2.5*1000*20)
        # # if not os.path.isdir(destination_dir+participant+'/'+file+'/'+str(r)):
        # #     os.makedirs(destination_dir+participant+'/'+file+'/'+str(r))
        # # final_path_1 = destination_dir+participant+'/'+file+'/'+str(r)+'/'
        # # data_col = {}
        # # count = 0
        for i,t in enumerate(ts_array[:-1]):
            index_ecg = np.where((ecg_rr[:,0]>=t)&(ecg_rr[:,0]<t+2.5*1000))[0]
            if len(index_ecg) < 2:
                continue
            ecg_window = ecg_rr[index_ecg,:]
            hr = 60000/np.mean(ecg_window[:,1])
            var = np.std(60000/ecg_window[:,1])
            index = np.where((ppg_data[:,0]>=t)&(ppg_data[:,0]<t+2.5*1000))[0]
            if len(index) < .95*25*2.5:
                continue
            ppg_window = ppg_data[index,:]
            ppg_window[:,2:5] = RobustScaler().fit_transform(ppg_window[:,2:5])
            window = np.concatenate([ppg_window[:,0].reshape(-1,1),signal.detrend(ppg_window[:,4]).reshape(-1,1)],
                                    axis=1)

            m = get_metric(window[:,1])
            fig,ax = plt.subplots(nrows=2,ncols=1)
            ax[1].plot((window[:,0]-window[0,0])/1000,window[:,1])
            f,pxx = signal.welch(window[:,1],fs=25,nperseg=len(window[:,1]),nfft=1000)
            pxx = np.abs(pxx)
            pxx = pxx/max(pxx)
            peaks_loc1,_ = signal.find_peaks(pxx[np.where((f>=.8)&(f<=2.5))[0]],height=.1)
            ax[0].plot(f,pxx,linestyle='--',color='r')
            peaks_loc2,_ = signal.find_peaks(pxx,height=.1)
            ax[0].plot(f[peaks_loc2],pxx[peaks_loc2],'*',color='y')
            ax[0].plot(f[np.where((f>=.8)&(f<=2.5))[0]][peaks_loc1],pxx[np.where((f>=.8)&(f<=2.5))[0]][peaks_loc1],'*',color='c')
            ax[0].set_ylabel('Power Spectral Density')
            ax[0].vlines([.8,2.5],0,1)
            ax[0].vlines(hr/60,0,1,colors='b')
            ax[1].set_ylabel('Normalized PPG Window')
            ax[0].set_title(str(len(window)))
            plt.show()
        #     if m==1:
        #         plt.savefig(destination_dir+'excellent'+'/'+str(count)+'.pdf',dps=1000)
        #     elif m==-1:
        #         plt.savefig(destination_dir+'worst'+'/'+str(count)+'.pdf',dps=1000)
        #     else:
        #         plt.savefig(destination_dir+'undecided1'+'/'+str(count)+'.pdf',dps=1000)
        #         plt.savefig(destination_dir+'undecided2'+'/'+str(count)+'.pdf',dps=1000)
        #         plt.savefig(destination_dir+'undecided3'+'/'+str(count)+'.pdf',dps=1000)
        #     window_col[count] = window
        #     count+=1
        #     plt.close('all')

        #     plt.savefig(final_path_1+str(count)+'.png')
        #     plt.close('all')
        #     data_col[str(count)] = window
        #     count+=1
        # pickle.dump(data_col,open(final_path_1+'window_col.p','wb'))
        # print(count1*100/count)count+=1
    except Exception as e:
        print(e)
# pickle.dump(window_col,open('./data_saved/window_col_all.p','wb'))