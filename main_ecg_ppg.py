import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel,delayed
from joblib import delayed,Parallel
from scipy import signal
import shutil
import math
from sklearn.preprocessing import RobustScaler
import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as pat
from sklearn.preprocessing import RobustScaler
from scipy import signal
from scipy.stats import skew,kurtosis
import numpy as np
from scipy import signal
from scipy.stats import iqr
import pickle
ppg_window_col,ecg_window_col = pickle.load(open('/data/azim/window_ecg_ppg.p','rb'))
from scipy.signal import find_peaks
from scipy.stats import iqr
from scipy import interpolate
Fs = 25
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff=2.5, fs=25, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

import heartpy as hp
import matplotlib.pyplot as plt
path = './images_ecg_ppg/'
for threshold in np.linspace(.1,.9,10):
    if not os.path.isdir(path+str(threshold)):
        os.makedirs(path+str(threshold))
    new_path = path+str(threshold)+'/'
    f_data = []
    for i,window in enumerate(ppg_window_col):
        ecg_window = ecg_window_col[i]
        data_col = []
        for k in range(0,3,1):
            if np.mean(window[:,11+k])<=0:
                continue
            index = np.where(window[:,11+k]>threshold)[0]
    #         if len(index)<=.7*25*60:
    #             continue
            y = window[:,2+k]
            prob_series = window[:,11+k].reshape(-1)
            x = window[:,0] - window[0,0]
            x1 = [x[0]] 
            for j in range(1,len(x),1):
                if x[j]-x[j-1]<50:
                    x1.append(x[j])
                else:
                    n = (x[j]-x[j-1])//40
                    x1.extend([x[j-1]+h*40 for h in range(1,int(n)+1,1) if (x[j-1]+h*40) <= (x[j]-40)])
                    x1.append(x[j])
            x_new = np.array(x1)
            f = interpolate.interp1d(x,y)
            y_new = f(x_new)
            y_new1 = np.convolve(y_new, np.ones((10,))/10, mode='same')
            y_new2 = butter_lowpass_filter(y_new1)
            f = interpolate.interp1d(x,prob_series)
            prob_new = f(x_new)
    #         data = hp.get_data(y_new)
    #         working_data, measures = hp.process(y_new2, 25)
    #         print(working_data.keys())
    #         plt.figure()
    #         hp.plotter(working_data, measures)
            peak_loc, peak_dict = find_peaks(y_new2, distance=Fs*.5,height=np.percentile(y_new2,40))
            ts = []
            rr = []
            p1 = peak_loc[0]
            indices = []
            for m in range(1,len(peak_loc),1):
                p2=peak_loc[m]
                if 400<(x_new[p2]-x_new[p1])<1200 and np.mean(prob_new[p1:p2])>=threshold:
    #                 if len(rr)<3:
                    ts.append(x_new[p2])
                    rr.append(x_new[p2]-x_new[p1])
                    indices.append(p2)
    #                 elif np.mean(rr)-3*np.std(rr)<=(x_new[p2]-x_new[p1])<=np.mean(rr)+3*np.std(rr):
    #                     ts.append(x_new[p2])
    #                     rr.append(x_new[p2]-x_new[p1])
    #                     indices.append(p2)
                p1=p2
            indices = np.array(indices)
            if len(indices)<2:
                continue
            fig,ax = plt.subplots(figsize=(10,6),nrows=3,ncols=1)
            ax[0].plot(x_new,y_new2)
            ax[0].plot(x_new[peak_loc],y_new2[peak_loc],'*')
            ax[0].plot(x_new[peak_loc],y_new2[peak_loc],'r*')
            ax[0].plot(ts,y_new2[indices],'g*')
            ax[1].vlines(x_new,0,prob_new)
            ax[2].plot(ecg_window[:,0]-window[0,0],ecg_window[:,1])
            ax[2].plot(ts,rr,'y')
            plt.savefig(new_path+str(i)+'_'+str(k)+'.pdf',dps=1000)
#             pickle.dump([ts,rr,x_new,y_new2,prob_new,ecg_window,window],open(new_path+str(i)+'_'+str(k)+'.p','wb'))
            f_data.append([ts,rr,x_new,y_new2,prob_new,ecg_window,window])
            plt.close('all')
    pickle.dump(f_data,open(new_path+'data.p','wb'))
#             plt.close(ax)
#             plt.show()