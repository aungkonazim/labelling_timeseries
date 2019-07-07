import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import RobustScaler,MinMaxScaler
from scipy import signal
from scipy.stats import kurtosis
from scipy.signal import find_peaks

def preProcessing(data,Fs=25,fil_type='ppg'):
    '''
    Inputs
    data: a numpy array of shape n*10 .. the columns are timestamp,ppg red, ppg infrared,
    ppg green, acl x,y,z, gyro x,y,z
    Fs: sampling rate
    fil_type: ppg or ecg
    Output X2: preprocessed signal data
    preprocessing the data by filtering

    '''

    X0 = data[:,1:]
    X1 = signal.detrend(X0,axis=0,type='constant')
    b = signal.firls(65,np.array([0,0.2, 0.3, 3 ,3.5,Fs/2]),np.array([0, 0 ,1 ,1 ,0, 0]),
                     np.array([100*0.02,0.02,0.02]),fs=Fs)
    X2 = np.zeros((np.shape(X1)[0]-len(b)+1,data.shape[1]))
    for i in range(X2.shape[1]):
        if i in [0,4,5,6]:
            X2[:,i] = data[64:,i]
        else:
            X2[:,i] = signal.convolve(X1[:,i-1],b,mode='valid')
    return X2


def plot_with_span_selector(x,y):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
    ax1.set(facecolor='#FFFFCC')
    z = []
    z1 = []
    ax1.plot(x, y, '-')
    ax2.set_xlim(x[0], x[-1])
    ax2.set_ylim(y.min(), y.max())
    ax1.set_title('Press left mouse button and drag to test')

    ax2.set(facecolor='#FFFFCC')
    line2, = ax2.plot(x, y, '-')


    def onselect(xmin, xmax):
        indmin, indmax = np.searchsorted(x, (xmin, xmax))
        indmax = min(len(x) - 1, indmax)
        thisx = x[indmin:indmax]
        thisy = y[indmin:indmax]
        z.append(np.array(list(zip(list(thisx),list(thisy)))).reshape(-1,2))
        final_data = np.concatenate(z)
        final_data = final_data[final_data[:,0].argsort()]
        thisx = final_data[:,0]
        thisy = final_data[:,1]
        z1.extend([(indmin,indmax)])
        line2.set_data(thisx, thisy)
        ax2.set_xlim(thisx[0], thisx[-1])
        ax2.set_ylim(thisy.min(), thisy.max())
        fig.canvas.draw()

    # Set useblit=True on most backends for enhanced performance.
    span = SpanSelector(ax1, onselect, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red'))


    plt.show()
    return z1

def get_ecg_windowss(rr_interval):
    window_col,ts_col = [],[]
    ts_array = np.arange(rr_interval[0,0],rr_interval[-1,0],3000)
    for t in ts_array:
        index = np.where((rr_interval[:,0]>=t-2560)&(rr_interval[:,0]<=t+3000))[0]
        if len(index)<.8*25*5.5:
            continue
        rr_temp = rr_interval[index,:]
        window_col.append(rr_temp)
        ts_col.append(t)
    return window_col,ts_col
def combine_data_sobc(window_col,ts_col,label_data,participant,no_of_feature):
    feature_matrix = []
    user_col = []
    label_col = []
    for i,item in enumerate(window_col):
        filtered_data = preProcessing(item)
        # filtered_data1 = signal.detrend(filtered_data[:,1:4],axis=0)
        filtered_data[:,1:4] = MinMaxScaler().fit_transform(RobustScaler(quantile_range=(20,80)).fit_transform(filtered_data[:,1:4]))
        for j in [1,2,3]:
            filtered_data = np.insert(filtered_data,filtered_data.shape[1],j,axis=1)
            z = plot_with_span_selector(filtered_data[:,0],filtered_data[:,j])
            if len(z)==0:
                filtered_data[:,-1] = j
            else:
                for a,b in z:
                    filtered_data[a:b,-1] = 0
        plt.plot(filtered_data[:,0],filtered_data[:,1:4])
        plt.plot(filtered_data[:,0],filtered_data[:,7:])
        plt.show()

def get_feature(ppg_data,label_data,participant,no_of_feature):
    window_col,ts_col = get_ecg_windowss(ppg_data[:,np.array([0,2,3,4,5,6,7])])
    feature_matrix,user_col,label_col = combine_data_sobc(window_col,ts_col,label_data,participant,no_of_feature)
    return [feature_matrix,user_col,label_col]



# final_data = pickle.load(open('./data/data_for_mperf.p','rb'))
# final_data = final_data[2:4]
# pickle.dump(final_data,open('./data/data_for_mperf1.p','wb'))
final_data = pickle.load(open('./data/data_for_mperf1.p','rb'))
print(len(final_data))
final_output = [get_feature(a[0],a[1],a[2],17) for a in final_data[:1]]
# x = np.arange(0.0, 5.0, 0.01)
# y = np.sin(2*np.pi*x) + 0.5*np.random.randn(len(x))
#
# print(z)