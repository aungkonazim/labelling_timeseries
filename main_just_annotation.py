import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')
from matplotlib.patches import Rectangle
import matplotlib.patches as pat
from scipy import signal
path = '/home/mullah/Downloads/mperf'
participants = os.listdir(path)
left_right = ['left_final_data_decoded.csv', 'right_final_data_decoded.csv']
data_col = []
total_segments = 0
total_duration = 0
annotated = 0
participants_col = []
duration_col = []
test_col = []
window_col = []
for participant in participants:
    file_list = os.listdir(path+'/'+participant)
    for file in file_list:
        final_path = path+'/'+participant+'/'+file+'/'
        files = os.listdir(final_path)
        for r in range(2):
            try:
                data = pickle.load(open(final_path+str(r)+'bkpnts.p','rb'))
                data1 = pickle.load(open(final_path+str(r)+'event.p','rb'))
                ppg_data = pd.read_csv(final_path+left_right[r],compression='gzip',sep=',',header=None).values
                participants_col.append(final_path+str(r))
                data1 = data1[data1[:,0].argsort()]
                total_segments+=data.shape[0]-1
                annotated+=(data[-1,0] - data[0,0])/3600000
                for i,a in enumerate(data):
                    if i==0:
                        if (data[i,0]-data[0,0])/60000 < 10:
                            total_duration +=(data[i,0]-data[0,0])/3600000
                            duration_col.append((data[i,0]-data[0,0])/1000)
                    else:
                        if (data[i,0]-data[i-1,0])/60000 < 10:
                            total_duration +=(data[i,0]-data[i-1,0])/3600000
                            duration_col.append((data[i,0]-data[i-1,0])/1000)
                ts_array = np.array([ppg_data[0,0]]+list(data[:,0]))
                for i,t in enumerate(ts_array[:-1]):
                    index = np.where((ppg_data[:,0]>=t)&(ppg_data[:,0]<ts_array[i+1]))[0]
                    if len(index) < .66*25*(ts_array[i+1]-t)/1000:
                        continue
                    ppg_window = ppg_data[index,:]
                    window_col.append(ppg_window)
                ts_array = np.arange(ppg_data[0,0],ppg_data[0,0]+3600*1000,.5*60*1000)
                data_labelled = []
                for i,t in enumerate(ts_array[:-1]):
                    index = np.where((ppg_data[:,0]>=t)&(ppg_data[:,0]<ts_array[i+1]))[0]
                    if len(index) < .66*25*1*60:
                        continue
                    ppg_window = ppg_data[index,:]
                    window_col.append(ppg_window)
                    upper_envelope = np.abs(signal.hilbert(ppg_window[:,4]))
                    upper_envelope = upper_envelope -np.mean(upper_envelope)
                    test_col.append(np.std(upper_envelope))
                    cpnt = data[np.where((data[:,0]>=t)&(data[:,0]<ts_array[i+1]))[0],0]
                    plt.figure(figsize=(16,6))
                    plt.plot(ppg_window[:,0],ppg_window[:,4],label='Green PPG Channel',linewidth=2,color='b')
                    # plt.plot(ppg_window[:,0],ppg_window[:,5]*2/16384-3.2,label='Accelerometer X',linewidth=1)
                    # plt.plot(ppg_window[:,0],ppg_window[:,6]*2/16384-3.2,label='Accelerometer Y',linewidth=1)
                    # plt.plot(ppg_window[:,0],ppg_window[:,7]*2/16384-3.2,label='Accelerometer Z',linewidth=1)
                    # plt.plot(ppg_window[:,0],[-3.2]*len(ppg_window[:,0]),label='Accelerometer Zero Line',linewidth=1)
                    # plt.vlines(cpnt,-2,2,label='Detected Change points',linewidth=3,color='black')
                    ax = plt.gca()
                    # # ax.set_title('oh no')
                    rect_col = []
                    count = 1
                    for a in data1:
                        if ppg_window[0,0]<a[0]<ppg_window[-1,0] and ppg_window[0,0]<a[1]<ppg_window[-1,0] and (a[1]-a[0])>1000*2:
                            rect = pat.Rectangle((a[0],-1.5),a[1]-a[0],3,facecolor='g',alpha=.5,label='Annotated Segment '+str(count))
                            count+=1
                            # rect_col.append(rect)
                            ax.add_patch(rect)
                    plt.ylim([-2,3])
                    plt.xticks([ppg_window[p,0] for p in range(0,len(ppg_window[:,0]),100)],
                               [int(np.round((ppg_window[p,0]-ppg_window[0,0])/(1000))) for p in range(0,len(ppg_window[:,0]),100)],
                               fontsize=14,rotation=90)
                    plt.xlabel('Seconds',fontsize=14)
                    # plt.ylabel('Bandpass Filtered Value')
                    plt.yticks([])
                    plt.legend(loc='best',ncol=3,fontsize=14)
                    # plt.savefig('./images/'+str(t)+'.pdf',dps=2000)
                    # plt.close('all')
                    plt.show()
            except Exception as e:
                print(e)
                a=1
print(total_segments,total_duration,annotated,len(np.unique(participants_col)))
import pickle
pickle.dump(window_col,open('./data_saved/window_col.p','wb'))
# plt.hist(duration_col,bins=1500)
# plt.ylabel('No of Segments')
# plt.xlabel('Seconds')
# # plt.xticks([int(i) for i in range(0,int(max(duration_col)),3)],[str(i) for i in range(0,int(max(duration_col)),3)],rotation='vertical')
# # plt.savefig('./images/'+str('distribution_all_data')+'.pdf',dps=1000)
# plt.show()
# import scipy
# print(scipy.stats.mode(duration_col))
# import scipy
# # print(scipy.stats.kstest(test_col,'norm'))
# # plt.hist(test_col)
# # plt.show()