import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
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
window_col_good = []
for participant in participants:
    file_list = os.listdir(path+'/'+participant)
    for file in file_list:
        final_path = path+'/'+participant+'/'+file+'/'
        files = os.listdir(final_path)
        for r in range(2):
            try:
                data = pickle.load(open(final_path+str(r)+'event.p','rb'))
                ppg_data = pd.read_csv(final_path+left_right[r],compression='gzip',sep=',',header=None).values
                participants_col.append(participant)
                data = data[data[:,0].argsort()]
                total_segments+=data.shape[0]
                annotated+=(data[-1,1] - data[0,0])/3600000
                for a in data:
                    total_duration +=(a[1]-a[0])/3600000
                    duration_col.append((a[1]-a[0])/1000)
                    # index = np.where((ppg_data[:,0]>=a[0])&(ppg_data[:,0]<=a[1]))[0]
                    # window_col_good.append(ppg_data[index,:])
                    # plt.plot(ppg_data[index,4])
                    # plt.show()
                ts_array = np.arange(ppg_data[0,0],ppg_data[0,0]+3600*1000,.5*60*1000)
                data_labelled = []
                for i,t in enumerate(ts_array[:-1]):
                    index = np.where((ppg_data[:,0]>=t)&(ppg_data[:,0]<t+1*60*1000))[0]
                    if len(index) < .66*25*1*60:
                        continue
                    ppg_window = ppg_data[index,:]
                    upper_envelope = np.abs(signal.hilbert(ppg_window[:,4]))
                #     upper_envelope = upper_envelope -np.mean(upper_envelope)
                #     test_col.append(np.std(upper_envelope))
                    plt.figure(figsize=(10,5))
                    plt.plot(ppg_window[:,0],ppg_window[:,4],label='Green Channel PPG',linewidth=2,color='b')
                    # plt.plot(ppg_window[:,0],upper_envelope,label='Upper Envelope',linewidth=2,color='r')
                    ax = plt.gca()
                    # # ax.set_title('oh no')
                    rect_col = []
                    count = 1
                    for a in data:
                        if ppg_window[0,0]<a[0]<ppg_window[-1,0] and ppg_window[0,0]<a[1]<ppg_window[-1,0]:
                            rect = pat.Rectangle((a[0],-1),a[1]-a[0],2,facecolor='g',alpha=.6,label='Annotated Segment '+str(count))
                            count+=1
                            # rect_col.append(rect)
                            ax.add_patch(rect)
                    if count==1:
                        plt.close('all')
                        continue
                    plt.ylim([-2,2.5])
                    plt.xticks([ppg_window[p,0] for p in range(0,len(ppg_window[:,0]),100)],
                               [int(np.round((ppg_window[p,0]-ppg_window[0,0])/(1000))) for p in range(0,len(ppg_window[:,0]),100)],
                               fontsize=13,weight='bold')
                    plt.xlabel('Seconds',fontsize=13,weight='bold')
                    # plt.ylabel('Bandpass Filtered PPG',fontsize=20)
                    plt.yticks([])
                    plt.legend(ncol=3,loc='upper right',fontsize=13)
                    plt.savefig('./images/'+str(t)+'.pdf',dps=1000)
                    plt.close('all')
                    plt.show()
            except Exception as e:
                print(e)
                a=1
print(total_segments,total_duration,annotated,len(np.unique(participants_col)))
plt.hist(duration_col,bins=100)
plt.ylabel('No of Segments',fontsize=16)
# plt.xlabel('Seconds',fontsize=16)
plt.xticks([int(i) for i in range(0,int(max(duration_col)),5)],[str(i) for i in range(0,int(max(duration_col)),5)],
           fontsize=18,rotation='vertical',weight='bold')
plt.yticks(weight='bold')
plt.savefig('./images/'+str('distribution')+'.pdf',dps=1000)
plt.show()
#
# import pickle
# pickle.dump(window_col_good,open('./data_saved/window_col_good.p','wb'))

# print(np.mean(duration_col))
# import scipy
# # print(scipy.stats.kstest(test_col,'norm'))
# plt.hist(test_col)
# plt.show()