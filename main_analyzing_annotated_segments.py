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
                data = pickle.load(open(final_path+str(r)+'event_good.p','rb'))
                # ppg_data = pd.read_csv(final_path+left_right[r],compression='gzip',sep=',',header=None).values
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
                # ts_array = np.arange(ppg_data[0,0],ppg_data[0,0]+3600*1000,1*60*1000)
                # data_labelled = []
                # for i,t in enumerate(ts_array[:-1]):
                #     index = np.where((ppg_data[:,0]>=t)&(ppg_data[:,0]<ts_array[i+1]))[0]
                #     if len(index) < .66*25*1*60:
                #         continue
                #     ppg_window = ppg_data[index,:]
                #     upper_envelope = np.abs(signal.hilbert(ppg_window[:,4]))
                #     upper_envelope = upper_envelope -np.mean(upper_envelope)
                #     test_col.append(np.std(upper_envelope))
                    # plt.figure(figsize=(16,8))
                    # plt.plot(ppg_window[:,0],ppg_window[:,4],label='Green PPG Channel',linewidth=2,color='b')
                    # ax = plt.gca()
                    # # ax.set_title('oh no')
                    # rect_col = []
                    # count = 1
                    # for a in data:
                    #     if ppg_window[0,0]<a[0]<ppg_window[-1,0] and ppg_window[0,0]<a[1]<ppg_window[-1,0]:
                    #         rect = pat.Rectangle((a[0],-2),a[1]-a[0],5,facecolor='y',alpha=.5,label='Annotated Segment '+str(count))
                    #         count+=1
                    #         # rect_col.append(rect)
                    #         ax.add_patch(rect)
                    # plt.ylim([-2,2])
                    # plt.xticks([ppg_window[p,0] for p in range(0,len(ppg_window[:,0]),100)],
                    #            [np.round((ppg_window[p,0]-ppg_window[0,0])/(1000)) for p in range(0,len(ppg_window[:,0]),100)])
                    # plt.xlabel('Seconds')
                    # plt.ylabel('Bandpass Filtered Value')
                    # plt.yticks([])
                    # plt.legend()
                    # plt.savefig('./images/'+str(t)+'.pdf',dps=1000)
                    # plt.close('all')
                    # plt.show()
            except Exception as e:
                # print(e)
                a=1
print(total_segments,total_duration,annotated,len(np.unique(participants_col)))
# plt.hist(duration_col,bins=100)
# plt.ylabel('No of Segments')
# plt.xlabel('Seconds')
# plt.xticks([int(i) for i in range(0,int(max(duration_col)),3)],[str(i) for i in range(0,int(max(duration_col)),3)],rotation='vertical')
# plt.savefig('./images/'+str('distribution')+'.pdf',dps=1000)
# plt.show()
#
# import pickle
# pickle.dump(window_col_good,open('./data_saved/window_col_good.p','wb'))

# print(np.mean(duration_col))
# import scipy
# # print(scipy.stats.kstest(test_col,'norm'))
# plt.hist(test_col)
# plt.show()