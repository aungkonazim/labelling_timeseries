import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
path = '/home/mullah/Downloads/mperf'
participants = os.listdir(path)
left_right = ['left_final_data_decoded.csv', 'right_final_data_decoded.csv']
data_col = []
total_segments = 0
total_duration = 0
annotated = 0
participants_col = []
duration_col = []
for participant in participants:
    file_list = os.listdir(path+'/'+participant)
    for file in file_list:
        final_path = path+'/'+participant+'/'+file+'/'
        files = os.listdir(final_path)
        for r in range(2):
            try:
                data = pickle.load(open(final_path+str(r)+'event.p','rb'))
                wrist_data = pd.read_csv(final_path+left_right[r],compression='gzip',sep=',',header=None).values
                participants_col.append(participant)
                data = data[data[:,0].argsort()]
                total_segments+=data.shape[0]
                annotated+=(data[-1,1] - data[0,0])/3600000
                for a in data:
                    total_duration +=(a[1]-a[0])/3600000
                    duration_col.append((a[1]-a[0])/1000)
            except Exception as e:
                # print(e)
                a=1
print(total_segments,total_duration,annotated,len(np.unique(participants_col)))
plt.hist(duration_col,bins=100)
plt.ylabel('No of Segments')
plt.xlabel('Seconds')
plt.xticks([int(i) for i in range(0,int(max(duration_col)),2)],[str(i) for i in range(0,int(max(duration_col)),2)],rotation='vertical')

plt.show()