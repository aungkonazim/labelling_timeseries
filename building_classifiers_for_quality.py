
# import pickle
# import os
# import numpy as np
# from scipy import signal
# from scipy.stats import iqr,skew,kurtosis
# import matplotlib.pyplot as plt
# path = './data_images_all/temp/'
# final_col = pickle.load(open('./data_saved/window_col_all_processed.p','rb'))
# window_col = {}
# count = 0
# directories = ['acceptable/','unacceptable/','undecided/']
# for i,a in enumerate(list(final_col.keys())):
#     for window in final_col[a]:
#         plt.figure()
#         plt.plot(window[:,0]-window[0,0],window[:,1])
#         plt.savefig(path+directories[i]+str(count)+'.pdf')
#         window_col[count] = window
#         count+=1
#         plt.close('all')
#
# pickle.dump(window_col,open('./data_saved/window_col_all1.p','wb'))

# import pickle
# import os
# import matplotlib.pyplot as plt
# window_col = pickle.load(open('./data_saved/window_col_all1.p','rb'))
# def get_keys(good,path):
#     final_files = []
#     for f in good:
#         file_col = [int(a.split('.')[0]) for a in os.listdir(path+f)]
#         final_files.extend(file_col)
#     return final_files
#
# keys = list(range(7000))
# path = './data_images_all/temp/'
# bad = ['unacceptable']
# undecided = ['undecided']
# bad_files = get_keys(bad,path)
# undecided_files = get_keys(undecided,path)
# good_files =  list(set(keys).difference(set(bad_files).union(set(undecided_files))))
# print(len(good_files),len(bad_files),len(undecided_files))
# final_col = {'good':[],'bad':[],'undecided':[]}
# label = list(final_col.keys())
# final_path = './data_images_all/temp/'
# directories = ['acceptable/','unacceptable/','undecided/']
# import shutil
# for a in directories:
#     shutil.rmtree(final_path+a)
# for i,a in enumerate([good_files,bad_files,undecided_files]):
#     for f in a:
#         window = window_col[f]
#         plt.figure()
#         plt.plot(window[:,0],window[:,1])
#         plt.savefig(final_path+directories[i]+str(f)+'.pdf')
#         plt.close('all')


import pickle
import os
window_col = pickle.load(open('./data_saved/window_col_all1.p','rb'))
def get_keys(good,path):
    final_files = []
    for f in good:
        file_col = [int(a.split('.')[0]) for a in os.listdir(path+f)]
        final_files.extend(file_col)
    return final_files

keys = list(range(7000))
path = './data_images_all/temp/'
# good = ['acceptable/','excellent/']
bad = ['unacceptable']
undecided = ['undecided']
# good_files = get_keys(good,path)
bad_files = get_keys(bad,path)
undecided_files = get_keys(undecided,path)
good_files =  list(set(keys).difference(set(bad_files).union(set(undecided_files))))
print(len(good_files),len(bad_files),len(undecided_files))
final_col = {'good':[],'bad':[],'undecided':[]}
label = list(final_col.keys())
print(label)
for i,a in enumerate([good_files,bad_files,undecided_files]):
    for f in a:
        final_col[label[i]].append(window_col[f])

print(len(list(final_col.values())[0]),len(list(final_col.values())[1]),len(list(final_col.values())[2]))
pickle.dump(final_col,open('./data_saved/window_col_all_processed.p','wb'))




# import pickle
# import os
# import numpy as np
# from scipy import signal
# from scipy.stats import iqr,skew,kurtosis
# import matplotlib.pyplot as plt
# final_col = pickle.load(open('./data_saved/window_col_all_processed.p','rb'))
# label = list(final_col.keys())
# feature = np.zeros((0,5))
# for i,k in enumerate(label):
#     window_col = final_col[k]
#     for window in window_col:
#         f,pxx = signal.welch(window[:,1],fs=25,nperseg=len(window[:,1]),nfft=1000)
#         pxx = np.abs(pxx)
#         pxx = pxx/max(pxx)
#         temp = [skew(window[:,1]),kurtosis(window[:,1]),iqr(window[:,1]),np.trapz(pxx[np.where((f>=.8)&(f<=2.5))[0]])/np.trapz(pxx),i]
#         feature = np.concatenate((feature,np.array(temp).reshape(-1,5)))
#
#
# pickle.dump(feature,open('./data_saved/window_col_all_processed_feature.p','wb'))


# import pickle
# import os
# import numpy as np
# from scipy import signal
# from scipy.stats import iqr,skew,kurtosis
# import matplotlib.pyplot as plt
# from sklearn import linear_model
# from sklearn.model_selection import cross_val_predict
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score,f1_score
# from sklearn.model_selection import StratifiedKFold
# from imblearn.ensemble import BalancedRandomForestClassifier,RUSBoostClassifier
# from copy import deepcopy
# feature = pickle.load(open('./data_saved/window_col_all_processed_feature.p','rb'))
# def get_metrics(feature,check):
#     if check in ['good']:
#         feature[np.where(feature[:,-1]==1)[0],-1] = -1
#         feature[np.where(feature[:,-1]==2)[0],-1] = -1
#         feature[np.where(feature[:,-1]==0)[0],-1] = 1
#     else:
#         feature[np.where(feature[:,-1]==0)[0],-1] = -1
#         feature[np.where(feature[:,-1]==2)[0],-1] = -1
#     # clf = RandomForestClassifier(n_estimators=100)
#     clf = RUSBoostClassifier()
#     cv = StratifiedKFold(n_splits=5)
#     y_pred = cross_val_predict(clf,feature[:,:-1],feature[:,-1],cv=cv.split(feature[:,:-1],feature[:,-1]))
#     y_true = feature[:,-1]
#     print(f1_score(y_true,y_pred),precision_score(y_true,y_pred),recall_score(y_true,y_pred),confusion_matrix(y_true,y_pred))
#
# get_metrics(deepcopy(feature),'good')
# get_metrics(deepcopy(feature),'bad')