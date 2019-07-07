import pandas as pd
from scipy.io import savemat,loadmat
path = './data_saved/right_final_data_decoded.csv'
data = pd.read_csv(path,delimiter=',',compression='gzip',header=None).values
savemat('./data_saved/data.mat',{'data':data})