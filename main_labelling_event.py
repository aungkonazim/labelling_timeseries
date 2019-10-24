import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import pickle



def plot_with_span_selector(x,y,y1,y2):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
    ax1.set(facecolor='#FFFFCC')
    z = []
    z1 = []
    # ax1.plot(x, y)
    # ax1.plot(x, y1)
    ax1.plot(x, y2)
    y = y2
    ax2.set_xlim(x[0], x[-1])
    ax2.set_ylim(-3, 3)
    ax1.set_ylim(-5, 5)
    ax1.set_title('Press left mouse button and drag to test')

    ax2.set(facecolor='#FFFFCC')
    line2, = ax2.plot(x, y, '-')


    def onselect(xmin, xmax):
        # print(xmin,xmax)
        indmin, indmax = np.searchsorted(x, (xmin, xmax))
        indmax = min(len(x) - 1, indmax)
        thisx = x[indmin:indmax]
        thisy = y[indmin:indmax]
        z.append(np.array(list(zip(list(thisx),list(thisy)))).reshape(-1,2))
        final_data = np.concatenate(z)
        final_data = final_data[final_data[:,0].argsort()]
        thisx = final_data[:,0]
        thisy = final_data[:,1]
        z1.append(np.array([xmin,xmax,indmin,indmax]))
        line2.set_data(thisx, thisy)
        ax2.set_xlim(thisx[0], thisx[-1])
        ax2.set_ylim(thisy.min(), thisy.max())
        fig.canvas.draw()

    # Set useblit=True on most backends for enhanced performance.
    span = SpanSelector(ax1, onselect, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red'))


    plt.show()
    if len(z1)>0:
        return np.array(z1)
    else:
        return np.zeros((0,4))





def get_feature(ppg_data,final_path,r):
    print(final_path)
    ts_array = np.arange(ppg_data[0,0],ppg_data[0,0]+3600*1000,.5*60*1000)
    data_labelled = []
    for i,t in enumerate(ts_array[:-1]):
        index = np.where((ppg_data[:,0]>=t)&(ppg_data[:,0]<ts_array[i+1]))[0]
        if len(index) < .66*25*.5*60:
            continue
        m = plot_with_span_selector(ppg_data[index,0],ppg_data[index,2],ppg_data[index,3],ppg_data[index,4])
        data_labelled.append(m)
        import os
        if len(m)>0:
            if os.path.isdir(final_path):
                # pickle.dump(np.concatenate(data_labelled),open(final_path+str(r)+'event_bad_temp.p','wb'))
                print(1)


final_data = pickle.load(open('./data_saved/data_all_mperf.p','rb'))
print(len(final_data))
import os
final_output = [get_feature(a[0],a[1],a[2]) for a in final_data[:5]]