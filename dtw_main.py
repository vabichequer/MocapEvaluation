from dtaidistance import dtw_ndim
from dtaidistance import dtw
from dtaidistance import dtw_ndim_visualisation as dtwvis_ndim
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math 
import csv

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

x = []
y = []

with open('root_mocap.csv', newline='') as csvfile:
    read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(read_csv, None)
    for row in read_csv:
        x.append(float(row[1]))
        y.append(float(row[3]))

x = np.asarray(x)
y = np.asarray(y)

curve = np.arange(0, math.pi / 2, .05)
d = 999999.0

printProgressBar(0, len(y) - len(curve), prefix = 'Progress:', suffix = 'Complete', length = 50)

for i in range(0, len(y) - len(curve)): 
    printProgressBar(i, len(y) - len(curve), prefix = 'Progress:', suffix = 'Complete')
    #s1 = np.vstack((x, np.sin(x))).T
    #s2 = np.vstack((x, np.sin(x - 1))).T

    s1_new = np.sin(curve)  #stats.zscore(np.sin(x))
    s2_new = y[i:len(s1_new)+i] #np.sin(y) #stats.zscore(np.sin(y))

    #d, p = dtw_ndim.warping_paths(s1, s2)
    d_new, p_new = dtw.warping_paths(s1_new, s2_new) #, window=25, psi=2)

    if (d_new < d):
        d = d_new
        
        p = p_new

        s1 = s1_new
        s2 = s2_new

print(d)
best_path = dtw.best_path(p)
fig, ax = dtwvis.plot_warpingpaths(s1, s2, p)
fig2, ax2 = dtwvis.plot_warping(s1, s2, best_path)
plt.show()

