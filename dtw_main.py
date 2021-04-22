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

# 0 -> pi/2
# pi/2 -> pi
# pi -> 3/2 * ip
# 3/2 * pi -> 2 * pi

start = [0, math.pi / 2, math.pi, (3 * math.pi) / 2]
end = [math.pi / 2, math.pi, (3 * math.pi) / 2, 2 * math.pi]

curves_x = []
curves_y = []

for i in range(0, 4):
    temp_x = np.arange(start[i], end[i], .05)
    curves_x.append(temp_x - (i * math.pi / 2))
    curves_y.append(np.sin(temp_x))

temp = np.arange(0, math.pi / 2, .05)
curves_x.append(temp)
curves_y.append(np.zeros(temp.shape))

curves_y.append(temp)
curves_x.append(np.zeros(temp.shape))
        
fig, axs = plt.subplots(2, 3)

for i in range(0, 2):
    for j in range(0, 3):
        k = i * 3 + j
        axs[i, j].plot(curves_x[k], curves_y[k])
        axs[i, j].set_title('Axis ' + str(k))

fig.savefig("curves.png")

d = 999999.0

resolution = len(np.arange(0, math.pi / 2, .05))

for i in range(0, 6):

    sx_new = curves_x[i]
    sy_new = curves_y[i]

    printProgressBar(0, len(y) - len(sx_new), prefix = 'Progress:', suffix = 'Complete', length = 50)

    for j in range(0, len(y) - len(sx_new)): 
        printProgressBar(j, len(y) - len(sx_new), prefix = 'Progress:', suffix = 'Complete')

        x_mocap = x[j:len(sx_new) + j] 
        y_mocap = y[j:len(sy_new) + j] 

        dx_new, px_new = dtw.warping_paths(sx_new, x_mocap)
        dy_new, py_new = dtw.warping_paths(sy_new, y_mocap)

        d_new = dx_new + dy_new

        if (d_new < d):
            d = d_new
            
            px = px_new
            py = py_new

            sx = sx_new
            sy = sy_new
            sx_mocap = x_mocap
            sy_mocap = y_mocap

    best_path_x = dtw.best_path(px)
    fig_x, ax_x = dtwvis.plot_warpingpaths(sx, sx_mocap, px)
    fig2_x, ax2_x = dtwvis.plot_warping(sx, sx_mocap, best_path_x)

    best_path_y = dtw.best_path(py)
    fig_y, ax_y = dtwvis.plot_warpingpaths(sy, sy_mocap, py)
    fig2_y, ax2_y = dtwvis.plot_warping(sy, sy_mocap, best_path_y)
    
    fig_x.savefig(str(i) + "x.png")
    fig2_x.savefig(str(i) + "x_2.png")
    fig_y.savefig(str(i) + "y.png")
    fig2_y.savefig(str(i) + "y_2.png")

