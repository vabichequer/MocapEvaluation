import csv
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import collections
import numpy as np
from numpy.testing._private.utils import tempdir
import pandas as pd
import math
import seaborn as sns
import pandas as pd

FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/MotionMatching/Assets/output/new/Mixamo/"))

def floatToString(inputValue):
    return ('%.15f' % inputValue).rstrip('0').rstrip('.')

def PieChart(labels, sizes):
    fig = plt.figure()
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    plt.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)

    return fig

def read_root(filename):
    file = []
    with open(FOLDER_FILES + '/' + filename + ".csv", newline='') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')        
        for row in read_csv:
            file.append(row[0])
    csvfile.close()
    return file

def read_csv(radius, prefix="", sufixes=""):
    speed = []
    nbr_blend = 0
    nbr_trans = 0
    for sufix in sufixes:
        with open(FOLDER_FILES + '/' + prefix + str(radius) + "_" + sufix + ".csv", newline='') as csvfile:
            read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
            if (sufix == "speed_dump"):
                for row in read_csv:
                    speed.append(float(row[0]))
            elif (sufix == "nbr_blended_frames"):          
                nbr_blend = float(next(read_csv)[0])
                nbr_blend = math.floor(nbr_blend)
            elif (sufix == "nbr_transitions"):
                nbr_trans = float(next(read_csv)[0])
                nbr_trans = math.floor(nbr_trans)
        csvfile.close()
    return speed, nbr_blend, nbr_trans

def AdditionalPlots(radiuses, path):
    if (os.path.isdir(path + '/images/')):
        pass
    else:
        os.mkdir(path + '/images/')

    df = pd.DataFrame()

    for r in radiuses:
        speed, nbr_blend, nbr_trans = read_csv(r, sufixes=["speed_dump", "nbr_blended_frames", "nbr_transitions"])
        
        dict = {'Radius':r,
                'Blended frames':nbr_blend,
                'Transitions':nbr_trans}

        df = df.append(dict, ignore_index=True)

        occurences_clips = collections.Counter(np.around(np.asarray(speed), 1))

        fig = PieChart(occurences_clips.keys(), occurences_clips.values())
        
        fig.savefig(path + '/images/' + str(r) + "_speed_distribution.svg")

    ax = df.plot(x='Radius', kind='bar', stacked=True,
            title='Comparison between radius')

    ax.set_ylabel("Number of frames")

    fig = ax.get_figure()

    fig.savefig(path + '/images/' + str(r) + "_blend_vs_transitions.svg")

    if (PLOT_ENABLED):
        plt.show()

def ProcessData(speed, orientation, temp_r, time_windows, mus, stds, dp, error):
    path = FOLDER_FILES + '/' + speed + '/' + orientation + '/'
    os.system('python C:/Users/vabicheq/Documents/mocap-evaluation/animation_analysis.py False ' + path + ' ' + temp_r)
    os.system('python C:/Users/vabicheq/Documents/mocap-evaluation/main.py animation_dataset.csv ' + temp_r + ' ' + time_windows + ' False ' + path)
    os.system('python C:/Users/vabicheq/Documents/mocap-evaluation/offset_graph.py ' + path + ' ' + temp_r)

    radiuses = [floatToString(float(x)) for x in temp_r.split(',')]
    for r in radiuses:
        loaded = np.load(path + str(r) + '_offset_error_mu_std.npy')
        mus.append(loaded[0])
        stds.append(loaded[1])

        loaded = np.load(path + str(r) + "_desiredPoint_and_error.npz")
        loaded['dp']
        dp.append(loaded['dp'])
        error.append(loaded['error'])

if (len(sys.argv) > 1):
    PLOT_ENABLED = eval(sys.argv[1])
else:
    PLOT_ENABLED = False

radiuses = [float(x) for x in read_root("radiuses")]
speeds = read_root("linearspeeds")
mus = []
stds = []

r = iter(radiuses)
c = next(r)

desired_points = []
std_linear_and_angular_errors = []

if(os.path.isfile(FOLDER_FILES + "/all_desiredPoints_and_errors.npz")):
    loaded = np.load(FOLDER_FILES + "/all_desiredPoints_and_errors.npz")
    desired_points = loaded['all_dps']
    std_linear_and_angular_errors = loaded['all_std_errors']
else:
    for speed in speeds:    
        temp_r = ""
        time_windows = "1,"
        while c < 0:
            #right
            temp_r += str(abs(c)) + ','
            time_windows += "1,"
            c = next(r)

        ProcessData(speed, "Right", temp_r[:-1], time_windows[:-1], mus, stds, desired_points, std_linear_and_angular_errors)

        temp_r = ""
        time_windows = "1,"
        while c == 0:
            #straight
            temp_r += str(c) + ','
            time_windows += "1,"
            c = next(r)

        ProcessData(speed, "Straight", temp_r[:-1], time_windows[:-1], mus, stds, desired_points, std_linear_and_angular_errors)
        
        temp_r = ""
        time_windows = "1,"
        while c > 0:
            #left
            temp_r += str(c) + ','
            time_windows += "1,"
            try:
                c = next(r)
            except:
                break
        
        ProcessData(speed, "Left", temp_r[:-1], time_windows[:-1], mus, stds, desired_points, std_linear_and_angular_errors)

    np.savez_compressed(FOLDER_FILES + "/all_desiredPoints_and_errors", all_dps=desired_points, all_std_errors=std_linear_and_angular_errors)

plt.figure(figsize=(16, 9))
loaded = np.load(FOLDER_FILES + "/coverage_plot.npz", allow_pickle=True)
data = loaded['data'].item()
z = loaded['z']
data = pd.DataFrame(data=data)
coverage = sns.scatterplot(data=data, x='theta', y='speed', c=z, cmap='mako')

for dp, std in zip(desired_points, std_linear_and_angular_errors):    
    std_linear = std[0]
    std_angular = std[1]
    plt.errorbar(dp[0], dp[1], xerr=std_angular, yerr=std_linear, color='red', capsize=5.0)
    coverage.scatter(dp[0], dp[1], color='lime', zorder=2)

plt.savefig(FOLDER_FILES + "/coverage_plot.png")

desired_points = np.asarray(desired_points)

x = np.unique(desired_points[:,0])
y = np.unique(desired_points[:,1])

mus = np.asarray(mus)

X, Y = np.meshgrid(x, y)
mus = mus.reshape((len(y), len(x)))
plt.contour(X, Y, mus, 45, cmap='RdGy')
plt.colorbar()

plt.savefig(FOLDER_FILES + "/offset_plot.png")

plt.show()
