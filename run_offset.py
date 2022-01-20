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

def ProcessData(speed, orientation, temp_r, mus, stds, dp):
    path = FOLDER_FILES + '/' + speed + '/' + orientation + '/'
    os.system('python C:/Users/vabicheq/Documents/mocap-evaluation/offset_graph.py ' + path + ' ' + temp_r)

    radiuses = [floatToString(float(x)) for x in temp_r.split(',')]
    for r in radiuses:
        loaded = np.load(path + str(r) + '_offset_error_mu_std.npy')
        mus.append(loaded[0])
        stds.append(loaded[1])

        loaded = np.load(path + str(r) + "_desiredPoint_and_error.npz")
        loaded['dp']
        dp.append(loaded['dp'])

if (len(sys.argv) > 1):
    PLOT_ENABLED = eval(sys.argv[1])
else:
    PLOT_ENABLED = False

radiuses = [float(x) for x in read_root("radiuses")]
speeds = read_root("linearspeeds")
mus = []
stds = []
desired_points = []

r = iter(radiuses)
c = next(r)

for speed in speeds:
    temp_r = ""
    while c < 0:
        #right
        temp_r += str(abs(c)) + ','
        c = next(r)

    ProcessData(speed, "Right", temp_r[:-1], mus, stds, desired_points)

    temp_r = ""
    while c == 0:
        #straight
        temp_r += str(c) + ','
        c = next(r)

    ProcessData(speed, "Straight", temp_r[:-1], mus, stds, desired_points)
    
    temp_r = ""
    while c > 0:
        #left
        temp_r += str(c) + ','
        try:
            c = next(r)
        except:
            break
    
    ProcessData(speed, "Left", temp_r[:-1], mus, stds, desired_points)

desired_points = np.asarray(desired_points)

x = np.unique(desired_points[:,0])
y = np.unique(desired_points[:,1])

mus = np.asarray(mus)

X, Y = np.meshgrid(x, y)
mus = mus.reshape((len(y), len(x)))
plt.contour(X, Y, mus, 45, cmap='RdGy')
plt.colorbar()

plt.show()