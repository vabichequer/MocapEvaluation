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

FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/MotionMatching/Assets/output/new/Mixamo/"))

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

def CallMainProgram(speed, orientation, temp_r, time_windows):
    path = FOLDER_FILES + '/' + speed + '/' + orientation + '/'
    os.system('python C:/Users/vabicheq/Documents/mocap-evaluation/animation_analysis.py False ' + path + ' ' + temp_r)
    os.system('python C:/Users/vabicheq/Documents/mocap-evaluation/main.py animation_dataset.csv ' + temp_r + ' ' + time_windows + ' False ' + path)

if (len(sys.argv) > 1):
    PLOT_ENABLED = eval(sys.argv[1])
else:
    PLOT_ENABLED = False

radiuses = [float(x) for x in read_root("radiuses")]
speeds = read_root("linearspeeds")

r = iter(radiuses)
c = next(r)

for speed in speeds:    
    temp_r = ""
    time_windows = "1,"
    while c < 0:
        #right
        temp_r += str(abs(c)) + ','
        time_windows += "1,"
        c = next(r)

    CallMainProgram(speed, "Right", temp_r[:-1], time_windows[:-1])

    temp_r = ""
    time_windows = "1,"
    while c == 0:
        #straight
        temp_r += str(c) + ','
        time_windows += "1,"
        c = next(r)

    CallMainProgram(speed, "Straight", temp_r[:-1], time_windows[:-1])
    
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
    
    CallMainProgram(speed, "Left", temp_r[:-1], time_windows[:-1])