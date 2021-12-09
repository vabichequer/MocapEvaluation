import csv
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import collections
import numpy as np
import pandas as pd
import math

FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/MotionMatching/Assets/output/Mixamo/1.5"))

def PieChart(labels, sizes):
    fig = plt.figure()
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    plt.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)

    return fig

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

radiuses = [5, 10, 15]

os.system('python animation_analysis.py False ' + FOLDER_FILES)
os.system('python main.py animation_dataset.csv 5,10,15 1,1,1,1 True ' + FOLDER_FILES)

df = pd.DataFrame()

if (os.path.isdir(FOLDER_FILES + '/images/')):
    pass
else:
    os.mkdir(FOLDER_FILES + '/images/')

if (len(sys.argv) > 1):
    PLOT_ENABLED = eval(sys.argv[1])
else:
    PLOT_ENABLED = False

for r in radiuses:
    speed, nbr_blend, nbr_trans = read_csv(r, sufixes=["speed_dump", "nbr_blended_frames", "nbr_transitions"])
    
    dict = {'Radius':r,
            'Blended frames':nbr_blend,
            'Transitions':nbr_trans}

    df = df.append(dict, ignore_index=True)

    occurences_clips = collections.Counter(np.around(np.asarray(speed), 1))

    fig = PieChart(occurences_clips.keys(), occurences_clips.values())
    
    fig.savefig(FOLDER_FILES + '/images/' + str(r) + "_speed_distribution.svg")

ax = df.plot(x='Radius', kind='bar', stacked=True,
        title='Comparison between radius')

ax.set_ylabel("Number of frames")

fig = ax.get_figure()

fig.savefig(FOLDER_FILES + '/images/' + str(r) + "_blend_vs_transitions.svg")

if (PLOT_ENABLED):
    plt.show()
    
