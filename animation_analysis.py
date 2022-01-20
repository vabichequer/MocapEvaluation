import os
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import collections
import numpy as np
import sys

def floatToString(inputValue):
    return ('%.15f' % inputValue).rstrip('0').rstrip('.')

def read_csv(radius, prefix="", sufixes=""):
    frame_collection = []
    frame = []
    info = {}
    channel = {}
    channel_number = 0
    for sufix in sufixes:
        with open(FOLDER_FILES + '/' + prefix + str(radius) + "_" + sufix + ".csv", newline='') as csvfile:
            read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
            if (sufix == "stats"):
                for row in read_csv:
                    if (len(row) == 2):
                        channel_number = int(row[1])                
                        if (channel_number > 0):
                            frame.append(channel)
                        channel = {}
                    elif (len(row) == 3):
                        channel[str(row[1])] = str(row[2])
                    elif (len(row) == 4):
                        #frame_nbr = int(row[1])
                        #idx = int(row[3])
                        frame.append(channel)
                        frame_collection.append(frame)
                        frame = []
                    else:
                        pass
            elif (sufix == "info"):            
                for row in read_csv:
                    info[row[0]] = float(row[1])
        csvfile.close()
    return frame_collection, info

def PieChart(labels, sizes):
    fig = plt.figure()
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    plt.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, normalize=True)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    return fig


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

def write_csv(name, info):
    f = open(FOLDER_FILES + '/' + str(r) + "_" + name + ".csv", 'w', newline='')

    w = csv.writer(f)

    for i in info:
        w.writerow([i])
    
    f.close()

print(25*'*')
print(sys.argv[0], "start")
print(25*'*')

PLOT_ENABLED = eval(sys.argv[1])
print("Plot enabled? ", PLOT_ENABLED)

if (len(sys.argv) > 2):
    FOLDER_FILES = str(Path(sys.argv[2]))
    radiuses = [floatToString(float(x)) for x in sys.argv[3].split(',')]
else:
    FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/MotionMatching/Assets/output/Mixamo/1.5/"))
    radiuses = [5]

if (os.path.isdir(FOLDER_FILES + '/images/')):
    pass
else:
    os.mkdir(FOLDER_FILES + '/images/')

for r in radiuses:
    blend_statuses = []

    frames, info = read_csv(r, sufixes=["stats", "info"])
    clips = []

    for frame in frames:
        for channel in frame:
            blend_statuses.append(channel['BlendStatus'])

            if (channel['BlendStatus'] == "Dominant"):
                clips.append(channel['Primary clip'])                
                
    occurences_clips = collections.Counter(clips)
    occurences_statuses = collections.Counter(blend_statuses)
    #print(occurences_clips)
    #print(occurences_statuses)

    labels = ['Used', 'Not used']
    sizes = [len(occurences_clips), 37]
    fig = PieChart(labels, sizes)
    fig.savefig(FOLDER_FILES + '/images/' + str(r) + "_used_vs_unused_animations.png")
    fig = PieChart(occurences_statuses.keys(), occurences_statuses.values())
    fig.savefig(FOLDER_FILES + '/images/' + str(r) + "_animation_statuses.png")

    clips = np.asarray(clips)
    transition_locations = np.where(np.roll(clips,1)!=clips)[0]
    number_of_transitions = len(transition_locations)    

    write_csv("frames", transition_locations)

    occurences_clips['Transitions'] = number_of_transitions

    bar_fig = plt.figure()
    barlist = plt.bar(occurences_clips.keys(), occurences_clips.values(), color='b')
    autolabel(barlist)
    barlist[-1].set_color('r')
    bar_fig.savefig(FOLDER_FILES + '/images/' + str(r) + "_occurences.png")

    

    # According to the MxMAnimator and the documentation, the Blend Time is set to 0.3, so it takes 0.3s to fully blenc in
    # an animation. Therefore, every time it changes, you can consider that 0.3s were taken to actually do it. Hence, the
    # multiplication by 0.3s.

    labels = ["Pure animation", "Blending animations"]
    fig = PieChart(labels, [info["Total time"], occurences_clips["Transitions"] * 0.3])
    fig.savefig(FOLDER_FILES + '/images/' + str(r) + "_blend_vs_pure_animations.png")

    if (occurences_clips["Transitions"] != 0):
        nbr_blended_frames = len(frames) * (info["Total time"] / occurences_clips["Transitions"] * 0.3)
    else:
        nbr_blended_frames = 0

    write_csv("nbr_blended_frames", [nbr_blended_frames])
    write_csv("nbr_transitions", [occurences_clips['Transitions']])

    
    if (PLOT_ENABLED):
        plt.show()


print(25*'*')
print(sys.argv[0], "end")
print(25*'*')