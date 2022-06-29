import os
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import collections
import numpy as np
import sys
import pandas as pd

def floatToString(inputValue):
    return ('%.15f' % inputValue).rstrip('0').rstrip('.')

def read_csv(radius, prefix="", sufixes=""):
    frame_collection = []
    frame = []
    info = {}
    lastChosenCost = []
    channel = {}
    channel_number = 0
    for sufix in sufixes:
        with open(FOLDER_FILES + '/' + prefix + str(radius) + "_" + sufix + ".csv", newline='') as csvfile:
            read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
            if (sufix == "stats"):
                for row in read_csv:
                    if (row[1] == "lastChosenCost"):
                        lastChosenCost.append(float(row[2]))
                    elif (len(row) == 2):
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
    return frame_collection, lastChosenCost, info

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

def pause():
    programPause = input("Press the <ENTER> key to continue...")

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

extension = "png"

if (os.path.isdir(FOLDER_FILES + '/images/')):
    pass
else:
    os.mkdir(FOLDER_FILES + '/images/')

for r in radiuses:
    blend_statuses = []

    frames, lastChosenCost, info = read_csv(r, sufixes=["stats", "info"])
    full_clips = []
    primary_clips = []

    all_channels = []

    for frame in frames:
        current_channels = []
        frame_in_transition = []
        for channel in frame:
            blend_statuses.append(channel['BlendStatus'])
            current_channels.append(channel)

            if (channel['BlendStatus'] == "Dominant"):
                full_clips.append(channel)
                primary_clips.append(channel['Primary clip'])

        all_channels.append(current_channels)

    costOverTime = plt.figure(figsize=(16,9))
    plt.plot(range(len(lastChosenCost)), lastChosenCost)
    plt.scatter(range(len(lastChosenCost)), lastChosenCost)
    plt.title("Transition cost over time")
    plt.xlabel("Frames")
    plt.ylabel("Transition cost")
    costOverTime.savefig((FOLDER_FILES + "/images/"  + str(r) + "_cost_over_time." + extension))
    
    np.savez_compressed(FOLDER_FILES + '/' + str(r) + "_dominant_clip_every_frame", full_clips=full_clips)
    np.savez_compressed(FOLDER_FILES + '/' + str(r) + "_all_channels", all_channels=all_channels)


    occurences_clips = collections.Counter(primary_clips)
    occurences_statuses = collections.Counter(blend_statuses)
    #print(occurences_clips)
    #print(occurences_statuses)

    
    total_number_of_clips = len(pd.read_csv(FOLDER_FILES + "/../../animation_clips.csv"))

    labels = ['Used', 'Not used']
    sizes = [len(occurences_clips), total_number_of_clips]
    fig = PieChart(labels, sizes)
    fig.savefig(FOLDER_FILES + '/images/' + str(r) + "_used_vs_unused_animations.png")
    fig = PieChart(occurences_statuses.keys(), occurences_statuses.values())
    fig.savefig(FOLDER_FILES + '/images/' + str(r) + "_animation_statuses.png")

    primary_clips = np.asarray(primary_clips)
    transition_locations = np.where(np.roll(primary_clips,1)!=primary_clips)[0]
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