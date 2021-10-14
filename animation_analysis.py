from os import write
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import collections
import numpy as np

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
    plt.figure()
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    plt.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/MotionMatching/Assets/output"))

radiuses = [5, 10, 15]

occurences = {}

blend_statuses = []

for r in radiuses:
    frames, info = read_csv(r, sufixes=["stats", "info"])
    clips = []

    for frame in frames:
        for channel in frame:
            blend_statuses.append(channel['BlendStatus'])

            if (channel['BlendStatus'] == "Dominant"):
                clips.append(channel['Primary clip'])                
                break

    occurences_clips = collections.Counter(clips)
    occurences_statuses = collections.Counter(blend_statuses)
    print(occurences_clips)   
    print(occurences_statuses)    

    labels = ['Used', 'Not used']
    sizes = [len(occurences_clips), 37]
    PieChart(labels, sizes)
    PieChart(occurences_statuses.keys(), occurences_statuses.values())

    clips = np.asarray(clips)
    transition_locations = np.where(np.roll(clips,1)!=clips)[0]
    number_of_transitions = len(transition_locations)    

    file = open(FOLDER_FILES + '/' + str(r) + "_frames.csv", 'w', newline='')

    writer = csv.writer(file)

    for idx in transition_locations:
        writer.writerow([idx])
    
    file.close()

    occurences_clips['Transitions'] = number_of_transitions

    bar_fig = plt.figure()
    barlist = plt.bar(occurences_clips.keys(), occurences_clips.values(), color='b')
    autolabel(barlist)
    barlist[-1].set_color('r')

    
    labels = ["Pure animation", "Blending animations"]
    PieChart(labels, [info["Total time"], occurences_clips["Transitions"] * 0.3])
    plt.show()