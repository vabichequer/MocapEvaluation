import matplotlib.pyplot as plt
import csv
from pathlib import Path
import collections

def read_csv(angle, prefix="", sufix=""):
    frame_collection = []
    frame = []
    channel = {}
    channel_number = 0
    with open(FOLDER_FILES + '/' + prefix + str(angle) + sufix + ".csv", newline='') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
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
    
    csvfile.close()
    return frame_collection

FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/MotionMatching/Assets/output"))

angles = [5]#, 10, 15]

occurences = {}

for a in angles:
    frames = read_csv(a, sufix="_stats")
    clips = []

    for frame in frames:
        for channel in frame:
            if (channel['BlendStatus'] == "Dominant"):
                clips.append(channel['Primary clip'])
                break
        
    occurences = collections.Counter(clips)
    print(occurences)

plt.bar(occurences.keys(), occurences.values(), color='b')
plt.show()
