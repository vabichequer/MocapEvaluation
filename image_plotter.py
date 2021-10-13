from operator import contains
from matplotlib.text import Annotation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import csv
from pathlib import Path
from scipy.spatial import distance
import math

def magnitude(vector): 
    return math.sqrt(sum(pow(element, 2) for element in vector))

def read_csv(read_time, radius, prefix="", sufix=""):
    x = []
    y = []
    t = []
    with open(FOLDER_FILES + '/' + prefix + str(radius) + sufix + ".csv", newline='') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in read_csv:
            if (len(row) == 3 and "" not in row):
                x.append(float(row[0]))
                y.append(float(row[2]))
            else:
                print("Error in row", row, "File", prefix, radius, sufix)
    
    csvfile.close()

    return np.asarray(x), np.asarray(y)

FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/MotionMatching/Assets/output"))

radiuses = [5]#, 10, 15]

x_planned = []
y_planned = []

for r in radiuses:
    distances = []
    x_p, y_p = read_csv(False, r, sufix="_planned")

    x_planned.append(x_p)
    y_planned.append(y_p)

if len(radiuses) > 1:
    fig, axs = plt.subplots(1, 3)

    axs = axs.ravel()

    for i in range(0, len(x_planned)):
        axs[i].set_title(str(radiuses[i]) + "m radius curve") 
        axs[i].scatter(x_planned[i], y_planned[i], label="planned")    
        axs[i].grid()
        axs[i].set_xlabel("X")
        axs[i].set_ylabel("Y")
        axs[i].legend()
else:
    fig = plt.figure()

    for i in range(0, len(x_planned)):
        plt.title(str(radiuses[i]) + "m radius curve") 
        plt.scatter(x_planned[i], y_planned[i], label="planned")    
        plt.grid()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
plt.show()
