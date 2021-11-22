import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
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
            if (read_time):
                if (len(row) == 7 and "" not in row):
                    t.append(float(row[0]))
                    x.append(float(row[1]))
                    y.append(float(row[3]))
                else:
                    print("Error in row", row, "File", prefix, radius, sufix)
            else:
                if (len(row) == 3 and "" not in row):
                    x.append(float(row[0]))
                    y.append(float(row[2]))
                else:
                    print("Error in row", row, "File", prefix, radius, sufix)
    
    csvfile.close()

    if (read_time):
        return np.asarray(x), np.asarray(y), np.asarray(t)
    else:
        return np.asarray(x), np.asarray(y)

FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/MotionMatching/Assets/output"))

radiuses = [5, 10, 15]

x_planned = []
y_planned = []
x_final = []
y_final = []
time = []

y_interp = []

all_distances = []

tolerance = 0

for r in radiuses:
    distances = []
    x_p, y_p = read_csv(False, r, sufix="_planned")
    x_f, y_f, t = read_csv(True, r, sufix="_final")

    x_planned.append(x_p)
    y_planned.append(y_p)
    x_final.append(x_f)
    y_final.append(y_f)
    time.append(t)

    # Find every control point in the data extracted from Unity
    # This is useful to compare the error generated at every control point

    dx = x_p - x_f
    dy = y_p - y_f

    for i in range(0, len(dx)):
        distances.append(magnitude([dx[i], dy[i]]))
    
    all_distances.append(distances)

fig, axs = plt.subplots(1, 3)

axs = axs.ravel()

for i in range(0, len(x_planned)):
    axs[i].set_title(str(radiuses[i]) + "m radius curve")
    axs[i].plot(x_final[i], y_final[i], label="final")    
    axs[i].plot(x_planned[i], y_planned[i], label="planned")    
    axs[i].grid()
    axs[i].set_xlabel("X")
    axs[i].set_ylabel("Y")
    axs[i].legend()

fig, axs = plt.subplots(1, 3)

axs = axs.ravel()

for i in range(0, len(radiuses)):
    axs[i].set_title(str(radiuses[i]) + "m radius curve offset")
    axs[i].plot(range(0, len(all_distances[i])), all_distances[i], marker="o")
    axs[i].grid()
    # Residual sum of squares
    #anchored_text1 = AnchoredText("Summed Distance Error: " + str(round(np.sum(all_distances[i]), 2)) + "m", loc='lower right')
    #axs[i].add_artist(anchored_text1)
    axs[i].set_ylabel("Distance")
    axs[i].set_xlabel("Trajectory point")

plt.show()
