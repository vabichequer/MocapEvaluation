from operator import contains
from matplotlib.text import Annotation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import csv
from pathlib import Path
from scipy.spatial import distance

def read_csv(read_time, angle, prefix="", sufix=""):
    x = []
    y = []
    t = []
    with open(FOLDER_FILES + '/' + prefix + str(angle) + sufix + ".csv", newline='') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in read_csv:
            if (read_time):
                if (len(row) == 4 and "" not in row):
                    t.append(float(row[0]))
                    x.append(float(row[1]))
                    y.append(float(row[3]))
                else:
                    print("Error in row", row, "File", prefix, angle, sufix)
            else:
                if (len(row) == 3 and "" not in row):
                    x.append(float(row[0]))
                    y.append(float(row[2]))
                else:
                    print("Error in row", row, "File", prefix, angle, sufix)
    
    csvfile.close()

    if (read_time):
        return np.asarray(x), np.asarray(y), np.asarray(t)
    else:
        return np.asarray(x), np.asarray(y)

FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/MotionMatching/Assets/output"))

angles = [5]#, 10, 15]

x_planned = []
y_planned = []
x_final = []
y_final = []
time = []

y_interp = []

all_x_offsets = []
all_y_offsets = []

tolerance = 0

for a in angles:
    x_p, y_p = read_csv(False, a, sufix="_planned")
    x_f, y_f, t = read_csv(True, a, sufix="_final")

    #distance()]

    # if (x_p[-1] < x_f[-1]):
    #     boolean_slicer = [x_f < x_p[-1]]
    # else:
    #     boolean_slicer = [x_p < x_f[-1]]     

    # x_f = x_f[boolean_slicer]
    # y_f = y_f[boolean_slicer]
    # t = t[boolean_slicer]

    interp = np.interp(x_f, x_p, y_p)

    y_interp.append(interp)

    x_planned.append(x_p)
    y_planned.append(y_p)
    x_final.append(x_f)
    y_final.append(y_f)
    time.append(t)

    t = np.asarray(t).astype(int)
    idx_of_changes = np.where(np.roll(t,1)!=t)[0]

    idx_of_changes = idx_of_changes[:len(x_p)]

    x_offset = x_p - x_f[idx_of_changes]
    y_offset = y_p - y_f[idx_of_changes]

    steps = np.linalg.norm(np.asarray([[x_offset], [y_offset]]), axis=1)

    #steps = steps[0, :] + steps[1, :]

    all_x_offsets.append(np.square(steps[0, :] - tolerance))
    all_y_offsets.append(np.square(steps[1, :] - tolerance))

fig, axs = plt.subplots(1, 3)

axs = axs.ravel()

for i in range(0, len(x_planned)):
    axs[i].set_title(str(angles[i]) + " degrees curve")
    #axs[i].plot(x_planned[i], y_planned[i], label="planned")
    axs[i].plot(x_final[i], y_final[i], label="final")    
    #axs[i].plot(x_final[i], y_final[i] - y_interp[i], label="area difference", color="black")
    axs[i].fill_between(x_final[i], y_interp[i], y_final[i], facecolor="none", hatch = '//', edgecolor="black")
    axs[i].set_xlabel("X")
    axs[i].set_ylabel("Y")
    #axs[i].set_ylim(-1, max(y_final[i]) * 1.1)
    axs[i].legend()

fig, axs = plt.subplots(1, 3)

axs = axs.ravel()

for i in range(0, len(angles)):
    axs[i].set_title(str(angles[i]) + " degrees curve offset")
    #axs[i].plot(x_planned[i], y_planned[i], label="planned")
    axs[i].plot(range(0, len(all_x_offsets[i])), all_x_offsets[i], label="X offset", marker="o")
    axs[i].plot(range(0, len(all_y_offsets[i])), all_y_offsets[i], label="Y offset", marker="o")
    #axs[i].plot(x_final[i], y_final[i] - y_interp[i], label="area difference", color="black")
    axs[i].grid()
    #axs[i].set_ylim(-1, max(y_final[i]) * 1.1)
    # Residual sum of squares
    anchored_text1 = AnchoredText("RSS_y: " + str(round(np.sum(all_y_offsets[i]), 2)), loc='lower right')
    anchored_text2 = AnchoredText("RSS_x: " + str(round(np.sum(all_x_offsets[i]), 2)), loc='lower left')
    axs[i].add_artist(anchored_text1)
    axs[i].add_artist(anchored_text2)
    axs[i].set_ylabel("Squared error")
    axs[i].set_xlabel("Trajectory point")
    axs[i].legend()

plt.show()
