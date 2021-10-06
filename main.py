import numpy as np
import matplotlib.pyplot as plt
import math 
import csv
import sys
from pathlib import Path

### ARGUMENTS ###
# <string> CSV file name
# <int> sampling rate

def scatter(fig, x, y, alpha, color, radius):    
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, alpha=alpha, c=color)

    ax.set_ylabel("Linear speed (m/s)")
    ax.set_xlabel("Turning speed (degrees/s)")
    ax.set_title("Speed coverage map, radius: " + str(radius))

def magnitude(vector): 
    return math.sqrt(sum(pow(element, 2) for element in vector))

FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/MotionMatching/Assets/output"))

animation_dataset_file = sys.argv[1]
radius = sys.argv[2].split(',')
sampling = [int(x) for x in sys.argv[3].split(',')]

csv_file_name = []
csv_file_name.append(animation_dataset_file)
for r in radius:
    csv_file_name.append(r + "_final.csv")

print('*' * 25)
print("Program setup:")
print("CSV files names: ", csv_file_name)
print('*' * 25)

speed_arrays = []
theta_arrays = []

def read_csv(radius, prefix="", sufixes=""):
    info = {}
    for sufix in sufixes:
        with open(FOLDER_FILES + '/' + prefix + str(radius) + "_" + sufix + ".csv", newline='') as csvfile:
            read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
            if (sufix == "info"):            
                for row in read_csv:
                    info[row[0]] = float(row[1])
        csvfile.close()
    return info

for file_idx, file in enumerate(csv_file_name):
    x = []
    y = []
    ry = []
    t = []

    with open(FOLDER_FILES + '/' + file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')    
        actual_sample = 0
        for i, row in enumerate(csv_reader):   
            if (sampling[file_idx] > actual_sample):
                actual_sample += 1
            else:
                x.append(float(row[1]))
                y.append(float(row[2]))
                ry.append(float(row[5]))
                t.append(float(row[0]))                    
                t_var = 0
                actual_sample = 0

    x = np.asarray(x)
    y = np.asarray(y)
    ry = np.asarray(ry)

    print("t size: ", len(t))
    print("X size: ", len(x))
    print("Y size: ", len(y))
    print("RY size: ", len(ry))

    speed = []
    theta = []
    all_dt_arrays = []
    all_x_arrays = []
    all_y_arrays = []
    all_ry_arrays = []

    dt_array = []
    x_array = []
    y_array = []
    ry_array = []

    for i in range(1, len(t)):
        if t[i - 1] < t[i]:
            dt_array.append(t[i] - t[i - 1])
            x_array.append(x[i])
            y_array.append(y[i])
            ry_array.append(ry[i])
        else:
            all_dt_arrays.append(dt_array)
            all_x_arrays.append(x_array)
            all_y_arrays.append(y_array)
            all_ry_arrays.append(ry_array)
            dt_array = []
            x_array = []
            y_array = []
            ry_array = []
    all_dt_arrays.append(dt_array)
    all_x_arrays.append(x_array)
    all_y_arrays.append(y_array)
    all_ry_arrays.append(ry_array)

    for i in range(0, len(all_dt_arrays)):    
        dt_array = all_dt_arrays[i]
        x_array = all_x_arrays[i]
        y_array = all_y_arrays[i]
        ry_array = all_ry_arrays[i]
        dx = []
        dy = []
        orientation = []
        
        orientation.append(ry_array[0])

        for i in range(1, len(dt_array)):
            dx.append(x_array[i] - x_array[i - 1])
            dy.append(y_array[i] - y_array[i - 1])

            orientation.append(ry_array[i - 1])

        for i in range(1, len(dt_array)):
            dt = dt_array[i - 1]

            dtheta = orientation[i] - orientation[i - 1]

            if (dtheta >= 180):
                dtheta -= 360
            elif (dtheta <= -180):
                dtheta += 360

            speed.append(magnitude([dx[i - 1], dy[i - 1]]) / dt)
            theta.append(dtheta / dt)
            #if (abs(theta[i - 1]) > 180):
                #print(i, dtheta, dt, theta[i - 1], orientation[i], orientation[i - 1])

    speed = np.asarray(speed)
    theta = np.asarray(theta)

    speed_arrays.append(speed)
    theta_arrays.append(theta)

file_nbr = len(csv_file_name)

for i, r in enumerate(radius):
    fig_scm = plt.figure()

    speed = speed_arrays[0]
    theta = theta_arrays[0]
    scatter(fig_scm, theta, speed, 0.5, "red", r)

    speed = speed_arrays[i + 1]
    theta = theta_arrays[i + 1]
    scatter(fig_scm, theta, speed, 0.25, "blue", r)

    fig_ts = plt.figure()
    plt.ylabel("Turning speed (degrees/s)")
    plt.title("Turning speed (degrees/s) vs time (s)")
    plt.plot([j for j in range(0, len(theta))], theta)
    plt.plot([l for l in range(0, len(theta))], [theta.mean() for m in range(0, len(speed))])


    fig_ls = plt.figure()
    plt.ylabel("Linear speed (m/s)")
    plt.title("Linear speed (m/s) vs time (s)")
    plt.plot([k for k in range(0, len(speed))], speed)
    plt.plot([l for l in range(0, len(speed))], [speed.mean() for m in range(0, len(speed))])

    info = read_csv(r, sufixes=["info"])
    scatter(fig_scm, info["Angle speed"], info["Desired linear speed"], 1, "yellow", r)

    fig_scm.set_size_inches((16, 9), forward=False)
    fig_scm.savefig(str(r) + "_scm.eps", )
    fig_ts.savefig(str(r) + "_ts.eps")
    fig_ls.savefig(str(r) + "_ls.eps")

    plt.show()