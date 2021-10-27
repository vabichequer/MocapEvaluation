from os import read
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

def magnitude(vx, vy): 
    return math.sqrt((vx * vx) + (vy * vy))

FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/MotionMatching/Assets/output"))

animation_dataset_file = sys.argv[1]
radiuses = sys.argv[2].split(',')
time_windows = [float(x) for x in sys.argv[3].split(',')]

csv_file_name = []
csv_file_name.append(animation_dataset_file)
for r in radiuses:
    csv_file_name.append(r + "_final.csv")

print('*' * 25)
print("Program setup:")
print("CSV files names: ", csv_file_name)
print('*' * 25)

radiuses.insert(0, 0) # for the dataset

speed_arrays = []
theta_arrays = []
dt_arrays = []

def read_csv(radius = "", prefix="", sufix=""):
    info = {}
    frames = []    
    x = []
    y = []
    ry = []
    t = []

    with open(FOLDER_FILES + '/' + prefix + str(radius) + "_" + sufix + ".csv", newline='') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
        if (sufix == "info"):            
            for row in read_csv:
                info[row[0]] = float(row[1])
        if (sufix == "frames"):
            for row in read_csv:
                frames.append(int(row[0]))
        if (sufix == "final" or sufix == "dataset"):
            for row in read_csv:   
                x.append(float(row[1]))
                y.append(float(row[2]))
                ry.append(float(row[5]))
                t.append(float(row[0]))    
    csvfile.close()
        
    if (sufix == "info"):   
        return info         
    if (sufix == "frames"):
        return frames
    if (sufix == "final" or sufix == "dataset"):
        return x, y, ry, t

for r in radiuses:  
    if (r == 0):
        x, y, ry, t = read_csv(prefix = "animation", sufix = "dataset")
    else:
        x, y, ry, t = read_csv(r, sufix = "final")

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

            speed.append(magnitude(dx[i - 1] / dt, dy[i - 1] / dt))
            theta.append(dtheta / dt)

    speed = np.asarray(speed)
    theta = np.asarray(theta)

    speed_arrays.append(speed)
    theta_arrays.append(theta)
    dt_arrays.append(all_dt_arrays)

file_nbr = len(csv_file_name)

# Averaging results
all_average_speeds = []
all_average_thetas = []

print(file_nbr, len(radiuses))

for i in range(0, file_nbr):
    if (time_windows[i] > 0):
        average_speed = []
        average_theta = []

        for array_nbr, single_dt_array in enumerate(dt_arrays[i]):
            time_acc = []
            for j, dt in enumerate(single_dt_array):
                time_acc.append(dt)
                idx = len(time_acc) - 1
                if (sum(time_acc) > time_windows[i]):
                    avg_speed = np.mean(speed_arrays[i][j - idx:j])
                    avg_theta = np.mean(theta_arrays[i][j - idx:j])
                    average_speed.append(np.mean(speed_arrays[i][j - idx:j]))
                    average_theta.append(np.mean(theta_arrays[i][j - idx:j]))
                    time_acc.pop(0)
                else:
                    if (j == len(single_dt_array) - 1):
                        average_speed.append(np.mean(speed_arrays[i][j - idx:j]))
                        average_theta.append(np.mean(theta_arrays[i][j - idx:j]))


        all_average_speeds.append(average_speed)
        all_average_thetas.append(average_theta)
    else:        
        all_average_speeds.append(speed_arrays[i])
        all_average_thetas.append(theta_arrays[i])

radiuses.remove(0)

for i, r in enumerate(radiuses):
    info = read_csv(r, sufix = "info")
    frames = read_csv(r, sufix = "frames")

    fig_scm = plt.figure()

    speed = all_average_speeds[0]
    theta = all_average_thetas[0]
    scatter(fig_scm, theta, speed, 0.5, "red", r)

    speed = np.asarray(all_average_speeds[i + 1])
    theta = np.asarray(all_average_thetas[i + 1])
    scatter(fig_scm, theta, speed, 0.25, "blue", r)

    # This is in order not to average the speed when transitioning
    # this needs to be a raw data, otherwise I can't pinpoint where
    # the transition actually occured, because the data is averaged
    speed = speed_arrays[i + 1]
    theta = theta_arrays[i + 1]

    fig_ts = plt.figure()
    plt.ylabel("Turning speed (degrees/s)")
    plt.title("Turning speed (degrees/s) vs time (s)")
    plt.plot([j for j in range(0, len(theta))], theta)
    plt.plot([l for l in range(0, len(theta))], [theta.mean() for m in range(0, len(speed))])
    plt.scatter(frames, theta[frames], color='r')
    plt.legend(["Turning speed", "Average", "Transitions"])

    fig_ls = plt.figure()
    plt.ylabel("Linear speed (m/s)")
    plt.title("Linear speed (m/s) vs time (s)")
    plt.plot([k for k in range(0, len(speed))], speed)
    plt.plot([l for l in range(0, len(speed))], [speed.mean() for m in range(0, len(speed))])
    plt.scatter(frames, speed[frames], color='r')
    plt.legend(["Linear speed", "Average", "Transitions"])

    scatter(fig_scm, info["Angle speed"], info["Desired linear speed"], 1, "lime", r)

    fig_scm.set_size_inches((16, 9), forward=False)
    fig_scm.savefig(str(r) + "_scm.eps", )
    fig_ts.savefig(str(r) + "_ts.eps")
    fig_ls.savefig(str(r) + "_ls.eps")

    plt.show()