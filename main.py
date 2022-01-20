import numpy as np
import matplotlib.pyplot as plt
import math 
import csv
import sys
from pathlib import Path
import seaborn as sns
from scipy.stats import gaussian_kde, norm
from scipy.spatial.distance import cdist
import pandas as pd
import os
import itertools
import random

### ARGUMENTS ###
# <string> CSV file name
# <int> sampling rate

marker = itertools.cycle((',', '+', '.', 'o', '*')) 
extension = "png"

def floatToString(inputValue):
    return ('%.15f' % inputValue).rstrip('0').rstrip('.')

def plotError(error_array, error_type):
    fig = plt.figure()
    mu, std = norm.fit(error_array)
    data = {'error': error_array}
    data = pd.DataFrame(data=data)
    # Plot the histogram
    sns.histplot(data, bins=25, stat="probability")

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    p = p/np.linalg.norm(p)
    plt.plot(x, p, 'k', linewidth=2)
    plt.xlabel(error_type + " speed error")
    plt.ylabel("Proportion")
    title = error_type + " speed error results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    return fig, mu, std

def pause():
    programPause = input("Press the <ENTER> key to continue...")
   
def magnitude(vx, vy): 
    return math.sqrt((vx * vx) + (vy * vy))

print(25*'*')
print(sys.argv[0], "start")
print(25*'*')

animation_dataset_file = sys.argv[1]
radiuses = [floatToString(float(x)) for x in sys.argv[2].split(',')]
time_windows = [float(x) for x in sys.argv[3].split(',')]
PLOT_ENABLED = eval(sys.argv[4])

if (len(sys.argv) > 5):
    FOLDER_FILES = str(Path(sys.argv[5]))
    prefix = "animation"
    parents = 1
else:
    FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/MotionMatching/Assets/output/Mixamo/1.5"))
    prefix = "animation"
    parents = 0

csv_file_name = []
csv_file_name.append(animation_dataset_file)
for r in radiuses:
    csv_file_name.append(str(r) + "_final.csv")

print('*' * 25)
print("Program setup:")
print("Animation dataset file: ", animation_dataset_file)
print("CSV files names: ", csv_file_name[1:])
print("Radiuses: ", radiuses)
print("Time windows: ", time_windows)
print("Plot?: ", PLOT_ENABLED)
print('*' * 25)

radiuses.insert(0, 0) # for the dataset

speed_arrays = []
theta_arrays = []
color_array = []
dataset_speeds = []
dataset_thetas = []
dt_arrays = []

def read_csv(radius = "", prefix="", sufix="", parents=0):
    info = {}
    frames = []    
    x, y, ry, t = [], [], [], []
    all_x, all_y, all_ry, all_dt = [], [], [], []

    if parents > 0:
        p = str(Path(FOLDER_FILES).parents[parents])
    else:
        p = FOLDER_FILES

    with open(p + '/' + prefix + str(radius) + "_" + sufix + ".csv", newline='') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
        if (sufix == "info"):            
            for row in read_csv:
                info[row[0]] = float(row[1])
        if (sufix == "frames"):
            for row in read_csv:
                frames.append(int(row[0]))
        if (sufix == "final" or sufix == "dataset"):
            time = 0
            last_size = 0
            for current_size, row in enumerate(read_csv):
                x.append(float(row[1]))
                y.append(float(row[3]))
                ry.append(float(row[5]))
                t.append(float(row[0]))
                if (float(row[0]) < time):
                    all_x.append(np.asarray(x[last_size:current_size]))
                    all_y.append(np.asarray(y[last_size:current_size]))
                    all_ry.append(np.asarray(ry[last_size:current_size]))
                    all_dt.append(np.diff(t[last_size:current_size]))
                    last_size = current_size
                time = float(row[0])

            all_x.append(np.asarray(x[last_size:current_size]))
            all_y.append(np.asarray(y[last_size:current_size]))
            all_ry.append(np.asarray(ry[last_size:current_size]))
            all_dt.append(np.diff(t[last_size:current_size]))

    csvfile.close()
        
    if (sufix == "info"):   
        return info         
    if (sufix == "frames"):
        return frames
    if (sufix == "final" or sufix == "dataset"):
        return np.asarray(all_x, dtype=object), np.asarray(all_y, dtype=object), np.asarray(all_ry, dtype=object), np.asarray(all_dt, dtype=object)

for idx, r in enumerate(radiuses):  
    if(os.path.isfile(FOLDER_FILES + "/../animation_dataset_dump.npz") and r == 0):
        print("Animation dataset dump loaded.")
        pass
    else:
        if (r == 0):
            all_x_arrays, all_y_arrays, all_ry_arrays, all_dt_arrays = read_csv(prefix = prefix, sufix = "dataset", parents=parents)
        else:
            all_x_arrays, all_y_arrays, all_ry_arrays, all_dt_arrays = read_csv(r, sufix = "final")

        # print("r: ", r)
        # print("t size: ", all_dt_arrays.shape)
        # print("X size: ", all_x_arrays.shape)
        # print("Y size: ", all_y_arrays.shape)
        # print("RY size: ", all_ry_arrays.shape, '\n')

        speed = []
        theta = []

        last_size = 0

        for i in range(0, len(all_dt_arrays)):    
            dx = np.diff(all_x_arrays[i])
            dy = np.diff(all_y_arrays[i])  

            if (r == 0):
                for j in range(0, i):
                    if (len(all_x_arrays[i]) == len(all_x_arrays[j])):
                        diff = sum(all_x_arrays[i] - all_x_arrays[j])
                        if (diff == 0):
                            print("Animations", i, "and", j, "are mirrored.")
                            all_x_arrays[i] = -all_x_arrays[i]
                            all_ry_arrays[i] = -all_ry_arrays[i]

            x = [j for j in range(0, len(all_ry_arrays[i]))]

            speed_acc = []
            theta_acc = []
            time_acc = []
            frames_last = False
            dtheta = np.diff(all_ry_arrays[i])

            dtheta[dtheta >= 180] -= 360
            dtheta[dtheta <= -180] += 360
            
            for j in range(1, len(all_dt_arrays[i])):
                dt = all_dt_arrays[i][j - 1]                  

                if (abs(dtheta[j] / dt) > 510):      
                    print(39*"*")
                    print("Potential problem detected. (Overspeed)")     
                    print("Speed captured:", dtheta[j] / dt)           
                    print("Radius:", r)
                    print("Animation:", i + 1)
                    print("Frame:", j)
                    print(39*"*")
                    continue
                    print("dtheta:", dtheta[j])
                    print("orientation[j]:", all_ry_arrays[i][j])
                    print("orientation[j - 1]:", all_ry_arrays[i][j - 1])
                    print("dt:", dt)
                    print("dtheta/dt:", dtheta[j]/dt)

                    fig, ax = plt.subplots()
                    plt.plot(x, all_ry_arrays[i], 'o', x, all_ry_arrays[i], label='original')
                    plt.scatter([j], [all_ry_arrays[i][j]], s = 100, c='r') 
                    plt.scatter([j - 1], [all_ry_arrays[i][j - 1]], s = 100, c='g') 
                    plt.legend()

                    for ann in range(0, len(all_x_arrays[i])):
                        ax.annotate(ann, (x[ann], all_ry_arrays[i][ann]))
                        
                    fig, ax = plt.subplots()
                    plt.plot(all_x_arrays[i], all_y_arrays[i], 'o', all_x_arrays[i], all_y_arrays[i])
                    
                    for ann in range(0, len(all_x_arrays[i])):
                        ax.annotate(ann, (all_x_arrays[i][ann], all_y_arrays[i][ann]))

                    plt.show()

                if (time_windows[idx] > 0):                
                    if (sum(time_acc) <= time_windows[idx]):
                        time_acc.append(dt)
                        speed_acc.append(magnitude(dx[j - 1] / dt, dy[j - 1] / dt))
                        theta_acc.append(dtheta[j] / dt)
                        frames_last = True
                    else:
                        speed.append(np.mean(speed_acc))
                        theta.append(np.mean(theta_acc))
                        frames_last = False
                        while (sum(time_acc) > time_windows[idx]):
                            time_acc.pop(0)
                            speed_acc.pop(0)
                            theta_acc.pop(0) 
                else:
                    speed.append(magnitude(dx[j - 1] / dt, dy[j - 1] / dt))
                    theta.append(dtheta[j] / dt)
                    
            if (frames_last):     
                speed.append(np.mean(speed_acc))
                theta.append(np.mean(theta_acc))

            if (r==0):
                current_size = len(speed)
                color_array.append((random.random(), random.random(), random.random()))
                dataset_speeds.append(speed[last_size:current_size])
                dataset_thetas.append(theta[last_size:current_size])
                last_size = current_size

        speed_arrays.append(np.asarray(speed))
        theta_arrays.append(np.asarray(theta))
        print("Radius", r, "processed.")

if(not os.path.isfile(FOLDER_FILES + "/../animation_dataset_dump.npz")):
    np.savez_compressed(FOLDER_FILES + "/../animation_dataset_dump", dataset_speed=speed_arrays[0], dataset_theta=theta_arrays[0])
    np.savez_compressed(FOLDER_FILES + "/../different_motion_dump", color_array=color_array, dataset_speeds=dataset_speeds, dataset_thetas=dataset_thetas)
else:
    loaded = np.load(FOLDER_FILES + "/../animation_dataset_dump.npz")
    speed_arrays.insert(0, loaded['dataset_speed'])
    theta_arrays.insert(0, loaded['dataset_theta'])
    if (not os.path.isfile(FOLDER_FILES + "/../animation_dataset_dump.npz")):
        print("different_motion_dump.npz is missing. Terminating program...")
        exit()
    loaded = np.load(FOLDER_FILES + "/../different_motion_dump.npz", allow_pickle=True)
    color_array = loaded['color_array']
    dataset_speeds = loaded['dataset_speeds']
    dataset_thetas = loaded['dataset_thetas']

        
file_nbr = len(csv_file_name)

radiuses.remove(0)

# different motions
different_motions = plt.figure(figsize=(16, 9))
for colors, speeds, thetas in zip(color_array, dataset_speeds, dataset_thetas):
    #data = {'theta': thetas, 'speed': speeds}
    #data = pd.DataFrame(data=data)
    #sns.scatterplot(data=data, x='theta', y='speed', palette=colors)
    #sns.lineplot(data=data, x='theta', y='speed', palette=colors)
    plt.scatter(thetas, speeds, color=colors, marker=next(marker))
    plt.plot(thetas, speeds, color=colors)
    
for i, r in enumerate(radiuses):
    info = read_csv(r, sufix = "info")
    frames = read_csv(r, sufix = "frames")

    dataset_speed = speed_arrays[0]
    dataset_theta = theta_arrays[0]
    trial_speed = speed_arrays[i + 1]
    trial_theta = theta_arrays[i + 1]

    # Calculate speed error
    # Angular
    angular_error = trial_theta - info["Angle speed"]
    aerror_fig, mu_angular, std_angular = plotError(angular_error, "Angular")

    # Linear
    linear_error = trial_speed - info["Desired linear speed"]
    lserror_fig, mu_linear, std_linear = plotError(linear_error, "Linear")

    # Plot the dataset
    plt.figure(figsize=(16, 9)) 
    plt.rc('axes', axisbelow=True)
    plt.grid(color='grey', linestyle='--', linewidth=1)
    data_dict = {'theta': dataset_theta, 'speed': dataset_speed}
    data = pd.DataFrame(data=data_dict)
    xy = np.vstack([dataset_theta, dataset_speed])
    z = gaussian_kde(xy)(xy) 
    coverage = sns.scatterplot(data=data, x='theta', y='speed', c=z, cmap='mako')
    path_coverage = str(Path(FOLDER_FILES).parents[1])
    np.savez_compressed(path_coverage + "/coverage_plot", data=data_dict, z=z)
    
    # Plot the traejctory points
    # data = {'theta': trial_theta, 'speed': trial_speed}
    # data = pd.DataFrame(data=data)
    # xy = np.vstack([trial_theta, trial_speed])
    # z = gaussian_kde(xy)(xy)
    # coverage = sns.scatterplot(data=data, x='theta', y='speed', c=z, cmap='autumn')

    # Plot the desired point
    desiredPoint = [info["Angle speed"], info["Desired linear speed"]]
    plt.errorbar(desiredPoint[0], desiredPoint[1], xerr=std_angular, yerr=std_linear, color='red', capsize=5.0)
    coverage.scatter(desiredPoint[0], desiredPoint[1], color='lime', zorder=2)

    np.savez_compressed(FOLDER_FILES + "/" + str(r) + "_desiredPoint_and_error", dp=desiredPoint, error=[std_linear, std_angular])

    # This is in order not to average the speed when transitioning
    # this needs to be raw data, otherwise I can't pinpoint where
    # the transition actually occured, because the data is averaged
    #speed = speed_arrays[i + 1]
    #theta = theta_arrays[i + 1]

    speed = np.asarray(trial_speed, dtype=object)
    theta = np.asarray(trial_theta, dtype=object)

    fig_ts = plt.figure()
    plt.ylabel("Turning speed (degrees/s)")
    plt.title("Average turning speed (degrees/s) vs time (s) for " + str(time_windows[i]) + "s")
    plt.plot([j for j in range(0, len(theta))], theta)
    plt.plot([l for l in range(0, len(theta))], [theta.mean() for m in range(0, len(speed))])
    #plt.scatter(frames, theta[frames], color='r')
    plt.legend(["Turning speed", "Average", "Transitions"])

    fig_ls = plt.figure()
    plt.ylabel("Linear speed (m/s)")
    plt.title("Average linear speed (m/s) vs time (s) for " + str(time_windows[i]) + "s")
    plt.plot([k for k in range(0, len(speed))], speed)
    plt.plot([l for l in range(0, len(speed))], [speed.mean() for m in range(0, len(speed))])
    #plt.scatter(frames, speed[frames], color='r'
    plt.legend(["Linear speed", "Average", "Transitions"])

    if (os.path.isdir(FOLDER_FILES + '/images/')):
        pass
    else:
        os.mkdir(FOLDER_FILES + '/images/')

    coverage.figure.savefig(FOLDER_FILES + '/images/' + str(r) + "_coverage." + extension)
    fig_ts.savefig(FOLDER_FILES + '/images/' + str(r) + "_turning_speed." + extension)
    fig_ls.savefig(FOLDER_FILES + '/images/' + str(r) + "_linear_speed." + extension)
    aerror_fig.savefig(FOLDER_FILES + '/images/' + str(r) + "_angular_speed_error." + extension)
    lserror_fig.savefig(FOLDER_FILES + '/images/' + str(r) + "_linear_speed_error." + extension)
    different_motions.savefig(FOLDER_FILES + '/images/' + str(r) + "_different_motions." + extension)
    
    trial_speed = np.asarray(trial_speed)
    
    file = open(FOLDER_FILES + '/' + str(r) + "_speed_dump.csv", 'w', newline='')

    writer = csv.writer(file)

    for speed in trial_speed:
        writer.writerow([speed])
    
    file.close()
    plt.close('all')

    if (PLOT_ENABLED):
        plt.show()

print(25*'*')
print(sys.argv[0], "end")
print(25*'*')