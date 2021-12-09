import numpy as np
import matplotlib.pyplot as plt
import math 
import csv
import sys
from pathlib import Path
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
import pandas as pd
import os
import statsmodels.api as sm
from sklearn.decomposition import PCA
from scipy.interpolate import UnivariateSpline

### ARGUMENTS ###
# <string> CSV file name
# <int> sampling rate

def pause():
    programPause = input("Press the <ENTER> key to continue...")

def PieChart(labels, sizes):
    plt.figure()
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    plt.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)

    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
def scatter(fig, ax, x, y, alpha, radius, color = "", cmap = ""):    
    if (color == ""):
        xy = np.vstack([x,y])
        grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
        positions = np.vstack([grid_x.ravel(), grid_y.ravel()])
        z = gaussian_kde(xy)(xy)
        f = np.reshape(z[positions].T, grid_x.shape)
        idx = z.argsort()
        x, y, z = np.asarray(x)[idx], np.asarray(y)[idx], np.asarray(z)[idx]
        cax = ax.scatter(x, y, alpha=alpha, c=z, s=30, cmap=cmap)
        cset = plt.contour(x,y,f)
        plt.clabel(cset, inline=1, fontsize=10)
        fig.colorbar(cax)
    else:
        ax.scatter(x, y, alpha=alpha, c=color, s=30)

    ax.set_ylabel("Linear speed (m/s)")
    ax.set_xlabel("Turning speed (degrees/s)")
    ax.set_title("Speed coverage map, radius: " + str(radius))
    

def magnitude(vx, vy): 
    return math.sqrt((vx * vx) + (vy * vy))

animation_dataset_file = sys.argv[1]
radiuses = [int(x) for x in sys.argv[2].split(',')]
time_windows = [float(x) for x in sys.argv[3].split(',')]
PLOT_ENABLED = eval(sys.argv[4])

if (len(sys.argv) > 5):
    FOLDER_FILES = str(Path(sys.argv[5]))
else:
    FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/MotionMatching/Assets/output/Debug/speed"))

csv_file_name = []
csv_file_name.append(animation_dataset_file)
for r in radiuses:
    csv_file_name.append(str(r) + "_final.csv")

print('*' * 25)
print("Program setup:")
print("CSV files names: ", csv_file_name)
print('*' * 25)

radiuses.insert(0, 0) # for the dataset

speed_arrays = []
theta_arrays = []
color_array = []
dataset_speeds = []
dataset_thetas = []
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
                y.append(float(row[3]))
                ry.append(float(row[5]))
                t.append(float(row[0]))    
    csvfile.close()
        
    if (sufix == "info"):   
        return info         
    if (sufix == "frames"):
        return frames
    if (sufix == "final" or sufix == "dataset"):
        return x, y, ry, t

time_acc = 0
dtheta_acc = 0
dx_acc = 0
dy_acc = 0

for idx, r in enumerate(radiuses):  
    if (r == 0):
        x, y, ry, t = read_csv(prefix = "../animation", sufix = "dataset")
    else:
        x, y, ry, t = read_csv(r, sufix = "final")

    x = np.asarray(x)
    y = np.asarray(y)
    ry = np.asarray(ry)

    print("r: ", r)
    print("t size: ", len(t))
    print("X size: ", len(x))
    print("Y size: ", len(y))
    print("RY size: ", len(ry), '\n')

    speed = []
    theta = []
    single_speed = []
    single_theta = []
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

        for j in range(1, len(dt_array)):
            dx.append(x_array[j] - x_array[j - 1])
            dy.append(y_array[j] - y_array[j - 1])

            orientation.append(ry_array[j])

        #plt.plot([k for k in range(0, len(dx))], dx)
        #plt.show()

        frames_left = True

        for j in range(1, len(dt_array)):
            dt = dt_array[j - 1]

            dtheta = orientation[j] - orientation[j - 1]       

            if (dtheta >= 180):
                dtheta -= 360
            elif (dtheta <= -180):
                dtheta += 360            

            if (time_windows[idx] > 0):
                if (time_acc > time_windows[idx]):
                    speed.append(magnitude(dx_acc / time_acc, dy_acc / time_acc))
                    theta.append(dtheta_acc / time_acc)
                    single_speed.append(magnitude(dx_acc / time_acc, dy_acc / time_acc))
                    single_theta.append(dtheta_acc / time_acc)
                    time_acc, dtheta_acc, dx_acc, dy_acc = 0,0,0,0
                    frames_left = False
                else:
                    frames_left = True
                    dtheta_acc += dtheta
                    dx_acc += dx[j - 1]
                    dy_acc += dy[j - 1]
                    time_acc += dt
            else:
                frames_left = False
                speed.append(magnitude(dx[j - 1] / dt, dy[j - 1] / dt))
                theta.append(dtheta / dt)
                single_speed.append(magnitude(dx[j - 1] / dt, dy[j - 1] / dt))
                single_theta.append(dtheta / dt)
        
        if (frames_left):
            speed.append(magnitude(dx_acc / time_acc, dy_acc / time_acc))
            theta.append(dtheta_acc / time_acc)
            single_speed.append(magnitude(dx_acc / time_acc, dy_acc / time_acc))
            single_theta.append(dtheta_acc / time_acc)

            # if ((dtheta / dt) > 4000):
            #     print("dtheta: ", dtheta)
            #     print("dt: ", dt)
            #     print("dtheta / dt: ", dtheta / dt)
            #     print("i, j: ", i, ",", j)
            #     pause()
        color_array.append(np.random.choice(range(256), size=3))
        dataset_speeds.append(single_speed)
        dataset_thetas.append(single_theta)
        single_speed = []
        single_theta = []

    speed_arrays.append(np.asarray(speed))
    theta_arrays.append(np.asarray(theta))

    print(len(speed))
    print(len(theta))

    plt.figure()
    plt.title("Linear speed")    
    plt.plot([x for x in range(0, len(speed))], speed)
    plt.plot([x for x in range(0, len(speed))], [np.mean(speed) for x in range(0, len(speed))])
    plt.scatter([x for x in range(0, len(speed))], speed)

    plt.figure()
    plt.title("Angular speed")
    plt.plot([x for x in range(0, len(theta))], theta, label='original')
    plt.plot([x for x in range(0, len(theta))], [np.mean(theta) for x in range(0, len(theta))], label='mean')
    plt.scatter([x for x in range(0, len(theta))], theta)
    n_comp = 2
    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(np.column_stack(([x for x in range(0, len(theta))], theta)))
    indxs = ["PC" + str(i + 1) for i in range(0, n_comp)]
    pcs = pd.DataFrame(data=pcs, columns=indxs)
    for i in range(0, n_comp):
        plt.plot([x for x in range(0, len(theta))], pcs["PC" + str(i + 1)], label=indxs[i])
    plt.plot([x for x in range(0, len(theta))], theta -  pcs["PC2"], label='original - PC2')
    plt.legend()

    #plt.figure()
    #plt.scatter(theta, speed)
    plt.show()

file_nbr = len(csv_file_name)

radiuses.remove(0)

# different motions
plt.figure(figsize=(16, 9))
for colors, speeds, thetas in zip(color_array, dataset_speeds, dataset_thetas):
    data = {'theta': thetas, 'speed': speeds}
    data = pd.DataFrame(data=data)
    sns.scatterplot(data=data, x='theta', y='speed', palette=colors)
    sns.lineplot(data=data, x='theta', y='speed', palette=colors)

for i, r in enumerate(radiuses):
    info = read_csv(r, sufix = "info")
    frames = read_csv(r, sufix = "frames")

    dataset_speed = speed_arrays[0]
    dataset_theta = theta_arrays[0]
    trial_speed = speed_arrays[i + 1]
    trial_theta = theta_arrays[i + 1]

    plt.figure(figsize=(16, 9)) 
    data = {'theta': dataset_theta, 'speed': dataset_speed}
    data = pd.DataFrame(data=data)
    xy = np.vstack([dataset_theta, dataset_speed])
    z = gaussian_kde(xy)(xy) 
    coverage = sns.scatterplot(data=data, x='theta', y='speed', c=z, cmap='mako')

    data = {'theta': trial_theta, 'speed': trial_speed}
    data = pd.DataFrame(data=data)
    xy = np.vstack([trial_theta, trial_speed])
    z = gaussian_kde(xy)(xy)
    coverage = sns.scatterplot(data=data, x='theta', y='speed', c=z, cmap='autumn')
    desiredPoint = [info["Angle speed"], info["Desired linear speed"]]
    coverage.scatter(desiredPoint[0], desiredPoint[1], color='lime')
    
    plt.figure()
    distance = cdist([desiredPoint], data, metric='euclidean')[0]
    df = pd.DataFrame({'x': [j for j in range(0, len(distance))], 'Distance': distance})
    ls_dist = sns.histplot(data=df, x='Distance', kde=True)
    ls_dist.set(xlabel='Distance', title="Distance from reference point")

    # This is in order not to average the speed when transitioning
    # this needs to be raw data, otherwise I can't pinpoint where
    # the transition actually occured, because the data is averaged
    #speed = speed_arrays[i + 1]
    #theta = theta_arrays[i + 1]

    speed = np.asarray(trial_speed)
    theta = np.asarray(trial_theta)

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

    coverage.figure.savefig(FOLDER_FILES + '/images/' + str(r) + "_coverage.svg")
    ls_dist.figure.savefig(FOLDER_FILES + '/images/' + str(r) + "_ls_dist.svg")
    fig_ts.savefig(FOLDER_FILES + '/images/' + str(r) + "_turning_speed.svg")
    fig_ls.savefig(FOLDER_FILES + '/images/' + str(r) + "_linear_speed.svg")

    trial_speed = np.asarray(trial_speed)
    
    file = open(FOLDER_FILES + '/' + str(r) + "_speed_dump.csv", 'w', newline='')

    writer = csv.writer(file)

    for speed in trial_speed:
        writer.writerow([speed])
    
    file.close()

    if (PLOT_ENABLED):
        plt.show()