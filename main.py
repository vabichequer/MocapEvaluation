from matplotlib.text import Annotation
import numpy as np
import matplotlib.pyplot as plt
import math 
import csv
import sys
from distutils.util import strtobool
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap

### ARGUMENTS ###
# <bool> graphical time annotation
# <float> frequency (Hz) (Unity or float in Hz)
# <string> CSV file name
# <bool> has headers
# <int> source (0: UMANS, 1: Mocap (Xsens)) 
# <int> sampling rate

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

def using_mpl_scatter_density(fig, x, y):
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis)

    if (ANNOTATE):
        txt = 0.1
        for i in range(0, len(speed)):
            if (i % 10 == 0):
                ax.annotate("%.2f" % txt, (theta[i], speed[i]))
            txt += 0.1

    fig.colorbar(density, label='Number of points per pixel')
    ax.set_ylabel("Linear speed (m/s)")
    ax.set_xlabel("Turning speed (degrees/s)")
    ax.set_title("Speed coverage map")

def magnitude(vector): 
    return math.sqrt(sum(pow(element, 2) for element in vector))

if (len(sys.argv) < 7):
    print("ERROR: Not enough arguments. The program expects 6 arguments: <bool> graphical time annotation, <float> frequency (Hz), <string> CSV file name, <bool> headers?, <int> source (0: UMANS, 1: Mocap (Xsens)), <int> sampling rate")

ANNOTATE = strtobool(sys.argv[1])
CALCULATE_DT = False

csv_file_name = str(sys.argv[3])
skip_header = strtobool(sys.argv[4])
source = int(sys.argv[5])
sampling = int(sys.argv[6])

if (sys.argv[2] == "Unity"):
    CALCULATE_DT = True
    t = []
    f = "Unity"
    dt = "Not defined yet"
else:
    f = float(sys.argv[2])
    dt = 1 / (f / sampling)



if (source == 0):
    y_pos = 2
else:
    y_pos = 3

print('*' * 25)
print("Program setup:")
print("Time annotation: ", ANNOTATE)
print("Frequency (Hz): ", f)
print("CSV file name: ", csv_file_name)
print("Headers: ", skip_header)
print("Source: ", source)
print("dT: ", dt)
print('*' * 25)

x = []
y = []
ry = []

with open(csv_file_name, newline='') as csvfile:
    read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')    
    if (skip_header):
        next(read_csv, None) # skipping the first line
    t_var = 0     
    actual_sample = 0
    for i, row in enumerate(read_csv):   
        if (sampling > actual_sample):
            if (CALCULATE_DT):
                t_var += float(row[0])
            actual_sample += 1
        else:
            x.append(float(row[1]))
            y.append(float(row[y_pos]))
            if (CALCULATE_DT):
                t.append(float(row[0]) + t_var)
                ry.append(float(row[5]))
                t_var = 0
            actual_sample = 0

x = np.asarray(x)
y = np.asarray(y)
ry = np.asarray(ry)

speed = []
theta = []
all_dt_arrays = []
all_x_arrays = []
all_y_arrays = []
all_ry_arrays = []

if(CALCULATE_DT):
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

        if (CALCULATE_DT):
            orientation.append(ry_array[i - 1])
        else:
            orientation.append(math.degrees(math.atan2(dy[i - 1], dx[i - 1]))) #t1 - t0

    for i in range(1, len(dt_array)):
        if (CALCULATE_DT):
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

fig = plt.figure()
using_mpl_scatter_density(fig, theta, speed)


fig = plt.figure()
plt.ylabel("Turning speed (degrees/s)")
plt.title("Turning speed (degrees/s) vs time (s)")
plt.plot([i for i in range(0, len(theta))], theta)


fig = plt.figure()
plt.ylabel("Linear speed (m/s)")
plt.title("Linear speed (m/s) vs time (s)")
plt.plot([i for i in range(0, len(speed))], speed)
plt.plot([i for i in range(0, len(speed))], [speed.mean() for i in range(0, len(speed))])
plt.show()