from matplotlib.text import Annotation
import numpy as np
import matplotlib.pyplot as plt
import math 
import csv
import sys
from distutils.util import strtobool
from scipy.stats import gaussian_kde

if (len(sys.argv) < 6):
    print("ERROR: Not enough arguments. The program expects 5 arguments: <bool> graphical time annotation, <float> frequency (Hz), <string> CSV file name, <bool> headers?, <int> source (0: UMANS, 1: Mocap (Xsens)")

ANNOTATE = strtobool(sys.argv[1])
CALCULATE_DT = False

if (sys.argv[2] == "Unity"):
    CALCULATE_DT = True
    t = []
    f = "Unity"
    dt = "Not defined yet"
else:
    f = float(sys.argv[2])
    dt = 1/f

csv_file_name = str(sys.argv[3])
skip_header = strtobool(sys.argv[4])
source = int(sys.argv[5])


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

with open(csv_file_name, newline='') as csvfile:
    read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
    if (skip_header):
        next(read_csv, None) # skipping the first line
    for i, row in enumerate(read_csv):
        x.append(float(row[1]))
        y.append(float(row[y_pos]))
        if (CALCULATE_DT):
            t.append(float(row[0]))

x = np.asarray(x)
y = np.asarray(y)

speed = []
theta = []

if(CALCULATE_DT):
    dt_array = []
    for i in range(1, len(t)):
        dt_array.append(t[i] - t[i - 1])

for i in range(1, len(x)):
    dx = x[i - 1] - x[i]
    dy = y[i - 1] - y[i]

    #t0 = math.degrees(math.atan2(y[i - 1], x[i - 1]))
    #t1 = math.degrees(math.atan2(y[i], x[i]))

    dtheta = math.degrees(math.atan2(dy, dx)) #t1 - t0
    
    if (CALCULATE_DT):
        dt = dt_array[i - 1]
    
    speed.append(np.linalg.norm(np.array([dx, dy])) / dt)
    theta.append(dtheta / dt)
    print(dtheta, dt, theta[i - 1])

    # Atan2 already returns the result between Pi and -Pi, so no need to check this
"""     if (abs(dtheta) > 180):
        print('-' * 25)
        print("dtheta: ", dtheta)
        # maybe there is an error here. The int should be used also
        dec, int = math.modf(dtheta / 180)
        if (dec == 0):
            dtheta = 0
        else:
            degree = (abs(dec) - 1) * 180
            actual = -math.copysign(degree, dtheta)
            dtheta = actual
            print("degree: ", degree)
            print("dec: ", dec)
            print("actual: ", actual)
        print('-' * 25) """

speed = np.asarray(speed)
theta = np.asarray(theta)

fig, ax = plt.subplots()
plt.ylabel("Speed (m/s)")
plt.xlabel("Turning speed (degrees/s)")

# Calculate density based on a gaussial kernel density estimate, which works like a histogram
xy = np.vstack([theta, speed])
z = gaussian_kde(xy)(xy)

# Sort denser points to be printed last, thus showing them on top
idx = z.argsort()
theta, speed, z = theta[idx], speed[idx], z[idx]

sc = plt.scatter(theta, speed, c=z)
plt.colorbar(sc)


if (ANNOTATE):
    txt = 0.1
    for i in range(0, len(speed)):
        if (i % 10 == 0):
            ax.annotate("%.2f" % txt, (theta[i], speed[i]))
        txt += 0.1

plt.show()
