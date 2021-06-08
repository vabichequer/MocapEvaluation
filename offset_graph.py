from matplotlib.text import Annotation
import numpy as np
import matplotlib.pyplot as plt
import math 
import csv
import sys
from distutils.util import strtobool

if (False):
    if (len(sys.argv) < 6):
        print("ERROR: Not enough arguments. The program expects 5 arguments: <bool> graphical time annotation, <float> frequency (Hz), <string> CSV file name, <bool> headers?, <int> source (0: UMANS, 1: Mocap (Xsens)")

    ANNOTATE = strtobool(sys.argv[1])
    f = float(sys.argv[2])
    csv_file_name = str(sys.argv[3])
    skip_header = strtobool(sys.argv[4])
    source = int(sys.argv[5])

    dt = 1/f
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
    time = []

    with open(csv_file_name, newline='') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
        if (skip_header):
            next(read_csv, None) # skipping the first line
        for i, row in enumerate(read_csv):
            time.append(float(row[0]))
            x.append(float(row[1]))
            y.append(float(row[y_pos]))

    x = np.asarray(x)
    y = np.asarray(y)
    time = np.asarray(time)

angles = [0, 15, 30, 45, 60, 75]

x = np.arange(0, 10, 0.1)

for a in angles:
    y = (x * math.tan(math.radians(a))) ** (1/2)

    for i in range(1, len(y)):
        if (i % 3 == 0):
            dy = y[i] - y[i - 1]
            dx = x[i] - x[i - 1]

            tangent_angle = math.atan2(dy, dx)

            x_len, y_len = math.cos(tangent_angle) / 8, math.sin(tangent_angle) / 8
                    
            plt.arrow(x[i], y[i], x_len, y_len, width = 0.02, color="C1")

    plt.plot(x, y)


plt.show()
