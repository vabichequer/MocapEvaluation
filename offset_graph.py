import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import math
import sys
from scipy.stats import norm
import seaborn as sns
import pandas as pd

def floatToString(inputValue):
    return ('%.15f' % inputValue).rstrip('0').rstrip('.')

def magnitude(vector): 
    return math.sqrt(sum(pow(element, 2) for element in vector))

def plotError(error_array):
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
    plt.xlabel("Offset error")
    plt.ylabel("Proportion")
    title = " Offset error results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    return fig, mu, std

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

if (len(sys.argv) > 2):
    FOLDER_FILES = str(Path(sys.argv[1]))
    radiuses = [floatToString(float(x)) for x in sys.argv[2].split(',')]
    print("Radiuses: ", radiuses)
else:
    FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/MotionMatching/Assets/output/mixamo/1.5"))
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

fig, axs = plt.subplots(1, len(radiuses))

axes = []

if (isinstance(axs, list)):
    for ax in axs:
        axes.append(ax)
else:
    axes.append(axs)

for i in range(0, len(x_planned)):
    axes[i].set_title(str(radiuses[i]) + "m radius curve")
    axes[i].plot(x_final[i], y_final[i], label="final")    
    axes[i].plot(x_planned[i], y_planned[i], label="planned")    
    axes[i].grid()
    axes[i].set_xlabel("X")
    axes[i].set_ylabel("Y")
    axes[i].legend()

fig, axs = plt.subplots(1, len(radiuses))

axes = []

if (isinstance(axs, list)):
    for ax in axs:
        axes.append(ax)
else:
    axes.append(axs)

for i in range(0, len(radiuses)):
    axes[i].set_title(str(radiuses[i]) + "m radius curve offset")
    axes[i].plot(range(0, len(all_distances[i])), all_distances[i], marker="o")
    axes[i].grid()
    axes[i].set_ylabel("Distance")
    axes[i].set_xlabel("Trajectory point")
    error_fig, mu, std = plotError(distances)
    with open(FOLDER_FILES + '/' + str(radiuses[i]) + '_offset_error_mu_std.npy', 'wb') as f:
        np.save(f, np.asarray([mu, std]))

#plt.show()
