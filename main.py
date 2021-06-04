import numpy as np
import matplotlib.pyplot as plt
import math 
import csv

f = 120
dt = 1/120

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

x = []
y = []

with open('root_mocap.csv', newline='') as csvfile:
    read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(read_csv, None)
    for row in read_csv:
        x.append(float(row[1]))
        y.append(float(row[3]))

x = np.asarray(x)
y = np.asarray(y)

speed = []
theta = []

for i in range(1, len(x)):
    dx = x[i] - x[i - 1]
    dy = y[i] - y[i - 1]

    speed.append(np.linalg.norm(np.array([dx, dy])) / 2*dt)

    theta.append(math.degrees(math.atan2(dx, dy)))

plt.ylabel("Speed")
plt.xlabel("Turning angle")
plt.scatter(theta, speed, marker='+')
plt.show()
