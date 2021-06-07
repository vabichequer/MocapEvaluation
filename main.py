import numpy as np
import matplotlib.pyplot as plt
import math 
import csv

ANNOTATE = False

f = 120
dt = 1/f

print('*' * 25)
print("Program setup:")
print("Frequency (Hz): ", f)
print("dt: ", dt)
print('*' * 25)

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
    next(read_csv, None) # skipping the first line
    for i, row in enumerate(read_csv):
        x.append(float(row[1]))
        y.append(float(row[3]))

x = np.asarray(x)
y = np.asarray(y)

speed = []
theta = []

for i in range(1, len(x)):
    dx = x[i] - x[i - 1]
    dy = y[i] - y[i - 1]

    t0 = math.degrees(math.atan2(y[i - 1], x[i - 1]))
    t1 = math.degrees(math.atan2(y[i], x[i]))

    dtheta = t1 - t0

    if (abs(dtheta) >= 180):
        print('-' * 25)
        print("dtheta: ", dtheta)
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
        print('-' * 25)

    speed.append(np.linalg.norm(np.array([dx, dy])) / dt)

    theta.append(dtheta / dt)

fig, ax = plt.subplots()
plt.ylabel("Speed (m/s)")
plt.xlabel("Turning speed (degrees/s)")
plt.scatter(theta, speed, marker='+')


if (ANNOTATE):
    txt = 0.1
    for i in range(0, len(speed)):
        if (i % 10 == 0):
            ax.annotate("%.2f" % txt, (theta[i], speed[i]))
        txt += 0.1


plt.show()
