import numpy as np
import matplotlib.pyplot as plt
import csv

radius = 5
linearSpeed = 1.4
circumference = 2 * np.pi * radius
period = circumference / linearSpeed
w = (linearSpeed / radius)
time = np.arange(0, period, 0.1)

x = radius * np.cos(time * w)
y = radius * np.sin(time * w)

print("Circumference: ", circumference)
print("Period: ", period)

fig, ax = plt.subplots()

plt.title(str(radius) + "m radius curve")
plt.scatter(x, y, label="curve")
plt.grid()
ax.set_aspect('equal', 'box')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# open the file in the write mode
f = open('C:/Users/vabicheq/Documents/mocap-evaluation/trajectory_' + str(radius) + '.csv', 'w', newline='')

# create the csv writer
writer = csv.writer(f)

# write a row to the csv file
for x, y in zip(x, y):
    writer.writerow([x, y])

# close the file
f.close()