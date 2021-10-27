import numpy as np
import matplotlib.pyplot as plt

radius = 5
linearSpeed = 10
circumference = 2 * np.pi * radius
period = circumference / linearSpeed

time = np.arange(0, 2 * np.pi * period, 0.1)

x = np.cos(time * (linearSpeed / radius)) 
y = np.sin(time * (linearSpeed / radius)) 

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
