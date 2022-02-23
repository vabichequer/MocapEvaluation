import math

new_radius = 3
x = -1
y = 0

a = math.atan2(-y, -x)

x1 = round(x + new_radius * math.cos(a), 3)
y1 = round(y + new_radius * math.sin(a), 3)

print(x1, y1)