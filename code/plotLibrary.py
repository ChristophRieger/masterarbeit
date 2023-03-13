import matplotlib.pyplot as plt
import numpy as np
import math

plt.close("all")

# plt.figure()
fig, ax = plt.subplots(1, 1)
#draw circle
r = 1
angles = np.linspace(0 * math.pi, 2 * math.pi, 100 )
anglesZ = np.linspace(0 * math.pi, 1/2 * math.pi, 50 )
xs = r * np.cos(angles)
ys = r * np.sin(angles)
plt.plot(xs, ys, color = 'black')

#draw radius
# plt.plot([0, 1], [0, 0], color = 'purple')
plt.gca().annotate('Radius', xy=(0.5, -0.2), xycoords='data', fontsize=10)
#draw arc
arc_angles = np.linspace(0 * math.pi, math.pi/4, 20)
arc_xs = r * np.cos(arc_angles)
arc_ys = r * np.sin(arc_angles)
# plt.plot(arc_xs, arc_ys, color = 'blue', lw = 1)
#draw another radius
# plt.plot(r * np.cos(math.pi /4), r * np.sin( math.pi / 4), marker = 'o', color = 'red')
# plt.plot([0, r * np.cos(math.pi /4)], [0, r * np.sin( math.pi / 4)], color = "green")

x = r * np.cos(anglesZ)
y = r * np.sin(anglesZ)
# plt.plot(x, y, marker = 'o', color = 'red')
# plt.fill()

colors = ['red', 'orange', 'yellow', 'green', 'blue']
for i in range(5):
  ax.fill_between(x, y, color=colors[i])

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.gca().set_aspect('equal')
plt.show()