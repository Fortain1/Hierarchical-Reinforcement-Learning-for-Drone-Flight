import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import json


with open('trajectories_XRkxfFHiJnuiuurlbMgY.txt', 'r') as f:
        data = json.load(f)
t = np.linspace(0, 20, 300)
trajectories = []     
for i in range(50):
    x = data[i][0][:-1]
    y= data[i][1][:-1]
    z =data[i][2][:-1]
    trajectories.append((x,y,z))

waypoints =[
    [ 3.0,  3.0,  1.0],
    [3.0, -3.0, 1.0],
    [0.0, 3.0, 1.0],
    [-0.2,-3.0, 1.0]
    ]

# Set up the figure and the axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fixed_point1 = np.array([3.0,  0.0,   1.0])
fixed_point2 = np.array([1.0,  1.0,   1.0])
fixed_point3 = np.array([-0.5,  2.0,   1.0])
fixed_point4 = np.array([-0.2,  -2.0,   1.0])
fixed_point5 = np.array([2.0,  -0.2,   2.1])

obstacles, = ax.plot([fixed_point1[0], fixed_point2[0], fixed_point3[0], fixed_point4[0], fixed_point5[0]], 
                        [fixed_point1[1], fixed_point2[1], fixed_point3[1], fixed_point4[1], fixed_point5[1]], 
                        [fixed_point1[2], fixed_point2[2], fixed_point3[2], fixed_point4[2], fixed_point5[2]], 'ro',  markersize=20)

fixed_point1 = np.array([ 3.0,  3.0,  1.0])
fixed_point2 = np.array([3.0, -3.0, 1.0])
fixed_point3 = np.array([0.0, 3.0, 1.0])
fixed_point4 = np.array([-0.2,-3.0, 1.0])
waypoints, = ax.plot([fixed_point1[0], fixed_point2[0], fixed_point3[0], fixed_point4[0]], 
                        [fixed_point1[1], fixed_point2[1], fixed_point3[1], fixed_point4[1]], 
                        [fixed_point1[2], fixed_point2[2], fixed_point3[2], fixed_point4[2]], 'go',  markersize=10)
# Create plot elements for each trajectory
lines = []
points = []
colors = ['purple', 'yellow', 'b', 'c', 'm', 'y', 'k']  # Add more colors if needed
for i, (x, y, z) in enumerate(trajectories):
    color = colors[i % len(colors)]
    line, = ax.plot([], [], [], lw=1, label=f'Trajectory {i+1}', color='black', alpha=0.8)
    point, = ax.plot([], [], [], 'o', color=color, markersize=8)
    lines.append(line)
    points.append(point)


# Setting the axes properties
all_x = np.concatenate([x for x, y, z in trajectories])
all_y = np.concatenate([y for x, y, z in trajectories])
all_z = np.concatenate([z for x, y, z in trajectories])
ax.set_xlim((np.min(all_x), np.max(all_x)))
ax.set_ylim((np.min(all_y), np.max(all_y)))
ax.set_zlim((np.min(all_z), np.max(all_z)))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Initialization function: plot the background of each frame
def init():
    for line, point in zip(lines, points):
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
    return lines + points + [waypoints] + [obstacles]

# Animation function which updates figure data. This is called sequentially
def animate(i):
    for line, point, (x, y, z) in zip(lines, points, trajectories):
        line.set_data(x[:i], y[:i])
        line.set_3d_properties(z[:i])
        point.set_data(x[i:i+1], y[i:i+1])
        point.set_3d_properties(z[i:i+1])
    return lines + points + [waypoints]+ [obstacles]

# Call the animator. blit=True means only re-draw the parts that have changed.
anim = FuncAnimation(fig, animate, init_func=init,
                     frames=len(t), interval=15, blit=True)

gif_writer = PillowWriter(fps=20)
anim.save("multiple_trajectories.gif", writer=gif_writer)

plt.show()
