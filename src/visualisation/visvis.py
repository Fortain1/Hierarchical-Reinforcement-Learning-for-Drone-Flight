import visvis as vv
import numpy as np

import json

with open('trajectories_XRkxfFHiJnuiuurlbMgY.txt', 'r') as f:
        data = json.load(f)

# Create a new figure
fig = vv.figure()

for i in range(50):
    x = data[i][0][:-1]
    y=data[i][1][:-1]
    z =data[i][2][:-1]

    line = vv.plot(x,y,z, lc='black', lw=2, ls='-')
    
    
waypoints =[
    [ 3.0,  3.0,  1.0],
    [3.0, -3.0, 1.0],
    [0.0, 3.0, 1.0],
    [-0.2,-3.0, 1.0]
    ]
obstacles = [
    [3.0,  0.0,   1.0],
    [1.0,  1.0,   1.0],
    [-0.5,  2.0,   1.0],
    [-0.2,  -2.0,   1.0],
    [2.0,  -0.2,   2.1],

            ]
vv.plot(0, 0, 0.8, ms='o', mc='blue', mw=30)

# Define a point (for example, at the end of the trajectory)
for waypoint in waypoints:
    vv.plot(waypoint[0], waypoint[1], waypoint[2], ms='o', mc='green', mw=20, alpha=0.9)

for obstacle in obstacles:
    ob = vv.plot(obstacle[0], obstacle[1], obstacle[2], ms='.', mc='r', mw=50, alpha=0.8)

# Set labels
a = vv.gca()
a.axis.xLabel = 'X axis'
a.axis.yLabel = 'Y axis'
a.axis.zLabel = 'Z axis'
a.axis.showGrid = True
# Set title
vv.title('3D Trajectory and Point Visualization')

# Show the figure
vv.use().Run()