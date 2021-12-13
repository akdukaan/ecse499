import matplotlib.pylab as plt
import math
import random

step_size = 0.02
neighborhood_length = 0.081
max_depth = 3 # this is how long our camera stick is
goal_reached = False

# depth, yaw, pitch, rotation
qinit = [0, 0, 0, 0, 0, None]
qgoal = [1, 1, 1, 1, None, None]

# depth, yaw, pitch, rotation, cost, parent
visited_configs = [[qinit[0], qinit[1], qinit[2], qinit[3], 0, None]]

def get_random_valid_position():
    while True:
        p0 = random.random() * max_depth
        p1 = 2 * math.pi * random.random() - math.pi
        p2 = 2 * math.pi * random.random() - math.pi
        p3 = 2 * math.pi * random.random() - math.pi
        q = [p0, p1, p2, p3, None, -1]
        if is_valid_config(q):
            return q

def get_closest_config(q):
    best_coord_index = None
    best_coord_distance = 6 * math.pi + max_depth
    for i in range(len(visited_configs)):
        this_distance = abs(visited_configs[i][0] - q[0]) + \
                        abs(visited_configs[i][1] - q[1]) + \
                        abs(visited_configs[i][2] - q[2]) + \
                        abs(visited_configs[i][3] - q[3])
        if this_distance < best_coord_distance:
            best_coord_distance = this_distance
            best_coord_index = i
    return visited_configs[best_coord_index]

def get_step_toward(qfrom, qto):
    qstep = [None, None, None, None, None, -1]
    for i in range(0, 4):
        if qfrom[i] > qto[i] + step_size:
            qstep[i] = qfrom[i] - step_size
        elif qfrom[i] < qto[i] - step_size:
            qstep[i] = qfrom[i] + step_size
        else:
            qstep[i] = qto[i]
    return qstep

def is_valid_path(qfrom, qto):
    return True

def is_valid_config(q):
    return True

def connect(qnew):
    for i in range(len(visited_configs)):
        distance = get_distance(qnew, visited_configs[i])
        if distance < neighborhood_length:
            if qnew[4] is None or qnew[4] > visited_configs[i][4] + distance:
                qnew[4] = visited_configs[i][4] + distance
                qnew[5] = i
    visited_configs.append(qnew)
    index = len(visited_configs) - 1
    update_neighbors(qnew, index)
    return

def update_neighbors(q, index):
    for i in range(len(visited_configs)):
        distance = get_distance(q, visited_configs[i])
        if distance < neighborhood_length:
            if q[4] + distance < visited_configs[i][4]:
                visited_configs[i][4] = q[4] + distance
                visited_configs[i][5] = index


def get_index_of(q):
    for i in range(len(visited_configs)):
        if visited_configs[i] is q:
            return i

# returns the distance between two configuration spaces
def get_distance(q1, q2):
    return abs(q1[0] - q2[0]) + abs(q1[1] - q2[1]) + abs(q1[2] - q2[2]) + abs(q1[3] - q2[3])


def rtt_star():
    global goal_reached
    for i in range(3500):
        print(i)
        # get a random config
        qrand = get_random_valid_position()
        if goal_reached is False and random.random() < 0.1:
            qrand = qgoal
        # if we can expand a step toward this random config, make a point there
        qnearest = get_closest_config(qrand)
        qnew = get_step_toward(qnearest, qrand)
        if is_valid_path(qnearest, qnew):
            connect(qnew)
            if not goal_reached and is_goal_in_visited():
                goal_reached = True
                print_path(qnew)
    for i in visited_configs:
        if qgoal[0] == i[0] and qgoal[1] == i[1] and qgoal[2] == i[2] and qgoal[3] == i[3]:
            print_path(i)

def is_goal_in_visited():
    for i in visited_configs:
        if qgoal[0] == i[0] and qgoal[1] == i[1] and qgoal[2] == i[2] and qgoal[3] == i[3]:
            return True

def print_path(q):
    print("PRINTING PATH")
    ax = plt.axes(projection='3d')
    xvals = []
    yvals = []
    zvals = []
    while q[5] is not None:
        xvals.append(q[0])
        yvals.append(q[1])
        zvals.append(q[2])
        print(f"[ {round(q[0], 2)}, {round(q[1], 2)}, {round(q[2], 2)}, {round(q[3], 2)}, {round(q[4], 2)}, {round(q[5], 2)}]")
        q = visited_configs[q[5]]
    print(qinit)
    ax.scatter(xvals, yvals, zvals)
    plt.show()


rtt_star()
