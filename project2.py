import matplotlib.pylab as plt
import math
import random
import numpy as np

# pitch, yaw, displacement, roll,
def get_pitch(q):
    return q[0]

def get_yaw(q):
    return q[1]

def get_depth(q):
    return q[2]

def get_roll(q):
    return q[3]

def get_x(q):
    return q[4]

def get_y(q):
    return q[5]

def get_z(q):
    return q[6]

def get_cost(q):
    return q[7]

def get_parent(q):
    return q[8]

def get_index(q):
    return q[9]

## TYLERS METHODS

def AofDH(x, DH):
  r = DH[x,0]
  d = DH[x,1]
  a = DH[x,2]
  q = DH[x,3]

  A = np.array([[np.cos(q), -np.sin(q)*np.cos(a), np.sin(q)*np.sin(a), r*np.cos(q)],
                [np.sin(q), np.cos(q)*np.cos(a), -np.cos(q)*np.sin(a), r*np.sin(q)],
                [0, np.sin(a), np.cos(a), d],
                [0, 0, 0, 1]])
  return A


def collides(q, obstacles):
    return False
    #return check_collision(get_endpoints(q), obstacles)


def check_collision(endpts, obstacles):
    end = endpts[3][0:3]  # endpoints of probe
    x_end = end[0]
    y_end = end[1]
    z_end = end[2]

    x = np.linspace(0, x_end, 1000)
    y = np.linspace(0, y_end, 1000)
    z = np.linspace(0, z_end, 1000)

    cam = endpts[4][0:3]  # endpoints of camera point
    x_cam = cam[0]
    y_cam = cam[1]
    z_cam = cam[2]

    x = np.concatenate((x, np.linspace(x_end, x_cam, 100)))
    y = np.concatenate((y, np.linspace(y_end, y_cam, 100)))
    z = np.concatenate((z, np.linspace(z_end, z_cam, 100)))

    x_ob = obstacles[0]
    y_ob = obstacles[1]
    z_ob = obstacles[2]

    i = x.size - 1
    check = False

    while (i >= 0 and check != True):
        diff = min(np.sqrt((x_ob - x[i]) ** 2 + (y_ob - y[i]) ** 2 + (z_ob - z[i]) ** 2))

        if diff < 0.05:
            check = True
            return check
        else:
            i = i - 1

    return check


def get_endpoints(q):

    q1 = q[0]  # pitch
    q2 = q[1]  # yaw
    q3 = q[2]  # displacement
    q4 = q[3]  # roll

    r = np.array([0, 0, 5 + q3, 0, .5])
    a = np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, 0, 0])
    d = np.array([0, 0, 0, 0, 0])
    q = np.array([q1, q2, 0, q4, 0])

    DH = np.array([r, d, a, q])
    DH = DH.transpose()
    # print(DH)
    # print(DH)

    A1 = AofDH(0, DH)
    A2 = AofDH(1, DH)
    A3 = AofDH(2, DH)
    A4 = AofDH(3, DH)
    A5 = AofDH(4, DH)

    A10 = A1
    A20 = np.dot(A1, A2)
    A30 = np.dot(A20, A3)
    A40 = np.dot(A30, A4)
    A50 = np.dot(A40, A5)

    endpts = np.array([A10[:, 3], A20[:, 3], A30[:, 3], A40[:, 3], A50[:, 3]])
    return endpts


x = np.linspace(0, 2 * np.pi, 21)
pts = np.zeros((x.size, 3))

fig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(x.size):
    q = np.array([np.pi / 8, np.pi / 3, 5, x[i]])
    pt = get_endpoints(q)
    ax.scatter(pt[0][0], pt[0][1], pt[0][2])
    ax.scatter(pt[1][0], pt[1][1], pt[1][2])
    ax.scatter(pt[2][0], pt[2][1], pt[2][2])
    ax.scatter(pt[3][0], pt[3][1], pt[3][2])
    ax.scatter(pt[4][0], pt[4][1], pt[4][2])

# get_endpoints(np.array([0, 0, 0, 3 * np.pi / 2]))


# making box
def make_box(A, B, G, X, Y, Z):
    y = np.arange(0, 2, .1)
    z = np.arange(0, 2, .1)
    mesh = np.meshgrid(y, z)
    base = np.array([np.zeros((y.size)), mesh[0][0], mesh[1][0], np.ones((y.size))])
    for i in range(y.size - 2):
        new = np.array([np.zeros((y.size)), mesh[0][i + 1], mesh[1][i + 1], np.ones((y.size))])
        base = np.concatenate((base, new), axis=1)

    x = np.arange(0, 1, .1)
    z = np.arange(0, 2, .1)
    mesh = np.meshgrid(x, z)
    for i in range(z.size - 1):
        new = np.array([mesh[0][i], np.zeros((x.size)), mesh[1][i], np.ones((x.size))])
        base = np.concatenate((base, new), axis=1)
        new = np.array([mesh[0][i], 2 * np.ones((x.size)), mesh[1][i], np.ones((x.size))])
        base = np.concatenate((base, new), axis=1)

    x = np.arange(0, 1, .1)
    y = np.arange(0, 2, .1)
    mesh = np.meshgrid(x, z)
    for i in range(z.size - 1):
        new = np.array([mesh[0][i], mesh[1][i], np.zeros((x.size)), np.ones((x.size))])
        base = np.concatenate((base, new), axis=1)
        new = np.array([mesh[0][i], mesh[1][i], 2 * np.ones((x.size)), np.ones((x.size))])
        base = np.concatenate((base, new), axis=1)

    R = np.array([[np.cos(A) * np.cos(B), np.cos(A) * np.sin(B) * np.sin(G) - np.sin(A) * np.cos(G),
                   np.cos(A) * np.sin(B) * np.cos(G) + np.sin(A) * np.sin(G), X],
                  [np.sin(A) * np.cos(B), np.sin(A) * np.sin(B) * np.sin(G) + np.cos(A) * np.cos(G),
                   np.sin(A) * np.sin(B) * np.cos(G) - np.cos(A) * np.sin(G), Y],
                  [-np.sin(B), np.cos(B) * np.sin(G), np.cos(B) * np.cos(G), Z],
                  [0, 0, 0, 1]])

    box = np.dot(R, base)
    box = box[0:3]
    goal = np.dot(R, np.transpose(np.array([0, 1, 1, 1])))
    return box, goal


base, goal = make_box(0, 0, 0, 0, 0, 0)

ax.scatter(base[0], base[1], base[2])
ax.scatter(goal[0], goal[1], goal[2], 'r')
# ax.set_xlim([0, 10])
# ax.set_ylim([0, 10])
# ax.set_zlim([0, 10])

print(check_collision(get_endpoints(q), base))


## KAANS METHODS
def rrt_star(qinit, goal, obstacles):
    init_endpoints = get_endpoints(qinit)[4]
    qinit.append(init_endpoints[0])  # X
    qinit.append(init_endpoints[1])  # Y
    qinit.append(init_endpoints[2])  # Z
    qinit.append(0)  # Cost
    qinit.append(-1)  # Parent
    qinit.append(0)  # Index
    global visited_configs
    visited_configs = [qinit]
    goal_reached = False
    global qgoal
    qgoal[5] = goal[0]
    qgoal[6] = goal[1]
    qgoal[7] = goal[2]

    for i in range(n):
        print("Iteration " + str(i))
        qrand = get_random_config()
        qnearest = get_nearest_config(qrand)
        qnew = get_step_toward(qnearest, qrand)
        qnew[7] = get_distance(qnearest, qnew) + get_cost(qnearest)
        if not collides(qnew, obstacles):
            visited_configs.append(qnew)
            connect(qnew)
            if not goal_reached and get_found_goal():
                goal_reached = True
                print_path(get_found_goal())
                original_cost = str(get_cost(get_found_goal()))
                print("original cost is " + original_cost + "\n")
    if get_found_goal():
        print_path(get_found_goal())
        print("original cost was " + original_cost)
        print("new cost is " + str(get_cost(get_found_goal())))
        return
    else:
        print(f"no goal was found after {n} iterations")

def print_path(q):
    index_list = []
    print("Found goal! Printing path")
    while get_parent(q) >= 0:
        index_list.append(q[9])
        print(f"[{round(q[4], 2)}, {round(q[5], 2)}, {round(q[6], 2)}, {round(q[7], 2)}]")
        q = visited_configs[get_parent(q)]
    index_list.append(q[9])
    print("path is " + str(index_list))

def get_found_goal():
    for item in visited_configs:
        if get_distance(item, qgoal) < threshold:
            return item
    return False


def get_neighbors(q):
    neighbors = []
    for item in visited_configs:
        if get_distance(item, q) < neighborhood_length:
            neighbors.append(item)
    return neighbors


def connect(q):
    neighbors = get_neighbors(q)
    # Figure out the actual best path to get to q
    for i in range(len(neighbors)):
        if get_cost(neighbors[i]) + get_distance(q, neighbors[i]) < get_cost(q):
            visited_configs[q[9]][7] = get_cost(neighbors[i]) + get_distance(q, neighbors[i])
            visited_configs[q[9]][8] = get_index(neighbors[i])
            q = visited_configs[q[9]]
    # Update any of it's neighbors with this new cost
    for neighbor in neighbors:
        if get_cost(neighbor) > get_cost(q):
            connect_neighbor(neighbor, q)


def connect_neighbor(neighbor, q):
    if get_cost(neighbor) > get_cost(q) + get_distance(q, neighbor):
        visited_configs[neighbor[9]][7] = get_cost(q) + get_distance(q, neighbor)
        visited_configs[neighbor[9]][8] = get_index(q)
        neighbor = visited_configs[neighbor[9]]
        for n in get_neighbors(neighbor):
            connect_neighbor(n, neighbor)


# Take a step toward the configspace of the new item
def get_step_toward(qfrom, qto):
    qstep = []
    for i in range(4):
        qstep.append(qto[i])
    endpoint = get_endpoints(qstep)[4]
    qstep.append(endpoint[0])  # X
    qstep.append(endpoint[1])  # Y
    qstep.append(endpoint[2])  # Z
    tries = 0
    while get_distance(qfrom, qstep) > step_size and tries < 20:
        tries = tries + 1
        for i in range(4):
            qstep[i] = (qfrom[i] + qstep[i]) / 2
        endpoint = get_endpoints(qstep)[4]
        qstep[4] = endpoint[0]  # X
        qstep[5] = endpoint[1]  # Y
        qstep[6] = endpoint[2]  # Z
    qstep.append(-1)  # Cost
    qstep.append(get_index(qfrom))  # Parent
    qstep.append(get_index(qto))  # Index

    return qstep

# get the worldspace distance
def get_distance(q1, q2):
    return abs(get_x(q1) - get_x(q2)) + \
           abs(get_y(q1) - get_y(q2)) + \
           abs(get_z(q1) - get_z(q2))

# get nearest config in worldspace
def get_nearest_config(q):
    qclosest = visited_configs[0]
    for item in visited_configs:
        if get_distance(item, q) < get_distance(qclosest, q):
            qclosest = item
    return qclosest

def get_random_config():
    pitch = 2 * math.pi * random.random() - math.pi
    yaw = 2 * math.pi * random.random() - math.pi
    depth = random.random() * max_depth
    roll = 2 * math.pi * random.random() - math.pi
    q = [pitch, yaw, depth, roll]
    endpoints = get_endpoints(q)[4]
    q.append(endpoints[0])  # X
    q.append(endpoints[1])  # Y
    q.append(endpoints[2])  # Z
    q.append(-1)  # Cost
    q.append(-1)  # Parent
    q.append(len(visited_configs))  # Index
    return q

visited_configs = []
qgoal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Change the middle three numbers to change the goal worldspace. All other numbers here are ignored


### THESE NEXT 7 LINES ARE THE CONFIGURABLE PARTS
step_size = 1.5  # How close in worldspace the points need to be in order to create a new point at that location
neighborhood_length = 1.5  # After connecting a point, how close in worldspace do its neighbors need to be for us to update them
max_depth = 3  # this is how long our camera stick is
threshold = 3.8  # This is how close we need to get to our goal in order to say we've reached it. worldspace
n = 2000  # This is the number of iterations we'll go through

obstacles, goal = make_box(0, 0, 0, 4.12, -1, -8.7)
rrt_star([0,0,0,0], goal, obstacles)
