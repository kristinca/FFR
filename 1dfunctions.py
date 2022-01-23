import math
import matplotlib.pyplot as plt
import numpy as np


y = [i for i in range(-35, 36)]
y1 = [i for i in y if i % 5 == 0]

hr = 70
f0 = 10**13
f = []

for el in y:
    f.append(f0*math.cos(math.radians((math.pi*el)/70)))

hv = {i for i in range(20)}

t = [i for i in range(25)]

i = 25
j = 45

Np2 = np.zeros((20, 2))
# Np2[:, 1] = np.ones(20)

# H = np.zeros((1, 2))
H = []

# speed = 1 [cm/s]
# here the Si-30 sample reaches the top
for el in t:
    if el == 0:
        Np2[:, 0] = [el for i in range(20)]
        Np2[:, 1] = [2*m for m in f[i:j]]
    # H[:, 0] = el
    # H[:, 1] = [(max(Np2[:, 1]) - min(Np2[:, 1]))/np.average(Np2[:, 1])]
        H.append((max(Np2[:, 1]) - min(Np2[:, 1]))/np.average(Np2[:, 1]))
        i -= 1
        j -= 1
        print(i, j)
    # print(Np2, H)
    else:
        Np2[:, 0] = [el for i in range(20)]
        Np2[:, 1] = Np2[:, 1] + [2 * el * m for m in f[i:j]]
        # H[:, 0] = el
        # H[:, 1] = [(max(Np2[:, 1]) - min(Np2[:, 1]))/np.average(Np2[:, 1])]
        H.append((max(Np2[:, 1]) - min(Np2[:, 1])) / np.average(Np2[:, 1]))
        i -= 1
        j -= 1
        print(i, j)


def t_up(n):
    """
    :param n: n-th number of time the sample goes up
    :return: list with the time interval needed for the sample to reach the top, with dt=1[s]
    """
    return [i for i in range(25+100*n, 75+100*n)]


def t_down(n):
    """
    :param n: n-th number of time the sample goes down
    :return: list with the time interval needed for the sample to reach the bottom, with dt=1[s]
    """
    return [i for i in range(75+100*n, 125+100*n)]


def going_up_down(n, i, j, up=False, down=False):
    if down==True:
        td = t_down(n)
        for el in td:
            Np2[:, 0] = [el for i in range(20)]
            Np2[:, 1] = Np2[:, 1] + [2 * el * m for m in f[i:j]]
            H.append((max(Np2[:, 1]) - min(Np2[:, 1])) / np.average(Np2[:, 1]))
            i += 1
            j += 1
            print(i, j)
            return n, i, j, Np2, H
    if up==True:
        tu = t_up(n)
        for el in tu:
            Np2[:, 0] = [el for i in range(20)]
            Np2[:, 1] = Np2[:, 1] + [2 * el * m for m in f[i:j]]
            H.append((max(Np2[:, 1]) - min(Np2[:, 1])) / np.average(Np2[:, 1]))
            i -= 1
            j -= 1
            print(i, j)
            return n, i, j, Np2, H


going_up_down(0,0,20, up=True)
