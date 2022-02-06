import math
import matplotlib.pyplot as plt
import numpy as np


def get_flux_1d():
    """ 1 D flux """
    y = [i for i in range(-35, 36)]
    y1 = [i for i in y if i % 5 == 0]
    # hr = 70
    f0 = 10**13
    f1 = []
    for el1 in y:
        f1.append(f0*math.cos(math.radians((math.pi*el1)/70)))
    return f1


f = get_flux_1d()
# plt.plot(f)
# plt.show()

hv = {i for i in range(20)}

t = [i for i in range(25)]

i = 25
j = 45

Np2 = np.zeros((20, 2))
# Np2[:, 1] = np.ones(20)

# H = np.zeros((1, 2))
H = []



def t_up(n_up):
    """
    :param n_up: n-th number of time the sample goes up
    :return: list with the time interval needed for the sample to reach the top, with dt=1[s]
    """
    return [i for i in range(25+100*n_up, 75+100*n_up)]


def t_down(n_down):
    """
    :param n_down: n-th number of time the sample goes down
    :return: list with the time interval needed for the sample to reach the bottom, with dt=1[s]
    """
    return [i for i in range(75+100*n_down, 125+100*n_down)]


def t_start():
    """
    :return: list with the time interval needed for the sample to reach the bottom from initial time = 0
    """
    return [i for i in range(0, 25)]
ii = []
jj = []

def going_down_first_time(i, j):
    td = t_start()
    for el1 in td:
        print(f't={el1} [s], i={i}, j={j}')
        Np2[:, 0] = [el1 for i in range(20)]
        Np2[:, 1] = Np2[:, 1] + [2 * el1 * m for m in f[i:j]]
        H.append((max(Np2[:, 1]) - min(Np2[:, 1])) / np.average(Np2[:, 1]))
        i += 1
        j += 1
    return i, j, Np2, H

def going_up(n, ii,jj):
    print('\nGoing UP\n')
    tu = t_up(n)
    for el2 in tu:
        print(f't={el2} [s], i={ii}, j={jj}')
        Np2[:, 0] = [el2 for i in range(20)]
        Np2[:, 1] = Np2[:, 1] + [2 * el2 * m for m in f[ii:jj]]
        H.append((max(Np2[:, 1]) - min(Np2[:, 1])) / np.average(Np2[:, 1]))
        ii -= 1
        jj -= 1
    return ii, jj, Np2, H


def going_down(n, ii, jj):
    print('\nGoing DOWN\n')
    td = t_down(n)
    for el1 in td:
        print(f't={el1} [s], i={ii}, j={jj}')
        Np2[:, 0] = [el1 for i in range(20)]
        Np2[:, 1] = Np2[:, 1] + [2 * el1 * m for m in f[ii:jj]]
        H.append((max(Np2[:, 1]) - min(Np2[:, 1])) / np.average(Np2[:, 1]))
        ii += 1
        jj += 1
    return ii, jj, Np2, H


def up_down(n_times):
    for n in range(0, n_times):
        iii1, ji1, Npi1, Hi1 = going_up(n, iii[-1], jjj[-1])
        iii.append(iii1)
        jjj.append(ji1)
        iii2, ji2, Npi2, Hi2 = going_down(n, iii[-1], jjj[-1])
        iii.append(iii2)
        jjj.append(ji2)
    return Npi2, Hi2
#
# i2, j2, Np22, H2 = going_up(0, i1, j1)
# i3, j3, Np3, H3 = going_down(0, i2, j2)
# i4, j4, Np4, H4 = going_up(1, i3, j3)
# i5, j5, Np5, H5 = going_down(1, i4, j4)
# i6, j6, Np6, H6 = going_up(2, i5, j5)
# i5, j7, Np7, H7 = going_down(2, i6, j6)
i1, j1, Np1, H1 = going_down_first_time(25, 45)
iii = [i1]
jjj = [j1]
N6, H6 = up_down(5)

# plt.plot(N6[:, 1], '-')
plt.plot(H6, '-')
plt.title(f'H = f (t)')
plt.xlabel('time [s]')
plt.ylabel('H [/]')
plt.grid(which='major', axis='both')
plt.show()
