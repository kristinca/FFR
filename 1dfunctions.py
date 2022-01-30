import math
import matplotlib.pyplot as plt
import numpy as np


def get_flux_1d():
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

# Np = np.zeros((20, 2))


# def np_h(flux, i, j):
#     Np[:, 0] = [el for i in range(20)]
#     Np[:, 1] = [2 * m for m in flux[i:j]]
#     return Np


H2 = []


# def homogeneity(np2):
#     H2.append((max(np2[:, 1]) - min(np2[:, 1])) / np.average(np2[:, 1]))
#     return H2


# speed = 1 [cm/s]
# here the Si-30 sample reaches the top

for el in t:
    if el == 0:
        Np2[:, 0] = [el for i in range(20)]
        Np2[:, 1] = [2 * el * m for m in f[i:j]]
        i -= 1
        j -= 1
        print(f't={el} [s], i={i}, j={j}')
    # print(Np2, H)
    else:
        Np2[:, 0] = [el for i in range(20)]
        Np2[:, 1] = Np2[:, 1] + [2 * el * m for m in f[i:j]]
        # H[:, 0] = el
        # H[:, 1] = [(max(Np2[:, 1]) - min(Np2[:, 1]))/np.average(Np2[:, 1])]
        H.append((max(Np2[:, 1]) - min(Np2[:, 1])) / np.average(Np2[:, 1]))
        i -= 1
        j -= 1
        print(f't={el} [s], i={i}, j={j}')


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


def going_up_down(n2, ii, jj, up=False, down=False):
    if down:
        td = t_down(n2)
        for el1 in td:
            Np2[:, 0] = [el1 for i in range(20)]
            Np2[:, 1] = Np2[:, 1] + [2 * el1 * m for m in f[ii:jj]]
            H.append((max(Np2[:, 1]) - min(Np2[:, 1])) / np.average(Np2[:, 1]))
            ii -= 1
            jj -= 1
            print(f't={el} [s], i={ii}, j={jj}')
            return n2, ii, jj, Np2, H
    if up:
        tu = t_up(n2)
        for el2 in tu:
            Np2[:, 0] = [el for i in range(20)]
            Np2[:, 1] = Np2[:, 1] + [2 * el2 * m for m in f[ii:jj]]
            H.append((max(Np2[:, 1]) - min(Np2[:, 1])) / np.average(Np2[:, 1]))
            ii += 1
            jj += 1
            print(f't={el} [s], i={ii}, j={jj}')
        return n2, ii, jj, Np2, H


def main(n_init, i_init, j_init, n_times):
    """
    Main program.

    :param n_init: The n-th time of going up and down (one cycle)
    :param i_init: The initial upper index of the sample
    :param j_init: The initial lower index of the sample
    :param n_times: Number of cycles

    :return:
    The n-th time of going up and down (one cycle)
    The initial upper index of the sample
    The initial lower index of the sample
    The P-31 density
    Homogeneity
    """

    for n_time in range(0, n_times):
        nm1, im1, jm1, Np21, H1 = going_up_down(n_time, i_init, j_init, up=True)
        # n, i, j, Np2, H = going_up_down(0, 0, 20, up=True, down=False)
        nm2, im2, jm2, Np22, H2 = going_up_down(n_init, im1, jm1, down=True)
    return nm2, im2, jm2, Np22, H2
    # nm2, im2, jm2, Np22, H2 = going_up_down(nm, im1, jm1, down=True)
    # n1, i, j, Np21, H1 = going_up_down(0, 50, 70, up=False, down=True)


if __name__ == '__main__':
    # the Si-30 sample is moving with v = 1 [cm/s]

    n1, i, j, Np21, H1 = main(0, 0, 20, 40)
    # print(H1)
    # plt.plot(Np21[:, 1], '-*')
    plt.plot(H1, '-')
    plt.title(f'H = f (t)')
    plt.xlabel('time [s]')
    plt.ylabel('H [/]')
    plt.grid(which='major', axis='both')
    plt.show()
