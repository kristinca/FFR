import math
import matplotlib.pyplot as plt
import numpy as np


def get_flux_1d():
    """
    :return: The reactor flux in 1D.
    """
    y = [i for i in range(-35, 36)]
    y1 = [i for i in y if i % 5 == 0]
    # hr = 70
    # Sigma (Si-30) = sigma(Si-30)(n,gamma)*N(Si-30)
    sigmaSi30 = 107.2 * 10**(-27)
    # NSi30 = (roSi30*Na)/MSi30
    NSi30 = (2.33*6.02214*10**23)/29.9737701
    SigmaSi30 = sigmaSi30*NSi30
    print(f'SigmaSi30 = {SigmaSi30}, NSi30= {NSi30}')
    f0 = 10**13
    f1 = []
    for el1 in y:
        f1.append(f0*SigmaSi30*math.cos(math.radians((math.pi*el1)/70)))
    return f1


# get the flux in 1D
f = get_flux_1d()

# plt.plot(f)
# plt.show()


i = 25
j = 45

Np2 = np.zeros((20, 2))

H = []


def t_up(n_up):
    """
    :param n_up: n-th number of time the sample goes up
    :return: list - the time interval needed for the sample to reach the top with dt=1[s], v=1[cm/s]
    """
    return [tup for tup in range(25+100*n_up, 75+100*n_up)]


def t_down(n_down):
    """
    :param n_down: n-th number of time the sample goes down
    :return: list - the time interval needed for the sample to reach the bottom with dt=1[s], v=1[cm/s]
    """
    return [tdd for tdd in range(75+100*n_down, 125+100*n_down)]


def t_start():
    """
    :return: list - the time interval needed for the sample to reach the bottom from initial t=0[s] with v=1[cm/s]
    """
    return [ts for ts in range(0, 25)]


# initialize ii-list with highest index position of the sample and jj-list with lowest index position of the sample
ii = []
jj = []


def going_down_first_time(ift, jft):
    """
    A function that calculates from initial time till the Si-30 sample reaches the bottom.
    :param ift: highest index of the Si-30 sample on the initial flux indexes scale
    :param jft: lowest index of the Si-30 sample on the initial flux indexes scale
    :return: highest and lowest index of the Si-30 sample on the initial flux indexes scale,
             P-31 number density, homogeneity
    """
    td = t_start()
    for el1 in td:
        print(f't={el1} [s], i={ift}, j={jft}')
        Np2[:, 0] = [el1 for i in range(20)]
        Np2[:, 1] = Np2[:, 1] + [2 * el1 * m for m in f[ift:jft]]
        if el1 != 0:
            H.append((max(Np2[:, 1]) - min(Np2[:, 1])) / np.average(Np2[:, 1]))
        ift += 1
        jft += 1
    return ift, jft, Np2, H


def going_up(n, ii, jj):
    """
    A function that calculates from initial time till the Si-30 sample reaches the top from the bottom.
    :param n: the n-th time of going up
    :param ii: highest index of the Si-30 sample on the initial flux indexes scale
    :param jj: lowest index of the Si-30 sample on the initial flux indexes scale
    :return: highest and lowest index of the Si-30 sample on the initial flux indexes scale,
             P-31 number density, homogeneity
    """
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
    """
    A function that calculates from initial time till the Si-30 sample reaches the bottom from the top.
    :param n: the n-th time of going down
    :param ii: highest index of the Si-30 sample on the initial flux indexes scale
    :param jj: lowest index of the Si-30 sample on the initial flux indexes scale
    :return: highest and lowest index of the Si-30 sample on the initial flux indexes scale,
             P-31 number density, homogeneity
    """
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
    """
    The driver function.
    :param n_times: Number of times of going up and down of the Si-30 sample.
    :return: P-31 number density, homogeneity
    """
    for n in range(0, n_times):
        iii1, ji1, Npi1, Hi1 = going_up(n, iii[-1], jjj[-1])
        iii.append(iii1)
        jjj.append(ji1)
        iii2, ji2, Npi2, Hi2 = going_down(n, iii[-1], jjj[-1])
        iii.append(iii2)
        jjj.append(ji2)
    return Npi2, Hi2


if __name__ == '__main__':
    i1, j1, Np1, H1 = going_down_first_time(25, 45)
    iii = [i1]
    jjj = [j1]
    # going up and down 5 times
    N, H = up_down(5)

    # plot
    # plt.plot(N[:, 1], '-')
    plt.plot(H, '-')
    plt.title(f'H = f (t)')
    plt.xlabel('time [s]')
    plt.ylabel('H [/]')
    plt.grid(which='major', axis='both')
    plt.show()
