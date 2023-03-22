import math
import matplotlib.pyplot as plt
import numpy as np

from sample_1_dimension import get_flux_1d


def t_up(
        n_up: int
) -> list:
    """
    A function that calculates from a given time till the Si-30 sample reaches the top from the bottom

    :param n_up: int, n-th number of time the sample goes up
    :return: list, the time interval needed for the sample to reach the top with dt=1[s], v=1[cm/s]
    """
    return [tup for tup in range(25+100*n_up, 75+100*n_up)]


def t_down(
        n_down: int
) -> list:
    """

    A function that calculates from a given time till the Si-30 sample reaches the top from the bottom.
    :param n_down: n-th number of time the sample goes down
    :return: list, the time interval needed for the sample to reach the bottom with dt=1[s], v=1[cm/s]
    """
    return [tdd for tdd in range(75+100*n_down, 125+100*n_down)]


def t_start() -> list:
    """
    :return: list - the time interval needed for the sample to reach the bottom from initial t=0[s] with v=1[cm/s]
    """
    return [ts for ts in range(0, 25)]


def going_down_first_time(ift, jft, reactor_flux, sigmasi30):
    """
    A function that calculates from initial time till the Si-30 sample reaches the bottom

    :param ift: highest index of the Si-30 sample on the initial flux indexes scale
    :param jft: lowest index of the Si-30 sample on the initial flux indexes scale
    :param reactor_flux: list, the reactor flux
    :param sigmasi30: float, the Sigma(Si-30)
    :return: highest and lowest index of the Si-30 sample on the initial flux indexes scale,
             P-31 number density, homogeneity
    """
    h_0 = []
    td = t_start()
    y = [i for i in range(1, 6)]
    for el1 in td:
        print(f't={el1} [s], i={ift}, j={jft}')
        for el2 in y:
            np2_[:, 0] = [el1 for i in range(20)]
            np2_[:, el2] = np2_[:, el2] + np.transpose(reactor_flux[ift:jft]) * math.exp(-sigmasi30 * (el2-1))
            if el1 != 0:
                h_0[:, 0] = [el1 for i in range(20)]
                h_0[:, el2] = ((max(np2_[:, el2]) - min(np2_[:, el2])) / np.average(np2_[:, el2]))
        if el1 != 0:
            h_0.append(np.average(h_[:, 1:]))
        ift += 1
        jft += 1
    return ift, jft, np2_, h_0


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
    y = [i for i in range(1, 6)]
    for el1 in tu:
        print(f't={el1} [s], i={ii}, j={jj}')
        for el2 in y:
            np2_[:, 0] = [el1 for i in range(20)]
            np2_[:, el2] = np2_[:, el2] + np.transpose(f1d[ii:jj]) * math.exp(-sSi30 * (4-el2))
            h_[:, 0] = [el1 for i in range(20)]
            h_[:, el2] = ((max(np2_[:, el2]) - min(np2_[:, el2])) / np.average(np2_[:, el2]))
        h.append(np.average(h_[:, 1:]))
        ii -= 1
        jj -= 1
    return ii, jj, np2_, h_, h


def going_down(n, id, jd):
    """
    A function that calculates from initial time till the Si-30 sample reaches the bottom from the top.
    :param n: the n-th time of going down
    :param id: highest index of the Si-30 sample on the initial flux indexes scale
    :param jd: lowest index of the Si-30 sample on the initial flux indexes scale
    :return: highest and lowest index of the Si-30 sample on the initial flux indexes scale,
             P-31 number density, homogeneity
    """
    print('\nGoing DOWN\n')
    td = t_down(n)
    y = [i for i in range(1, 6)]
    for el1 in td:
        print(f't={el1} [s], i={id}, j={jd}')
        for el2 in y:
            np2_[:, 0] = [el1 for i in range(20)]
            np2_[:, el2] = np2_[:, el2] + np.transpose(f1d[id:jd]) * math.exp(-sSi30 * (el2-1))
            h_[:, 0] = [el1 for i in range(20)]
            h_[:, el2] = ((max(np2_[:, el2]) - min(np2_[:, el2])) / np.average(np2_[:, el2]))
        h.append(np.average(h_[:, 1:]))
        id += 1
        jd += 1
    return id, jd, np2_, h_, h


def up_down(n_times):
    """
    The driver function.
    :param n_times: Number of times of going up and down of the Si-30 sample.
    :return: P-31 number density, homogeneity
    """
    for n_time in range(0, n_times):
        ith1, jth1, nth1_, hth1_, hh1 = going_up(n_time, ith[-1], jth[-1])
        ith.append(ith1)
        jth.append(jth1)
        ith2, jth2, nth2_, hth2_, hh2 = going_down(n_time, ith[-1], jth[-1])
        ith.append(ith2)
        jth.append(jth2)
    return nth2_, hth2_, hh2


if __name__ == '__main__':
    # get the flux in 1D and SigmaSi30
    f1d, sigma_si30 = get_flux_1d()
    # initialize 2D flux, P-31 number density, homogeneity
    f2d = np.ones((20, 6)) * 1.0
    np2_ = np.zeros((20, 6))
    h_ = np.zeros((20, 6))
    i1, j1, n1, h111 = going_down_first_time(25, 45, f1d, sigma_si30)
    ith = [i1]
    jth = [j1]

    n, h_, h112 = up_down(1000)

    # plot Np at final t of the time interval defined above

    fig = plt.figure(dpi=128, figsize=(10, 10))
    plt.imshow(n[:, 1:])
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.gca().invert_yaxis()
    plt.title(f'$N_{"{P-31}"}$ at t = {n[-1,0]} s')
    plt.xlabel('dv [cm]')
    plt.ylabel('h [cm]')
    plt.colorbar(pad=0.15)
    plt.xticks(ticks=range(5), labels=range(1, 6))
    plt.yticks(ticks=range(20), labels=range(1, 21))
    # save figure
    plt.savefig(f'2dN{n[-1,0]}.png')
    plt.show()

    # plot H at final t of the time interval defined above

    plt.imshow(h_[:, 1:])
    plt.colorbar(pad=0.15)
    plt.gca().invert_yaxis()
    plt.xticks(ticks=range(5), labels=range(1, 6))
    plt.yticks(ticks=range(20), labels=range(1, 21))
    plt.title(f'H at t = {n[-1,0]} s')
    plt.xlabel('dv [cm]')
    plt.ylabel('h [cm]')
    plt.grid(which='major', axis='both')
    # save figure
    plt.savefig(f'2dH{n[-1,0]}.png')
    plt.show()


    # plot H = f(t)

    plt.plot(h112, '-')
    plt.title(f'H = f (t)')
    plt.xlabel('time [s]')
    plt.ylabel('H [/]')
    plt.grid(which='major', axis='both')
    # save figure
    plt.savefig(f'2dHt{n[-1,0]}.png')
    plt.show()
