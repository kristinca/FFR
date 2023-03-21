import math
import matplotlib.pyplot as plt
import numpy as np


def get_flux_1d() -> tuple:
    """
    Function that calculates the reactor flux in 1D

    :return: The reactor flux in 1D, Sigma(Si-30)
    """
    # hr = 70 [cm]
    h = 70
    # split the y axis into 70 elements, 1 cm each
    y = [elem for elem in range(-35, 36)]
    # Sigma(Si-30) = sigma(Si-30)(n,gamma)*N(Si-30)
    sigma_si_30_micro = 107.2 * 10 ** (-27)
    # n_si_30 = (roSi30*Na)/MSi30
    # rho(Si-30) = 2.33 [g/cm^3]
    rho_si_30 = 2.33
    na = 6.022 * math.pow(10, 23)
    # M(Si-30) = 29.973770 [g/mol]
    molar_mass_si_30 = 29.973770
    n_si_30 = (rho_si_30 * na) / molar_mass_si_30
    sigma_si_30 = sigma_si_30_micro * n_si_30
    print(f'SigmaSi30 = {sigma_si_30}, NSi30= {n_si_30}')
    # initial flux = f0 = 10^13 [1/s*cm^2]
    f0 = math.pow(10, 13)
    flux_1d = [f0 * (math.cos((math.pi * element) / 70)) for element in y]
    return flux_1d, sigma_si_30


def t_up(
        n_up: int
) -> list:
    """
    Function that returns list with the time interval when the sample is going up

    :param n_up: n-th number of time the sample goes up
    :return: list - the time interval needed for the sample to reach the top with dt=1[s], v=1[cm/s]
    """
    return [tup for tup in range(25 + 100 * n_up, 75 + 100 * n_up)]


def t_down(
        n_down: int
) -> list:
    """
    Function that returns list with the time interval when the sample is going down

    :param n_down: n-th number of time the sample goes down
    :return: list - the time interval needed for the sample to reach the bottom with dt=1[s], v=1[cm/s]
    """
    return [tdd for tdd in range(75 + 100 * n_down, 125 + 100 * n_down)]


def t_start() -> list:
    """
    Function that returns list with the time interval when the sample is going down for the first time

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
    for el1 in td:
        print(f't={el1} [s], i={ift}, j={jft}')
        np2_[:, 0] = [el1 for i in range(20)]
        np2_[:, 1] = np2_[:, 1] + [sigmasi30 * el1 * m for m in reactor_flux[ift:jft]]
        if el1 != 0:
            h_0.append((max(np2_[:, 1]) - min(np2_[:, 1])) / np.average(np2_[:, 1]))
        ift += 1
        jft += 1
    return ift, jft, np2_, h_0


def going_up(n, ii, jj, reactor_flux, sigmasi30, np_up, h_up):
    """
    A function that calculates from initial time till the Si-30 sample reaches the top from the bottom.
    :param n: the n-th time of going  up
    :param ii: highest index of the Si-30 sample on the initial flux indexes scale
    :param jj: lowest index of the Si-30 sample on the initial flux indexes scale
    :param reactor_flux: list, the reactor flux
    :param sigmasi30: float, the Sigma(Si-30)
    :param np_up: P-31 number density
    :param h_up: homogeneity
    :return: highest and lowest index of the Si-30 sample on the initial flux indexes scale,
             P-31 number density, homogeneity
    """
    print('\nGoing UP\n')
    tu = t_up(n)
    for el2 in tu:
        print(f't={el2} [s], i={ii}, j={jj}')
        np_up[:, 0] = [el2 for i in range(20)]
        np_up[:, 1] = np_up[:, 1] + [sigmasi30 * el2 * m for m in reactor_flux[ii:jj]]
        h_up.append((max(np_up[:, 1]) - min(np_up[:, 1])) / np.average(np_up[:, 1]))
        ii -= 1
        jj -= 1
    return ii, jj, np_up, h_up


def going_down(n, ii, jj, reactor_flux, sigmasi30, np_down, h_down):
    """
    A function that calculates from initial time till the Si-30 sample reaches the bottom from the top.
    :param n: the n-th time of going down
    :param ii: highest index of the Si-30 sample on the initial flux indexes scale
    :param jj: lowest index of the Si-30 sample on the initial flux indexes scale
    :param reactor_flux: list, the reactor flux
    :param sigmasi30: float, the Sigma(Si-30)
    :param np_down: P-31 number density
    :param h_down: homogeneity
    :return: highest and lowest index of the Si-30 sample on the initial flux indexes scale,
             P-31 number density, homogeneity
    """
    print('\nGoing DOWN\n')
    td = t_down(n)
    for el1 in td:
        print(f't={el1} [s], i={ii}, j={jj}')
        np_down[:, 0] = [el1 for i in range(20)]
        np_down[:, 1] = np_down[:, 1] + [sigmasi30 * el1 * m for m in reactor_flux[ii:jj]]
        h_down.append((max(np_down[:, 1]) - min(np_down[:, 1])) / np.average(np2_[:, 1]))
        ii += 1
        jj += 1
    return ii, jj, np_down, h_down


def up_down(n_times, np31, hom1, reactor_flux, sigmasi30):
    """
    The driver function.

    :param n_times: int, number of times of going up and down of the Si-30 sample
    :param reactor_flux: list, the reactor flux
    :param sigmasi30: float, the Sigma(Si-30)
    :return: P-31 number density, homogeneity
    """
    np32 = []
    for n in range(0, n_times):
        iii1, ji1, npi1_, hi1_ = going_up(n, iii[-1], jjj[-1], reactor_flux, sigmasi30, np31, hom1)
        iii.append(iii1)
        jjj.append(ji1)
        iii2, ji2, np32, hom1 = going_down(n, iii[-1], jjj[-1], reactor_flux, sigmasi30, npi1_, hi1_)
        iii.append(iii2)
        jjj.append(ji2)
    return np32, hom1


if __name__ == '__main__':
    # get the flux in 1D and SigmaSi30
    f, sigma_si30 = get_flux_1d()
    # initialize P-31 number density, homogeneity
    np2_ = np.zeros((20, 2))
    i1, j1, np1_, h1_ = going_down_first_time(25, 45, f, sigma_si30)
    iii = [i1]
    jjj = [j1]
    # going up and down 5 times
    num = 50
    np31, hom = up_down(num, np1_, h1_, reactor_flux=f, sigmasi30=sigma_si30)

    # plot Np31 = f(h)

    plt.plot(np31[:, 1] / 10 ** 13, '-', color='purple')
    plt.title('$\mathregular{N_{P-31 }}$ = f (h)')
    plt.xlabel('h')
    plt.ylabel('$\mathregular{N_{P-31} [10^{13}/cm^3]}$')
    plt.ticklabel_format(style='plain', useOffset=False, axis='both')
    plt.subplots_adjust(left=0.2)
    # save figure
    # plt.savefig(f'1dN{num}.png')
    plt.show()

    # plot H = f(t)

    plt.plot(hom, '-')
    plt.title(f'H = f (t)')
    plt.xlabel('time [s]')
    plt.ylabel('H [/]')
    plt.grid(which='major', axis='both')
    plt.subplots_adjust(left=0.2)
    # save figure
    # plt.savefig(f'1dH{num}.png')
    plt.show()
