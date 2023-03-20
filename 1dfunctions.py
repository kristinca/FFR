import math
import matplotlib.pyplot as plt
import numpy as np


def get_flux_1d():
    """
    :return: The reactor flux in 1D
    """
    # hr = 70 [cm]
    h = 70
    # split the y axis into 70 elements, 1 cm each
    y = [elem for elem in range(-35, 36)]
    # Sigma(Si-30) = sigma(Si-30)(n,gamma)*N(Si-30)
    sigma_si_30_micro = 107.2 * 10**(-27)
    # n_si_30 = (roSi30*Na)/MSi30
    # rho(Si-30) = 2.33 [g/cm^3]
    rho_si_30 = 2.33
    na = 6.022*math.pow(10, 23)
    # M(Si-30) = 29.973770 [g/mol]
    molar_mass_si_30 = 29.973770
    n_si_30 = (rho_si_30*na)/molar_mass_si_30
    sigma_si_30 = sigma_si_30_micro*n_si_30
    print(f'SigmaSi30 = {sigma_si_30}, NSi30= {n_si_30}')
    # initial flux = f0 = 10^13 [1/s*cm^2]
    f0 = math.pow(10, 13)
    flux_1d = [f0*(math.cos((math.pi*element)/70)) for element in y]
    return flux_1d, sigma_si_30


# get the flux in 1D and SigmaSi30
f, sSi30 = get_flux_1d()

# plt.plot(f)
# plt.show()

# initialize P-31 number density, homogeneity
np2_ = np.zeros((20, 2))
h_ = []


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
        np2_[:, 0] = [el1 for i in range(20)]
        np2_[:, 1] = np2_[:, 1] + [sSi30 * el1 * m for m in f[ift:jft]]
        if el1 != 0:
            h_.append((max(np2_[:, 1]) - min(np2_[:, 1])) / np.average(np2_[:, 1]))
        ift += 1
        jft += 1
    return ift, jft, np2_, h_


def going_up(n, ii, jj):
    """
    A function that calculates from initial time till the Si-30 sample reaches the top from the bottom.
    :param n: the n-th time of going  up
    :param ii: highest index of the Si-30 sample on the initial flux indexes scale
    :param jj: lowest index of the Si-30 sample on the initial flux indexes scale
    :return: highest and lowest index of the Si-30 sample on the initial flux indexes scale,
             P-31 number density, homogeneity
    """
    print('\nGoing UP\n')
    tu = t_up(n)
    for el2 in tu:
        print(f't={el2} [s], i={ii}, j={jj}')
        np2_[:, 0] = [el2 for i in range(20)]
        np2_[:, 1] = np2_[:, 1] + [sSi30 * el2 * m for m in f[ii:jj]]
        h_.append((max(np2_[:, 1]) - min(np2_[:, 1])) / np.average(np2_[:, 1]))
        ii -= 1
        jj -= 1
    return ii, jj, np2_, h_


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
        np2_[:, 0] = [el1 for i in range(20)]
        np2_[:, 1] = np2_[:, 1] + [sSi30 * el1 * m for m in f[ii:jj]]
        h_.append((max(np2_[:, 1]) - min(np2_[:, 1])) / np.average(np2_[:, 1]))
        ii += 1
        jj += 1
    return ii, jj, np2_, h_


def up_down(n_times):
    """
    The driver function.
    :param n_times: Number of times of going up and down of the Si-30 sample.
    :return: P-31 number density, homogeneity
    """
    for n in range(0, n_times):
        iii1, ji1, npi1_, hi1_ = going_up(n, iii[-1], jjj[-1])
        iii.append(iii1)
        jjj.append(ji1)
        iii2, ji2, npi2_, hi2_ = going_down(n, iii[-1], jjj[-1])
        iii.append(iii2)
        jjj.append(ji2)
    return npi2_, hi2_


if __name__ == '__main__':
    i1, j1, np1_, h1_ = going_down_first_time(25, 45)
    iii = [i1]
    jjj = [j1]
    # going up and down 5 times
    num = 50
    n, h_ = up_down(num)

    # plot Np31 = f(h)

    plt.plot(n[:, 1]/10**13, '-', color='purple')
    plt.title('$\mathregular{N_{P-31 }}$ = f (h)')
    plt.xlabel('h')
    plt.ylabel('$\mathregular{N_{P-31} [10^{13}/cm^3]}$')
    plt.ticklabel_format(style='plain', useOffset=False, axis='both')
    plt.subplots_adjust(left=0.2)
    # save figure
    # plt.savefig(f'1dN{num}.png')
    plt.show()

    # plot H = f(t)

    plt.plot(h_, '-')
    plt.title(f'H = f (t)')
    plt.xlabel('time [s]')
    plt.ylabel('H [/]')
    plt.grid(which='major', axis='both')
    plt.subplots_adjust(left=0.2)
    # save figure
    # plt.savefig(f'1dH{num}.png')
    plt.show()
