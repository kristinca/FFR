import math
import matplotlib.pyplot as plt
import numpy as np

# initialize 2Dflux, P-31 number density, homogeneity
f2d = np.ones((20, 5)) * 1.0
np2_ = np.zeros((20, 2))
h_ = []


def get_flux_1d():
    """
    :return: The reactor flux in 1D.
    """
    y = [i for i in range(-35, 36)]
    # hr = 70
    # Sigma (Si-30) = sigma(Si-30)(n,gamma)*N(Si-30)
    sigmaSi30 = 107.2 * 10 ** (-27)
    # nSi30_ = (roSi30*Na)/MSi30
    nSi30_ = (2.33 * 6.02214 * 10 ** 23) / 29.9737701
    sigmaSi30_ = sigmaSi30 * nSi30_
    print(f'sigmaSi30_ = {sigmaSi30_}, nSi30_= {nSi30_}')
    f0 = 10 ** 13
    f1 = []
    for el1 in y:
        f1.append(f0 * sigmaSi30_ * math.cos(math.radians((math.pi * el1) / 70)))
    # print(f'f 1d ] {f1}')
    return f1


# plt.plot(f)
# plt.show()


def flux_attenuation_2d(ifa, jfa):
    """
    :param ifa: The highest index of the Si-30 sample on the initial flux indexes scale
    :param jfa: lowest index of the Si-30 sample on the initial flux indexes scale
    :return: The flux attenuation in the 2D sample.
    """

    # get the flux in 1D at dv=0
    a1 = get_flux_1d()
    # get flux in 1D in the Si-30 sample
    a = a1[ifa:jfa]

    y = [i for i in range(1, 6)]
    # hr = 70
    # Sigma (Si-30) = sigma(Si-30)(n,gamma)*N(Si-30)
    sigmaSi30 = 107.2 * 10 ** (-27)
    # nSi30_ = (roSi30*Na)/MSi30
    nSi30_ = (2.33 * 6.02214 * 10 ** 23) / 29.9737701
    sigmaSi30_ = sigmaSi30 * nSi30_
    print(f'SigmaSi30_ = {sigmaSi30_}, NSi30_= {nSi30_}')
    for elem in y:
        f2d[:, elem - 1] = np.transpose(a) * math.exp(-sigmaSi30_ * (elem - 1))
    # print(f2d)
    return f2d


if __name__ == '__main__':
    # at t=0
    n = flux_attenuation_2d(25, 45)
    print(n)
    # plot Np31 = f(h)

    # plt.plot(n[:, 1] / 10 ** 13, '-')
    # plt.title(f'Np2 = f (h)')
    # plt.xlabel('h')
    # plt.ylabel('N_p31 [10^13/cm^3]')
    # plt.ticklabel_format(style='plain', useOffset=False, axis='both')
