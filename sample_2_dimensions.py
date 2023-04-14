import math
import matplotlib.pyplot as plt
import numpy as np

from sample_1_dimension import get_flux_1d


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
    np2_ = np.zeros((20, 6))
    h_0 = np.zeros((20, 6))
    h_down = []
    td = t_start()
    y = [i for i in range(1, 6)]
    for el1 in td:
        print(f't={el1} [s], i={ift}, j={jft}')
        for el2 in y:
            np2_[:, 0] = [el1 for i in range(20)]
            np2_[:, el2] = np2_[:, el2] + sigma_si30*np.array(reactor_flux[ift:jft]) * math.exp(-sigmasi30 * el2)
            if el1 != 0:
                h_0[:, 0] = [el1 for i in range(20)]
                # h_0[:, el2] = (np.max(np2_[:, el2]) - np.min(np2_[:, el2])) / np.mean(np2_[:, el2])
        if el1 != 0:
            h_down.append((np.amax(np2_[:, 1:6]) - np.amin(np2_[:, 1:6]))/np.mean(np2_[:, 1:6]))
        ift += 1
        jft += 1
    return ift, jft, np2_, h_down, h_0


def going_up(n_times, ii, jj, reactor_flux, sigmasi30, np_up, h_up):
    """
    A function that calculates from initial time till the Si-30 sample reaches the top from the bottom

    :param n_times: the n-th time of going up
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
    tu = t_up(n_times)
    h_0 = np.zeros((20, 6))
    y = [i for i in range(1, 6)]
    for el1 in tu:
        print(f't={el1} [s], i={ii}, j={jj}')
        for el2 in y:
            np_up[:, 0] = [el1 for i in range(20)]
            np_up[:, el2] = np_up[:, el2] + sigmasi30*np.array(reactor_flux[ii:jj]) * math.exp(-sigmasi30 * el2)
            h_0[:, 0] = [el1 for i in range(20)]
            # h_0[:, el2] = (np.max(np_up[:, el2]) - np.min(np_up[:, el2])) / np.mean(np_up[:, el2])
        h_up.append((np.amax(np_up[:, 1:6]) - np.amin(np_up[:, 1:6])) / np.mean(np_up[:, 1:6]))
        ii -= 1
        jj -= 1
    return ii, jj, np_up, h_up, h_0


def going_down(n_times, i_d, j_d, reactor_flux, sigmasi30, np_down, h_down):
    """
    A function that calculates from initial time till the Si-30 sample reaches the bottom from the top

    :param n_times: the n-th time of going down
    :param i_d: highest index of the Si-30 sample on the initial flux indexes scale
    :param j_d: lowest index of the Si-30 sample on the initial flux indexes scale
    :param reactor_flux: list, the reactor flux
    :param sigmasi30: float, the Sigma(Si-30)
    :param np_down: P-31 number density
    :param h_down: homogeneity
    :return: highest and lowest index of the Si-30 sample on the initial flux indexes scale,
             P-31 number density, homogeneity
    """
    print('\nGoing DOWN\n')
    td = t_down(n_times)
    h_0 = np.zeros((20, 6))
    y = [i for i in range(1, 6)]
    for el1 in td:
        print(f't={el1} [s], i={i_d}, j={j_d}')
        for el2 in y:
            np_down[:, 0] = [el1 for i in range(20)]
            np_down[:, el2] = np_down[:, el2] + sigmasi30*np.array(reactor_flux[i_d:j_d]) * math.exp(-sigmasi30 * el2)
            h_0[:, 0] = [el1 for i in range(20)]
            # h_0[:, el2] = (np.max(np_down[:, el2]) - np.min(np_down[:, el2])) / np.mean(np_down[:, el2])
        h_down.append((np.amax(np_down[:, 1:6]) - np.amin(np_down[:, 1:6])) / np.mean(np_down[:, 1:6]))
        i_d += 1
        j_d += 1
    return i_d, j_d, np_down, h_down, h_0


def up_down(n_times, np31, hom1, reactor_flux, sigmasi30):
    """
    The driver function.

    :param n_times: Number of times of going up and down of the Si-30 sample.
    :return: P-31 number density, homogeneity
    """
    nth2_, hom_3, h_zero2 = None, None, None
    for n_time in range(0, n_times):
        ith1, jth1, nth1_, hom_2, h_zero1 = going_up(n_time, ith[-1], jth[-1], reactor_flux, sigmasi30, np31, hom1)
        ith.append(ith1)
        jth.append(jth1)
        ith2, jth2, nth2_, hom_3, h_zero2 = going_down(n_time, ith[-1], jth[-1], reactor_flux, sigmasi30, nth1_, hom_2)
        ith.append(ith2)
        jth.append(jth2)
    return nth2_, hom_3, h_zero2


if __name__ == '__main__':
    # get the flux in 1D and SigmaSi30
    f, sigma_si30 = get_flux_1d()
    # initialize 2D flux, P-31 number density, homogeneity
    i1, j1, np1_, h1_, hz = going_down_first_time(25, 45, f, sigma_si30)
    ith = [i1]
    jth = [j1]

    n_final, h_final, hz1 = up_down(5, np1_, h1_, reactor_flux=f, sigmasi30=sigma_si30)

    # plot Np at final t of the time interval defined above

    plt.imshow(n_final[:, 1:])
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.title(f'$N_{"{P-31}"}$ at t = {int(n_final[-1,0])} [s]')
    plt.xlabel('dv [cm]')
    plt.ylabel('h [cm]')
    plt.colorbar(pad=0.15)
    plt.xticks(ticks=range(5), labels=range(1, 6))
    plt.yticks(ticks=range(20), labels=range(1, 21))
    # save figure
    plt.savefig(f'images/2dN_t{int(n_final[-1,0])}.png')
    plt.show()

    # plot H = f(t)

    # plot max, min and final H value

    max_val = max(h_final)
    max_pos = h_final.index(max_val)
    min_val = min(h_final[1:])
    min_pos = h_final.index(min_val)
    last_val = h_final[-1]
    last_pos = len(h_final)-1
    plt.scatter(min_pos, min_val, color='blue', label=f"$H_{'{min}'}$ = {round(min_val, 4)} at t = {min_pos} [s]")
    plt.scatter(max_pos, max_val, color='red', label=f"$H_{'{max}'}$ = {round(max_val,4)} at t = {max_pos} [s]")
    plt.scatter(last_pos, last_val, color='green', label=f"$H_{'{final}'}$ = {round(h_final[-1], 4)} at t = {len(h_final)} [s]")
    plt.plot(h_final, '-', color='purple')
    plt.scatter(max_pos, max_val, color='red')
    plt.scatter(min_pos, min_val, color='blue')
    plt.scatter(last_pos, last_val, color='green')
    plt.title(f'H = f (t)')
    plt.xlabel('time [s]')
    plt.ylabel('H [/]')
    plt.legend()
    plt.grid(which='major', axis='both')
    # save figure
    plt.savefig(f'images/2dHt{last_pos+1}.png')
    plt.show()
