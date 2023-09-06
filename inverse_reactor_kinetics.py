import math
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit


def clean_data(file_name):
    """
    A function that cleans the data from a .out file.
    :param file_name: name of the .out file to be cleaned
    :return: t : list, the time from the .out file; pp0 : list, P(t)/P(0) from the .out file
    """
    df = pd.read_csv(f'{file_name}.out', delimiter="\t")
    t = []
    pp0 = []
    for row in df.iterrows():
        ii = re.sub(r'^(\D*)\d', '', str(row[1]))
        iii = re.sub(' +', ' ', ii).split(' ')
        iiii = re.sub('\nName:$', '', iii[3])
        t.append(iii[2])
        # print(f't = {iii[2]}', end=',\t')
        pp0.append(iiii)
        # print(f'P(t)/P(0) = {iiii}', end='\n')
    return t, pp0


def get_txt_file(num):
    """
    A function to write cleaned .txt file from the .out file
    :param num: number of Scenarij.out file
    """
    # get the time and P(t)/P(0)
    tnum, pp0num = clean_data(f'Scenarij_{num}')

    # write the time and P(t)/P(0) from the .out file to a txt file
    with open(f'scenarij{num}.txt', 'w') as f:
        for row in range(len(tnum)):
            f.write(tnum[row] + ' ' + pp0num[row]+'\n')


def get_time_and_power_rate(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        t = []
        pp0 = []
        for line in lines:
            t.append(line.split(' ')[0])
            pp0.append(line.split(' ')[1])
    return t, pp0


def delayed_neutron_kernel(t_end):
    """
    A function that calculates the delayed neutron kernel.
    :param t_end: total time
    :return: delayed neutron kernel function for this problem
    """
    # D(u) = lambda*exp(-lambda*u)
    lam = 0.077
    d = []
    for i in t_end:
        d.append(lam*math.exp(-lam*i))
    return d


def inhour_eq(reactor1_period):
    """
    The inhour equation.
    :param reactor1_period: reactor period
    :return: reactivity.
    """
    llambda = 40*(10**(-6))
    beta = 0.007
    lam = 0.077
    ro1 = []
    for el in reactor1_period:
        ro11 = (llambda/el)+(beta/(1+lam*el))
        ro1.append(ro11)
    return ro1


@njit(parallel=True)
def the_delayed_neutrons(the_time, p, delayedn2):
    lam = 0.077
    for t in range(len(the_time)):
        print(t)
        int1 = 0.0
        for u in range(t):
            int1 += lam * np.exp(-lam * (the_time[t] - the_time[u])) * (p[u] / p[t]) * (the_time[u + 1] - the_time[u])
        delayed1 = 1 - int1
        delayedn2[t] = delayed1
    return delayedn2


def the_prompt_neutrons(p, t_n):
    """
    A function that calculates the prompt neutrons contribution to the reactivity.
    :param p: power ratio
    :param t_n: time
    :return: the value of this part of the equation.
    """
    llambda = float(40e-6)
    prompt = []
    for i in range(1, len(t_n.keys())-1):
        prompt.append((llambda*(p[i+1]-p[i-1]))/(p[i]*0.007*2*(t_n[i+1]-t_n[i])))
    return prompt


if __name__ == '__main__':

    for no in range(1, 7):
        st = time.perf_counter()
        # get the txt file
        # get_txt_file(no)

        # get numpy array with cleaned data -> time, power ratio
        tpp0 = np.loadtxt(f'scenarij{no}.txt', dtype='float')
        t_dict = {index: element for index, element in enumerate(tpp0[:, 0])}
        pp0_dict = {index1: element1 for index1, element1 in enumerate(tpp0[:, 1])}



    # #   1. plot and save P(t)/P(0) = f(t) for each case
    #     plt.plot(tpp0[:, 0], tpp0[:, 1])
    #     plt.title(f'Scenarij {no}')
    #     plt.tick_params(axis='both', which='major', labelsize=11)
    #     plt.xlabel('t [s]')
    #     plt.ylabel(f'P(t)/P(0)(t)')
    #     plt.grid(which='major', axis='both')
    #     # plot P(t)/P(0) = f(t) at max, min and last P(t)/P(0) value
    #     max_val = np.max(tpp0[:, 1])
    #     max_pos = tpp0[:, 0][np.argmax(tpp0[:, 1])]
    #     min_val = np.min(tpp0[:, 1])
    #     min_pos = tpp0[:, 0][np.argmin(tpp0[:, 1])]
    #     last_val = tpp0[:, 1][-1]
    #     last_pos = tpp0[:, 0][-1]
    #     plt.scatter(min_pos, min_val, color='orange',
    #                 label=f"$P(t)/P(0)_{'{min}'}$ = {min_val} at t = {min_pos} [s]")
    #     plt.scatter(max_pos, max_val, color='red', marker='s',
    #                 label=f"$P(t)/P(0)_{'{max}'}$ = {round(max_val, 4)} at t = {max_pos} [s]")
    #     plt.scatter(last_pos, last_val, color='green', marker='^',
    #                 label=f"$P(t)/P(0)_{'{final}'}$ = {round(last_val, 4)} at t = {last_pos} [s]")
    #     plt.legend()
    #     plt.yscale('log')
    #     plt.xscale('log')
    #   1.1 save figure
    #     plt.savefig(f'images2/logp{no}.png')
    #     plt.show()

    # # 1.2. plot reactor period  T = t / ln(P(t)/P0)
    #     lnpp0 = np.log(tpp0[:, 1])
    #     reactor_period = [tpp0[:, 0][elem]/lnpp0[elem] for elem in range(1, len(tpp0[:, 0]))]
    #     plt.plot(tpp0[1:, 0], reactor_period)
    #     plt.tick_params(axis='both', which='major', labelsize=11)
    #     plt.title(f'Scenarij {no}')
    #     plt.xlabel('t [s]')
    #     plt.ylabel(f'T [s]')
    #     plt.grid(which='major', axis='both')
    #     # T = f(t) at max, min and last T = f(t) value
    #     max_val = max(reactor_period)
    #     max_pos = tpp0[1:, 0][reactor_period.index(max_val)]
    #     min_val = min(reactor_period)
    #     min_pos = tpp0[1:, 0][reactor_period.index(min_val)]
    #     last_val = reactor_period[-1]
    #     last_pos = tpp0[1:, 0][-1]
    #     plt.scatter(min_pos, min_val, color='orange',
    #                 label=f"$T_{'{min}'}$ = {round(min_val, 4)} [s] at t = {min_pos} [s]")
    #     plt.scatter(max_pos, max_val, color='red', marker='s',
    #                 label=f"$T_{'{max}'}$ = {round(max_val, 4)} [s] at t = {max_pos} [s]")
    #     plt.scatter(last_pos, last_val, color='green', marker='^',
    #                 label=f"$T_{'{final}'}$ = {round(last_val, 4)} [s] at t = {last_pos} [s]")
    #     plt.legend()
    #     # plt.xscale('log')
    #     # plt.yscale('log')
    #     #   1.1 save figure
    #     # plt.savefig(f'images2/reactor_period/T{no}.png')
    #     plt.show()

    # 2. prompt neutrons part in the reactivity equation
        tt = tpp0[:, 0]
        pp = the_prompt_neutrons(pp0_dict, t_dict)
        #
        # # 2.1. plot the prompt neutrons part in the reactivity equation
        # plt.plot(tt[1:-1], pp)
        # plt.title(f'Prompt neutrons part - scenarij {no}')
        # plt.tick_params(axis='both', which='major', labelsize=11)
        # plt.xlabel('t [s]')
        # plt.ylabel(r"$\rho [\$]$")
        # plt.grid(which='major', axis='both')
        # #
        # # rho = f(t) at max, min and last rho = f(t) value
        # max_val = max(pp)
        # max_pos = tpp0[:, 0][pp.index(max_val)]
        # min_val = min(pp)
        # min_pos = tpp0[:, 0][pp.index(min_val)]
        # last_val = pp[-1]
        # last_pos = tpp0[:, 0][-1]
        # plt.scatter(min_pos, min_val, color='orange',
        #             label=fr"$\rho_{'{min}'}$ = {round(min_val, 4)} [\$] at t = {min_pos} [s]")
        # plt.scatter(max_pos, max_val, color='red', marker='s',
        #             label=fr"$\rho_{'{max}'}$ = {round(max_val, 4)} [\$] at t = {max_pos} [s]")
        # plt.scatter(last_pos, last_val, color='green', marker='^',
        #             label=fr"$\rho_{'{final}'}$ = {round(last_val, 4)} [\$] at t = {last_pos} [s]")
        # plt.legend()
        # # plt.xscale('log')
        # # plt.yscale('log')
        # # 2.2 save figure
        # # plt.savefig(f'prompt{no}.png')
        # plt.show()
        # exit(0)

        # 3. delayed neutrons part in the equation

        # 3.1. get the delayed neutron kernel
        # d1 = delayed_neutron_kernel(tt)


        # # plot D(t) = f(t)
        # plt.plot(d1)
        # plt.title(f'Delayed neutron kernel scenarij {no}')
        # plt.tick_params(axis='both', which='major', labelsize=11)
        # plt.xlabel('time after fission event, u[s]')
        # plt.ylabel(f'Probability of delayed neutron emission within du')
        # plt.grid(which='major', axis='both')
        # # save figure
        # # plt.savefig(f'D(u){no}.png')
        # plt.show()
        # d1 = two_for(pp0_dict, t_dict)
        # np.savetxt('rho12.txt', d1)
        # exit(0)

        # d1 = the_delayed_neutrons(tpp0[:, 0], tpp0[:, 1])

        delayed2 = np.zeros(len(tpp0[:, 0]))
        d1 = the_delayed_neutrons(tpp0[:, 0], tpp0[:, 1], delayed2)

        beta = 0.007
        r_final = []
        for el in range(len(d1)):
            try:
                r_final.append(pp[el]+d1[el])
            except IndexError:
                r_final.append(d1[el])

        end = time.perf_counter()
        print(f'scenarij {no} reactivity calculated in {round(end-st, 4)} seconds.')

        plt.plot(tpp0[:, 0][:len(r_final)], r_final, color='magenta')
        plt.title(f'Reactivity scenarij {no}')
        plt.tick_params(axis='both', which='major', labelsize=11)
        plt.xlabel('time [s]')
        plt.ylabel(r"$\rho [\$]$")
        plt.grid(which='major', axis='both')

        # rho = f(t) at max, min and last rho = f(t) value
        max_val = max(r_final)
        max_pos = tpp0[:, 0][r_final.index(max_val)]
        min_val = min(r_final)
        min_pos = tpp0[:, 0][r_final.index(min_val)]
        last_val = r_final[-1]
        last_pos = tpp0[:, 0][-1]
        plt.scatter(min_pos, min_val, color='blue',
                    label=fr"$\rho_{'{min}'}$ = {round(min_val, 4)} [\$] at t = {min_pos} [s]")
        plt.scatter(max_pos, max_val, color='red', marker='s',
                    label=fr"$\rho_{'{max}'}$ = {round(max_val, 4)} [\$] at t = {max_pos} [s]")
        plt.scatter(last_pos, last_val, color='green', marker='^',
                    label=fr"$\rho_{'{final}'}$ = {round(last_val, 4)} [\$] at t = {last_pos} [s]")
        plt.legend()
        # plt.xscale('log')
        # plt.yscale('log')
        # save figure
        plt.savefig(f'rho{no}.png')
        plt.show()

        # # 3.2 get the indexes for the power ratio in the integral
        # indexes = []
        # for i in range(len(tt)):
        #     indexes.append(int(len(tt) - i))
        #
        # # 3.3 power ratio as it is recorded in time -> to list
        # f_list = tpp0[:, 1].tolist()
        #
        # # 3.4 power ratio as needed in the integral
        # f_int = []
        # for i in range(1,len(tt)):
        #     f_int.append(f_list[indexes[i]])
        #
        # # 3.5 the integral
        # integrand1 = the_integrand(tt[1:].tolist(), f_int)
        # integral = it.cumtrapz(integrand1, tt[1:].tolist(), dx=0.002)
        #
        # # 3.6. the delayed neutrons part
        # dd = the_delayed_neutrons(tt[1:], tpp0[:,1])
        # plt.plot(tt[:-1], dd)
        # plt.show()
        #
        # # 4. plot prompt + delayed neutrons reactivity
        # r = []
        # for i in range(len(dd)):
        #     # if the sum of prompt + delayed neutrons part is less than 1 beta
        #     if abs(pp[i]+dd[i]) < 0.007:
        #         r.append((pp[i]+dd[i])*10**5)
        #     else:
        #         # the prompt part only
        #         r.append(pp[i]*10**5)
        # plt.plot(tt[:-1], r, color='#FF00FF')
        # plt.title(f'Reactivity scenarij {no}')
        # plt.tick_params(axis='both', which='major', labelsize=11)
        # plt.subplots_adjust(left=0.17, bottom=0.17)
        # plt.xlabel('t [s]')
        # plt.ylabel(r"$\rho [pcm]$")
        # plt.grid(which='major', axis='both')
        #
        # 4.1. save figure
        # plt.savefig(f'rho{no}.png')
        # plt.show()
        #
        # 5. multiplication factor k = 1/(1 - rho)
        k = [1/(1-i*beta) for i in r_final]

        plt.plot(tt, k, color='#800020')
        plt.tick_params(axis='both', which='major', labelsize=11)
        plt.subplots_adjust(left=0.17, bottom=0.17)
        plt.xlabel('t [s]')
        plt.ylabel('k')
        plt.grid(which='major', axis='both')

        # rho = f(t) at max, min and last rho = f(t) value
        max_val = max(k)
        max_pos = tpp0[:, 0][k.index(max_val)]
        min_val = min(k)
        min_pos = tpp0[:, 0][k.index(min_val)]
        last_val = k[-1]
        last_pos = tpp0[:, 0][-1]
        plt.scatter(min_pos, min_val, color='blue',
                    label=fr"$k{'{min}'}$ = {round(min_val, 4)} [\$] at t = {min_pos} [s]")
        plt.scatter(max_pos, max_val, color='red', marker='s',
                    label=fr"$k{'{max}'}$ = {round(max_val, 4)} [\$] at t = {max_pos} [s]")
        plt.scatter(last_pos, last_val, color='green', marker='^',
                    label=fr"$k{'{final}'}$ = {round(last_val, 4)} [\$] at t = {last_pos} [s]")
        plt.legend()

        # # 5.1. save figure
        plt.savefig(f'k{no}.png')
        plt.show()