import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as it
from scipy.integrate import quad


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


def the_integrand(the_time, fint):
    lam = 0.077
    t_int = []
    for i in range(len(the_time)):
        t_int.append(lam*fint[i]*math.exp(-lam*the_time[i]))
    return t_int


def the_delayed_neutrons(the_time, p, integral_the):
    """
    A function that calculates the delayed neutrons contribution to the reactivity.
    :param the_time: the time interval for this case
    :param d_kernel: delayed neutron kernel
    :param p_int: the power ratio inside of the integral
    :param p: power ratio outside of the integral
    :param the_indexes: the indexes for the power ratio
    :return: the value of this part of the equation.
    """
    the_integral = integral_the
    beta = 0.007
    delayed2 = []

    for i in range(len(the_time)):
        try:
            delayed1 = beta - (beta*integral_the[i]/p[i])
        # print(delayed1)
            delayed2.append(100000*delayed1)
        except IndexError:
            break

    # for i in range(1,len(the_indexes)):
    #     try:
    #         the_inti += d_kernel[i]*p_int[i]*(the_time[i+1]-the_time[i])
    #         # print(the_inti)
    #         delayed1 = beta - (beta*the_inti/p[i])
    #         delayed2.append(100000*delayed1)
    #         # print(f'{the_int}, ind = {i}')
    #     except IndexError:
    #         break
    return delayed2


def the_prompt_neutrons(p, t_n):
    """
    A function that calculates the prompt neutrons contribution to the reactivity.
    :param p: power ratio
    :param t_n: time
    :return: the value of this part of the equation.
    """
    llambda = 40*(10**(-6))
    prompt = []
    for i in range(len(t_n)-1):
        prompt.append(100000*llambda*(p[i+1]-p[i])/(p[i]*(t_n[i+1]-t_n[i])))
    return prompt

# def full_integral_part(b, integral, p, t_i):
#     """
#     A function that calculates the full part w/ the integral in the reactivity equation.
#     :param b: effective fraction of delayed neutrons, beta
#     :param integral: the integral
#     :param p: power ratio
#     :param t_i: the time for this i-th case
#     :return: the total part w/ the integral in the reactivity equation.
#     """
#


if __name__ == '__main__':

    # get .txt files with cleaned data
    # for i in range(1, 7):
    #     get_txt_file(i)

    for no in range(1, 7):
        tpp0 = np.loadtxt(f'scenarij{no}.txt', dtype='float')

          # plot and save P(t)/P(0) = f(t) for each case
    #     # plot P(t)/P(0) = f(t)
    #     plt.plot(tpp0[:, 0], tpp0[:, 1])
    #     plt.title(f'Scenarij {no}')
    #     plt.tick_params(axis='both', which='major', labelsize=11)
    #     # plt.yticks(ticks=[i for i in range(0, 60, 5)], labels=[i for i in range(0, 60, 5)])
    #     plt.xlabel('t [s]')
    #     plt.ylabel(f'P(t)/P$_{"0"}$(t)')
    #     plt.grid(which='major', axis='both')
    #
    #     # save figure
    #     plt.savefig(f'p{no}.png')
    #
    #     plt.show()

        tt = tpp0[:, 0]

        # pp = the_prompt_neutrons(tpp0[:,1], tt)

        # # plot the prompt neutrons part in the reactivity equation
        # plt.plot(tt[:-1], pp)
        # plt.subplots_adjust(left=0.17, bottom=0.17)
        # plt.title(f'Prompt neutrons part - scenarij {no}')
        # plt.tick_params(axis='both', which='major', labelsize=11)
        # plt.xlabel('t [s]')
        # plt.ylabel(r"$\rho$ [pcm]")
        # plt.grid(which='major', axis='both')
        #
        #     # save figure
        # plt.savefig(f'prompt{no}.png')
        # plt.show()

        # print(len(pp), len(dd))

        # r = []
        #
        # for i in range(len(tt)-1):
        #     r.append(pp[i]+dd[i])
        #
        # plt.plot(r)
        # plt.show()
        # # print(f'3the delayed neutrons part for scenarij {no} is: {iint}.\n')

        # get the delayed neutron kernel
        d1 = delayed_neutron_kernel(tt)


        # get the indexes for the power ratio in the integral
        aa = []
        for i in range(len(tt)):
            aa.append(int(len(tt)-i))

        # power ratio as it is recorded in time -> to list
        f_list = tpp0[:, 1].tolist()


        # power ratio as needed inside the integral
        f_int = []
        for i in range(1,len(tt)):
            # print(f_list[aa[i]])
            f_int.append(f_list[aa[i]])

        integrand1=the_integrand(tt[1:].tolist(),f_int)
        integral = it.cumtrapz(integrand1, tt[1:].tolist())


        # i = it.cumtrapz(p, ttt)

        # print(f'len d kernel = {len(d1[:-1])}, len f_int = {len(f_int)}, len f_list={len(f_list[:-1])}, len aa={len(aa[:-1])}')
        # dd = the_delayed_neutrons(tt[:-1], d1[:-1], f_int[:-1], f_list[:-1], aa[:-1])

        dd = the_delayed_neutrons(tt[1:], f_list, integral)
        plt.plot(tt[1:-1], dd)
        plt.show()

        # plot D(t) = f(t)
        # plt.plot(d1)
        # plt.title(f'Delayed neutron kernel scenarij {no}')
        # plt.tick_params(axis='both', which='major', labelsize=11)
        # plt.xlabel('time after fission event, u[s]')
        # plt.ylabel(f'Probability of delayed neutron emission within du')
        # plt.grid(which='major', axis='both')
        #
        #     # save figure
        # plt.savefig(f'D(u){no}.png')
        # plt.show()
