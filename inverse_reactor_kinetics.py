import re
import pandas as pd


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


# get the time and P(t)/P(0)
t1, pp01 = clean_data('Scenarij_1')

# write the time and P(t)/P(0) from the .out file to a txt file
with open('scenarij1.txt', 'w') as f:
    for row in range(len(t1)):
        f.write(t1[row] + ' ' + pp01[row]+'\n')
