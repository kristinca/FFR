import math
import matplotlib.pyplot as plt
import numpy as np


y = [i for i in range(-35, 36)]
y1 = [i for i in y if i % 5 == 0]

hr = 70
f0 = 10**13
f = []

for el in y:
    f.append(math.cos(math.radians((math.pi*el)/70)))

hv = {i for i in range(20)}

t = [i for i in range(1, 26)]

i = 25
j = 46

Np2 = np.zeros((21, 2))
Np2[:, 1] = np.ones(21)

# H = np.zeros((1, 2))
H = []

# speed = 1 [cm/s]
# here the Si-30 sample reaches the top and stops there

for el in t:
    Np2[:, 0] = [el for i in range(21)]
    Np2[:, 1] = [2*el*m for m in f[i:j]]
    # H[:, 0] = el
    # H[:, 1] = [(max(Np2[:, 1]) - min(Np2[:, 1]))/np.average(Np2[:, 1])]
    H.append((max(Np2[:, 1]) - min(Np2[:, 1]))/np.average(Np2[:, 1]))
    i += 1
    j += 1
    # print(i, j)
    # print(Np2, H)


# print(f[25:46])
print(H)
# plt.plot(H[:, 1], '*')
plt.plot(H, 'o-')
plt.title(f'H = f (t)')
plt.xlabel('time [s]')
plt.ylabel('H [/]')
plt.grid(which='major', axis='both')
plt.show()

# plt.plot(f, y, linewidth='2')
# plt.ylabel('Hr[cm]')
# plt.xlabel('flux [10^13/s*cm^2]')
# # plt.xticks([10**12, 10**13])
# plt.ticklabel_format(style='plain')
# plt.yticks(y1)
# plt.title('Flux = f(x)')
# plt.show()
