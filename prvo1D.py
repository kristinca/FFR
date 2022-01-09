import math
import matplotlib.pyplot as plt


y = [i for i in range(-35, 36)]
y1 = [i for i in y if i % 5 == 0]

hr = 70
f0 = 10**13
f = []

for el in y:
    f.append(math.cos(math.radians((math.pi*el)/70)))

hv = {i for i in range(20)}

keys = [i*0 for i in hv]
Np = dict.fromkeys(hv)
# print(Np)
for el in Np.keys():
    Np[el] = 0

t = [i for i in range(20)]



# plt.plot(f, y, linewidth='2')
# plt.ylabel('Hr[cm]')
# plt.xlabel('flux [10^13/s*cm^2]')
# # plt.xticks([10**12, 10**13])
# plt.ticklabel_format(style='plain')
# plt.yticks(y1)
# plt.title('Flux = f(x)')
# plt.show()
