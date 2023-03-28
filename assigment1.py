import numpy as np
import matplotlib.pyplot as plt

def f(i, j):
    return 1

def f1(x, y):
    e = 0.1
    if 0 <= x and x <= 0.5 and 2 * x - e <= y and y <= 2 * x + e:
        return 1
    if 0.5 <= x and x <= 1 and 2 - 2 * x - e <= y and y <= 2 - 2 * x + e:
        return 1
    return 0
        
def slae(h, hx, hy, func): #system of linear algebraic equations
    scale = (int(1 / h) - 1)
    a = np.zeros((scale ** 2, scale ** 2), dtype=float)
    b = np.zeros(scale ** 2, dtype=float)
    for i in range(0, scale ** 2):
        a[i, i] = 2 * (hx ** 2 + hy ** 2)
        if i % scale > 0:
            a[i, i - 1] = -1 * hx ** 2
        if i % scale < (int(1 / h) - 2):
            a[i, i + 1] = -1 * hx ** 2
        if i >= scale:
            a[i, i - int(1 / h) + 1] = -1 * hy ** 2
        if i < scale ** 2 - scale:
            a[i, i + int(1 / h) - 1] = -1 * hy ** 2
        b[i] = func((float(i // (int(1 / hx) - 1) + 1)/ scale), float(i % (int(1 / hy) - 1) + 1) / scale) * (hx ** 2) * (hy ** 2)
    return (a, b)



#task_1
h = 0.25
a = slae(h, h, h * 2, f)
#plt.spy(a[0], markersize = 192/(int(1 / h) - 1) ** 2)
#print(a[0])
#print(np.linalg.solve(a[0],a[1]))
#plt.show()

#task_2
h = 1 / 20
d = slae(h, h, h / 1.1, f1)
print(np.linalg.norm(d[0] - np.transpose(d[0])))
plt.spy(d[0], markersize = 192/(int(1 / h) - 1) ** 2)
image = np.zeros((int(1 / h) + 1, (int(1 / h) + 1)))
image[1:int(1 / h), 1:int(1 / h)] = np.transpose(np.reshape(np.linalg.solve(d[0],d[1]), (int(1 / h) - 1, int(1 / h) - 1)))
fig2, axs2 = plt.subplots(1, 1)
cs = axs2.contourf (np.arange (0, 1, 1/(int(1 / h) + 1)), np.arange (0, 1, 1/(int(1 / h) + 1)), image)
cbar = fig2.colorbar (cs)
plt.show()