import numpy as np
import matplotlib.pyplot as plt

def func_of_system(u, v, A, B):
    return np.array([A + u**2 * v - (B + 1) * v, B * u - u**2 * v])

def runge_cutte_1_order(grid_step, A, B, edges):
    grid = np.arange(edges[0], edges[1], grid_step)
    y = np.ndarray((int(edges[1] / grid_step), 2), dtype = float)
    y[0, :] = 1
    for i in np.arange(edges[0], round(edges[1] / grid_step) - 1):
        k1 = func_of_system(y[i, 0], y[i, 1], A, B)
        y[i + 1, :] = y[i, :] + grid_step * k1
    return y

def runge_cutte_4_order(grid_step, A, B, edges):
    grid = np.arange(edges[0], edges[1], grid_step)
    y = np.ndarray((int(edges[1] / grid_step), 2), dtype = float)
    k2 = np.ndarray((2), dtype = float)
    k3 = np.ndarray((2), dtype = float)
    k4 = np.ndarray((2), dtype = float)
    y[0, :] = 1
    for i in np.arange(edges[0], round(edges[1] / grid_step) - 1):
        k1 = func_of_system(y[i, 0], y[i, 1], A, B)
        k2[:] = func_of_system(y[i, 0] + grid_step / 2 * k1[0], y[i, 1] + grid_step / 2 * k1[1], A, B)
        k3[:] = func_of_system(y[i, 0] + grid_step / 2 * k2[0], y[i, 1] + grid_step / 2 * k2[1], A, B)
        k4[:] = func_of_system(y[i, 0] + grid_step * k3[0], y[i, 1] + grid_step * k3[1], A, B)
        y[i + 1, :] = y[i, :] + grid_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y

grid_step_array = [0.001, 0.01, 0.05, 0.1]
edges = (0, 10)
plt.rcParams["figure.figsize"] = [12, 8]
for grid_step in grid_step_array:
    y = runge_cutte_1_order(grid_step, 1, 2, edges)
    y_4_runge = runge_cutte_4_order(grid_step, 1, 2, edges)
    fig, axs = plt.subplots(nrows= 4 , ncols= 1)
    axs[0].plot(np.arange(edges[0], edges[1], grid_step), y)
    axs[1].plot(np.arange(edges[0], edges[1], grid_step), y_4_runge)
    axs[3].plot(np.arange(edges[0], edges[1], grid_step), np.concatenate((y, y_4_runge), axis=1))
    axs[2].plot(y[:,0], y[:,1])
    axs[0].set_title("1 order runge-kutte")
    axs[1].set_title("4 order runge-kutte")
    axs[3].set_title("1 and 4 order runge-kutte together")
    axs[2].set_title("phase trajectory")
    axs[0].legend(("u", "v"))
    axs[1].legend(("u", "v"))
    axs[3].legend(("u_1", "v_1", "u_4", "v_4"))
    fig.suptitle("grid_step = " + str(grid_step))
    fig.tight_layout()
    fig.savefig("runge" + str(grid_step) + ".png")

B_set =[1.1, 2, 3, 4, 4.9]
fig, axs = plt.subplots(nrows= 5 , ncols= 1 )
for i in range(5):
    y = runge_cutte_4_order(0.01, 1, B_set[i], edges)
    axs[i].plot(np.arange(edges[0], edges[1], 0.01), y)
    axs[i].set_title("4 order runge-kutte for B = " + str(B_set[i]))
    axs[i].legend(("u", "v"))
fig.tight_layout()
fig.savefig("different_B.png")