import numpy as np
import matplotlib.pyplot as plt

def f(x):
    if x >= 0.2 and x <= 0.3:
        return np.sin((x - 0.2) * np.pi / 0.1)
    elif x >= 0.6 and x <= 0.7:
        return 0.5 * np.sin((x - 0.6) * np.pi / 0.1)
    elif x >= 0 and x <= 1:
        return 0
    else:
        print("wrong input")

def Fn(u_n_iter, n, h):
    res = np.ndarray((n - 1))
    for j in range(1, n):
        res[j - 1] = -(u_n_iter[j + 1] - 2 * u_n_iter[j] + u_n_iter[j - 1]) / (h ** 2) + u_n_iter[j] ** 2 - f(j * h)
    return res

def derivative_matrix(u_n_iter, h, n):
    A = np.full((n - 1, n - 1), 0, dtype=np.float64)
    A[0][0] = 2 / (h ** 2) + 2 * u_n_iter[1]
    A[0][1] = - 1 / (h ** 2)
    A[n - 2][n - 2] = 2 / (h ** 2) + 2 * u_n_iter[n - 1]
    A[n - 2][n - 3] = - 1 / (h ** 2)
    for i in range(0, n - 2):
        for j in range(0, n - 1):
            if j < i - 1:
                A[i][j] = 0
            if j == i - 1:
                A[i][j] = - 1 / (h ** 2)
            if j == i:
                A[i][j] = 2 / (h ** 2) + 2 * u_n_iter[i + 1]
            if j == i + 1:
                A[i][j] = -1 / (h ** 2)
            if j > i + 1:
                A[i][j] = 0
    return A

def solve_equation(h, number_of_iterations, epsilon=None):
    n = int(1 / h) + 1
    u_n = np.full(n + 1, 0, dtype=np.float64)
    u_n[0] = 0
    u_n[n] = 0
    u_n_new = u_n
    u_iterations = np.zeros((number_of_iterations + 1, n + 1), dtype=np.float64)
    u_iterations[0] = u_n
    if epsilon == None:
        for i in range(0, number_of_iterations):
            u_n_new[1:-1] = u_n[1:-1] - np.linalg.inv(derivative_matrix(u_n, h, n)) @ Fn(u_n, n, h)
            #print(np.unique(np.linalg.inv(derivative_matrix(u_n, h, n)) @ Fn(u_n, n, h)))
            u_n = u_n_new
            u_iterations[i + 1] = u_n
    
    elif epsilon != None:
        u_iterations = np.zeros((1000 * number_of_iterations + 1, n + 1), dtype=np.float64)
        u_iterations[0] = u_n
        i = 0
        delta = np.linalg.norm(Fn(u_n, n, h))
        print(delta, np.linalg.norm(u_iterations[i]))
        while np.abs(delta) > epsilon:
            u_n_new[1:-1] = u_n[1:-1] - np.linalg.inv(derivative_matrix(u_n, h, n)) @ Fn(u_n, n, h)
            #print(np.unique(np.linalg.inv(derivative_matrix(u_n, h, n)) @ Fn(u_n, n, h)))
            u_n = u_n_new
            
            delta = np.linalg.norm(Fn(u_n, n, h))
            i = i + 1
            u_iterations[i] = u_n
            
            print(delta, np.linalg.norm(u_iterations[i]))

    return (u_n, u_iterations)


def solution_plot(n, y):
    fig, ax = plt.subplots(1, 1)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05*np.amin(y), 1.05*np.amax(y))
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u', fontsize=12)

    ax.grid(visible=True, which='major')
    ax.grid(visible=True, which='minor',c="#DDDDDD")
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.minorticks_on()

    ax.plot (np.linspace (0, 1, n + 1), y)
    plt.show ()

def convergence_plot(u_iterations, n, number_of_iterations):
    i = 0
    y = list()
    for i in range(0, number_of_iterations):
        print(np.linalg.norm(u_iterations[i]))
        y.append(np.linalg.norm(u_iterations[i + 1] - u_iterations[i]))
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("number of iteration", fontsize=12)
    ax.set_ylabel("|u[i+1] - u[i]|", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.minorticks_on()
    ax.grid(visible=True, which='major')
    ax.grid(visible=True, which='minor',c="#DDDDDD")
    ax.plot(range(0, number_of_iterations), y)
    plt.show()

h = 0.01
number_of_iterations = 10
y, u_iterations = solve_equation(h, number_of_iterations)
n = int(1 / h) + 1
solution_plot(n, y)
convergence_plot(u_iterations, n, number_of_iterations)