"""
This is an excersise from Yandex course 'Математика и Python для анализа данных'

Рассмотрим сложную математическую функцию на отрезке [1, 15]:
f(x) = sin(x / 5) * exp(x / 10) + 5 * exp(-x / 2)
В этом задании мы будем приближать указанную функцию с помощью многочленов.
"""
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def f(x):
    return np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)


def coef_matrix(Xs):
    """Returns a coefficient matrix"""
    A = []
    for x in Xs:
        v = []
        for n in range(len(Xs)):
            v.append(x**n)
        A.append(v)
    return np.array(A)


def val_vector(Xs, func):
    """Returns a vector with values of a function at given point"""
    v = []
    for x in Xs:
        v.append(func(x))
    return v


def solve_coef(Xs, func):
    """Returns the coefficients of the polynomial, approximated
    at Xs to function func"""
    A = coef_matrix(Xs)  # np.array([[1, 1], [1, 15]])
    b = val_vector(Xs, f)  # np.array([f(1), f(15)])
    return scipy.linalg.solve(A, b)  # solve for coefficients


def poly(w, x):
    """Returns the values of the approximation function"""
    y = w[0] * np.power(x, 0)
    for n in range(1, len(w)):
        y += w[n] * np.power(x, n)
    return y


def func_approximate(func, Xs, x):
    """Approximate values of the funtion with len(Xs) degree polynomial"""
    return poly(solve_coef(Xs, func), x)


if __name__ == '__main__':

    x = np.linspace(0, 15, 50)
    y = f(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, 'g--')
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 6])

    # 1st degree polynomial
    Xs = [1., 15.]  # approximate polynomial with these X

    A = coef_matrix(Xs)  # np.array([[1, 1], [1, 15]])
    b = val_vector(Xs, f)  # np.array([f(1), f(15)])
    w = scipy.linalg.solve(A, b)  # solve for coefficients

    y = w[0] + w[1] * x

    ax.plot(x, y, 'r--')

    # 2nd degree polynomial
    y = func_approximate(f, [1., 8., 15.], x)

    ax.plot(x, y, 'y--')

    # 3rd degree polynomial
    y = func_approximate(f, [1., 4., 10., 15.], x)

    ax.plot(x, y, 'b--')

    ax.legend(['original', '1d polynomial', '2d polynomial', '3d polynomial'], loc=2)
    plt.show()
