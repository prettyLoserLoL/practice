import numpy as np
import numpy.ma
from scipy.integrate import odeint
from matplotlib import pyplot as pl
import pandas as pd
import decimal


def float_range(start, stop, step):
    while start < stop:
        yield float(start)
        start += decimal.Decimal(step)


alf1 = 0.65
alf2 = 0.71
bet1 = 0.13
bet2 = 0.34
T1 = 0.9
T2 = 1
x2const = 2

def f_calc(t, x):
    x1, x2 = x
    return [alf1 * x1 - bet1 * x1 * x2, -alf2 * x2 + bet2 * x1 * x2]


def u_calc(t, x):
    x1, x2 = x
    try:
        fi1 = (alf2 * x2 - (1 / T2) * (x2 - x2const)) / (bet2 * x2)
        f1 = alf1 * x1 - bet1 * x1 * x2
        f2 = - alf2 * x2 + bet2 * x1 * x2
        fi2 = (x2const / (T2 * bet2 * x2 * x2)) * f2
        e = - 1 / T1 * (x1 - fi1) - f1 + fi2
    except ZeroDivisionError:
        return [0, 0]
    return [e, 0]


def itog(f, u, min_max, start_vals, dt):
    T = [x for x in float_range(min_max[0], min_max[1], dt)]    
    w = len(T)                                                  
    h = len(start_vals)                                          
    x = [[0 for x in range(w)] for y in range(h)]               
    for i in range(len(start_vals)):
        x[i][0] = start_vals[i]                                  

    dt = float(dt)

    for i in range(0, len(T) - 1):
        ti = dt * i
        fi = f(ti, [x[0][i - 1], x[1][i - 1]])
        ui = u(ti, [x[0][i - 1], x[1][i - 1]])
        for j in range(0, len(fi)):
            h = x[j][i] + dt * fi[j] + ui[j]              
            if h > 1000:
                h = 999
            elif h < -1000:
                h = -999
            x[j][i + 1] = h
    return T, x


time, mas = itog(f_calc, u_calc, [0, 7], [7, 5], '0.01')

print('значения Х:')
print(mas)
print('значения Т:')
print(time)

print()
print(len(mas[0]))
print(len(mas[1]))
print(len(time))
print()

# Data
df1 = pd.DataFrame({
    'f1': mas[0]
    ,
    'Time': time
})

df2 = pd.DataFrame({
    'f2': mas[1]
    ,
    'Time': time
})

pl.plot('Time', 'f1', data=df1, marker=',', color='blue', linewidth=1)
pl.plot('Time', 'f2', data=df2, marker=',', color='red', linewidth=1)
pl.xlabel(" Value of t ")
pl.ylabel(" Value of x ")
pl.show()


