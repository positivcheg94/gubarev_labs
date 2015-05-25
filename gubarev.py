__author__ = 'yang'

import random
import numpy
import math
import matplotlib.pyplot as plt
from openpyxl import Workbook


# random complex pairs constants
CONST_SIZE = 5
CONST_FUNC_SIZE = CONST_SIZE*3
CONST_FIRST_ALIGN = 5.0
CONST_FIRST_MULTIPLIER = 2.0
CONST_SECOND_ALIGN = 5.0
CONST_SECOND_MULTIPLIER = 2.0
CONST_DIVISOR = 1e4
CONST_T_X_ALIGN = 10.0
CONST_T_Y_ALIGN = 10.0
CONST_T_X_MULTIPLIER = 10.0
CONST_T_Y_MULTIPLIER = 10.0

# const n
CONST_N = 150

# eps impact
CONST_EPS_QUANTITY = CONST_N*2
CONST_EPS = 1e-2

# const detla_t
CONST_DELTA = 1e-2

times = [i*CONST_DELTA for i in xrange(CONST_EPS_QUANTITY)]

# const excel file name
CONST_EXCEL_FILE_NAME = "gubaref.xlsx"

# create workbook
wb = Workbook()
# get active worksheet
ws = wb.active

series11 = [random.random() for i in xrange(CONST_SIZE)]
series12 = [random.random() for i in xrange(CONST_SIZE)]

series21 = [random.random() for i in xrange(CONST_SIZE)]
series22 = [random.random() for i in xrange(CONST_SIZE)]

series31 = [random.random() for i in xrange(CONST_SIZE)]
series32 = [random.random() for i in xrange(CONST_SIZE)]

series1 = []
series2 = []
series3 = []

for i in xrange(CONST_SIZE):
    # first
    series11[i]=series11[i]*CONST_FIRST_MULTIPLIER+CONST_FIRST_ALIGN
    series12[i]=series12[i]/CONST_DIVISOR
    series1.append(complex(series12[i],series11[i]))
    # second
    series21[i]=series21[i]*CONST_SECOND_MULTIPLIER+CONST_SECOND_ALIGN
    series22[i]=series21[i]/CONST_DIVISOR
    series2.append(complex(series21[i],series22[i]))
    # third
    series31[i] = (series31[i]-0.5)*CONST_T_X_MULTIPLIER + CONST_T_X_ALIGN
    series32[i] = (series32[i]-0.5)*CONST_T_Y_MULTIPLIER + CONST_T_Y_ALIGN
    series3.append(complex(series31[i],series32[i]))

series = series1 + series2 + series3

ws.append(["our random values"])
ws.append(["Real"]+map(lambda x:x.real,series))
ws.append(["Imaginary"]+map(lambda x:x.imag,series))


# f and g funcs o_O
fs = [(random.random()*0.3+0.7)*(random.randint(0,1)*2-1) for i in xrange(3*CONST_SIZE)]
fc = [(random.random()*0.3+0.7)*(random.randint(0,1)*2-1) for i in xrange(3*CONST_SIZE)]
ws.append([])
ws.append(["function"])
ws.append(["fs"]+fs)
ws.append(["fc"]+fc)

# define h,y,h_back
h = [0.0]*CONST_EPS_QUANTITY
y = [0.0]*CONST_EPS_QUANTITY
h_back = [0.0]*CONST_EPS_QUANTITY

# choosing eps
eps = [2.0*CONST_EPS*(random.random()-0.5) for i in xrange(CONST_EPS_QUANTITY)]
ws.append([])
ws.append(["epsilon impact"])
ws.append(range(1,CONST_EPS_QUANTITY+1))
ws.append(eps)

for j in xrange(1,CONST_EPS_QUANTITY+1):
    sum = 0.0
    for p in xrange(CONST_FUNC_SIZE):
        sum += (fc[p]*math.cos(series[p].imag*CONST_DELTA*j)+fs[p]*math.sin(series[p].imag*CONST_DELTA*j))*math.exp(-series[p].real*CONST_DELTA*j)
    h[j-1] = sum

ws.append([])
ws.append(["h0j"])
ws.append(range(1,CONST_EPS_QUANTITY+1))
ws.append(h)

for k in xrange(0,CONST_EPS_QUANTITY):
    sum = 0.0
    for j in xrange(k):
        sum += h[j]*eps[k-j]
    y[k] = sum

ws.append([])
ws.append(["y"])
ws.append(range(1,CONST_EPS_QUANTITY+1))
ws.append(y)


H_list = [h[i:i+CONST_N] for i in xrange(CONST_N)]
H = numpy.matrix(H_list)

S,V,D = numpy.linalg.svd(H,True)


# showing all data via matplotlib
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.axis([-1,15,-1,15])
plt.plot([series[i].real for i in xrange(len(series))],[series[i].imag for i in xrange(len(series))],"mo")
plt.show()

fig,axes = plt.subplots(2,1,True)
axes[0].plot(fc)
axes[1].plot(fs)
plt.show()

plt.plot(times,eps)
plt.show()

plt.plot(times,h)
plt.show()

plt.plot(times,y)
plt.show()










wb.save(CONST_EXCEL_FILE_NAME)
