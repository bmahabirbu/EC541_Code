from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import convolve1d
from scipy.stats import expon
from scipy import signal

from numpy.random import default_rng
rng = default_rng()

def Problem_1():

    print("#Solution to A")
    X1 = np.random.uniform(size=10**6)
    print(np.mean(X1))
    #Output = 0.5000792400121572

    print("#Solution to B")
    X2 = np.random.uniform(size=10**6)
    X3 = np.minimum(X1,X2)
    print(np.mean(X3))
    # #Output = 0.3337066875030083

    print("#Solution to C")
    X4 = np.maximum(X1,X2)
    print(np.mean(X4))
    #Output = 0.6669679886039491


    print("#Solution to D\n")
    # CCDF = 1 - CDF
    X5 = X1+X2
    count, bins_count = np.histogram(X5, bins='auto')
    pdf = count / sum(count)
    ccdf = 1-np.cumsum(pdf)

    plt.plot(bins_count[1:], pdf, label="X3 PDF")
    plt.title("PDF") 
    plt.show()

    plt.plot(bins_count[1:], ccdf, label=" X5 CCDF")
    plt.ylabel('P(X > x)') 
    plt.xlabel('values') 
    plt.legend()
    plt.show()

def Problem_2():
    print("Problem 2")
    print("#Solution to A and B")
    X1 = np.random.uniform(1, 2, size=10**6)
    print(np.mean(X1))

def Problem_3():
    print("Problem 3")
    #scale = 1/mean
    Y1 = np.random.exponential(scale=1,size=10**6)
    Y2 = np.random.exponential(scale=1,size=10**6)
    Y2_5 = np.random.exponential(scale=0.5,size=10**6)
    Y3 = Y1+Y2
    print("#Solution to A")
    print(np.mean(Y3))
    #Output = 2.0001132592789936

    print("#Solution to B")
    count, bins_count = np.histogram(Y3, bins='auto')
    pdf = count / sum(count)
    ccdf = 1-(np.cumsum(pdf))

    count2, bins_count2 = np.histogram(Y2_5, bins='auto')
    pdf2 = count2 / sum(count2)
    ccdf2 = 1-(np.cumsum(pdf2))

    plt.plot(bins_count[1:], ccdf, label="Y3 CCDF")
    plt.plot(bins_count2[1:], ccdf2, label="Exp-mean=2 CCDF")
    # plt.plot(Y2_5, ccdf, label="CCDF")
    plt.ylabel('P(Y > y)') 
    plt.xlabel('values') 
    plt.legend()
    plt.show()

def Problem_4():
    #creating an array of values between
    #-1 to 10 with a difference of 0.1

    x = np.linspace(-10, 10, 1000)
    dx = x[1] - x[0]
    
    X = expon.pdf(x, 0, 1)
    Y = expon.pdf(x, 0, 1)

    Z = convolve1d(X,Y) * dx
    
    plt.plot(x, X, label="X and Y PDF")
    plt.plot(x, Z, label='Conv(X,Y) PDF') 
    plt.legend()
    plt.show()

def Problem_5():
    sampled = []
    right = 0
    sample_size = 10**6
    sampled.append(rng.integers(1,64, size=(sample_size,10)))
    for i in range(0,sample_size):
        if len(np.unique(sampled[0][i])) == 9:
            right += 1
    print(right/sample_size)
    #Output = 0.392119 (pretty close)

def Problem_6():
    #6a
    # holding_time_a rng.uniform(0,1,[sample_size,1])
    pb = 0
    sample_size = 10**6

    arrival_times_a = rng.uniform(0,2,[sample_size,1])
    holding_time_a = np.ones([sample_size,1])

    arrival_times_b = rng.uniform(0,2,[sample_size,1])
    holding_time_b = np.ones([sample_size,1])

    a = np.concatenate((arrival_times_a, holding_time_a), axis=1)
    b = np.concatenate((arrival_times_b, holding_time_b), axis=1)

    for i in range(0,sample_size):
        if a[i][0] < b[i][0] and b[i][0] < a[i][0]+a[i][1]:
           # "call a reached first and call b starts before call a finishes"
            pb += 1
        elif b[i][0] < a[i][0] and a[i][0] < b[i][0]+b[i][1]:
            # "call b reached and call a starts before call b finishes"
            pb += 1
    print(pb/(sample_size))

    #6b
    # holding_time_a = rng.uniform(0,1,[sample_size,1])
    pb = 0
    sample_size = 10**6

    arrival_times_a = rng.uniform(0,2,[sample_size,1])
    holding_time_a = rng.uniform(0,1,[sample_size,1])

    arrival_times_b = rng.uniform(0,2,[sample_size,1])
    holding_time_b = rng.uniform(0,1,[sample_size,1])

    a = np.concatenate((arrival_times_a, holding_time_a), axis=1)
    b = np.concatenate((arrival_times_b, holding_time_b), axis=1)

    for i in range(0,sample_size):
        if a[i][0] < b[i][0] and b[i][0] < a[i][0]+a[i][1]:
           # "call a reached first and call b starts before call a finishes"
            pb += 1
        elif b[i][0] < a[i][0] and a[i][0] < b[i][0]+b[i][1]:
            # "call b reached and call a starts before call b finishes"
            pb += 1
    print(pb/(sample_size)) 
    #Outputs 
    #6a = 0.750116
    #6b = 0.417168

Problem_6()