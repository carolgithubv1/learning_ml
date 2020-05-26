import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


# 1) find maximum d
# distance between point (x,y) to line, ax+by+c=0
# ========================================================
def find_maxDistance_pointLine(a_b_c, x_y):
    a = a_b_c[0]
    b = a_b_c[1]
    c = a_b_c[2]
    x = x_y[:,0]
    y = x_y[:,1]
    # compute normal distance between (x,y) and the line(a,b,c)
    # a_b_c = np.asarray([a, b, c])
    flag = a*x+b*y+c
    flag_n = np.copy(flag)
    flag_p = np.copy(flag)
    flag_n[flag_n>0] = 0
    flag_n = - flag_n
    flag_p[flag_p<0] = 0
    flag = flag_n + flag_p
    # d = (a * x + b * y + c) / math.sqrt(a ** 2 + b ** 2)
    d = flag / math.sqrt(a ** 2 + b ** 2)
    # d_sorted_big2small = np.sort(d)[::-1]
    # return the d_max and its index
    d_max = np.max(d)
    d_max_index = np.argmax(d)
    return d_max_index, d_max



# 2) compute derivates for a, b ,c from that point
def df_cost1(a_b_c, x_y):
    [a, b, c] = a_b_c
    [x, y] = x_y
    flag = a * x + b * y + c
    flag_n = np.copy(flag)
    flag_p = np.copy(flag)
    flag_n[flag_n > 0] = 0
    flag_n = - flag_n
    flag_p[flag_p < 0] = 0
    flag = flag_n + flag_p
    dC1_da = x * math.sqrt(a ** 2 + b ** 2) - a * flag / (a ** 2 + b ** 2)
    dC1_db = y * math.sqrt(a ** 2 + b ** 2) - b * flag / (a ** 2 + b ** 2)
    dC1_dc = 1 / math.sqrt(a ** 2 + b ** 2)

    d_a = 1/dC1_da
    d_b = 1/dC1_db
    d_c = 1/dC1_dc
    df = np.asarray([d_a, d_b, d_c])
    return df

# 3)add those derivates to a, b, c
def update_abc_n(a_b_c, df, learning_rate):
    a_updated = a_b_c[0] - df[0]*learning_rate
    b_updated = a_b_c[1] - df[1]*learning_rate
    c_updated = a_b_c[2] - df[2]*learning_rate
    a_b_c_updated = [a_updated, b_updated, c_updated]
    return a_b_c_updated
def update_abc_p(a_b_c, df, learning_rate):
    a_updated = a_b_c[0] + df[0]*learning_rate
    b_updated = a_b_c[1] + df[1]*learning_rate
    c_updated = a_b_c[2] + df[2]*learning_rate
    a_b_c_updated = [a_updated, b_updated, c_updated]
    return a_b_c_updated

# 4) get y from ax+by+c=0
def get_y_from_a_b_c(x, a_b_c_final):
    # [a, b, c] = a_b_c_final
    a = a_b_c_final[0]
    b = a_b_c_final[1]
    c = a_b_c_final[2]
    # y = -ax/b -c/b
    y_regression_line = -a * x / b - c / b
    return y_regression_line


# ###############################################
# Plot
# ###############################################
def plot_graph(a_b_c, x_y):
    x = x_y[:,0]
    y = x_y[:,1]
    y_regression_line = get_y_from_a_b_c(x, a_b_c)
    plt.figure()
    plt.scatter(X, Y, color='blue', marker='o', label='Input Points')
    plt.plot(X, y_1, 'g', label='MiniMax')
    plt.yscale('linear')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Liner Regression Classifies Input Data')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()





# ###############################################
# Initialize
# ###############################################
N_epoch = 76
N_iterations = 81*5
# N = 49
# x_y = np.random.randint(1, 100, size=(N,2))
a_b_c = np.random.rand(3,1)
print('Initial Value of a,b,c ')
print('a =', a_b_c[0], ', b = ', a_b_c[1], ', c = ', a_b_c[2])

# Reading Data
data = pd.read_csv('headbrain.csv')
# data.head()
# Collecting X and Y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values
x_y = np.asarray([X,Y])
x_y = x_y.reshape(len(X),2)
learning_rate = 0.05


# ###############################################
# Processing: MiniMax
# ###############################################
print('\n')
print('Normal Distance Change:')
print('================================')
for i in range(N_epoch):
    for j in range(N_iterations):
        d_max_index, d_max = find_maxDistance_pointLine(a_b_c, x_y)
        df = df_cost1(a_b_c, x_y[d_max_index])
        a_b_c = update_abc_n(a_b_c, df, learning_rate)
    print('epoch=', i, 'd_max = ', d_max)
# plot_graph(a_b_c, x_y)
a_b_c_1 = a_b_c
[a,b,c] = a_b_c_1
print('================================')
print('-a/b = ', -a/b, ', -c/b =', -c/b)
# max_x = np.max(X) + 100
# min_x = np.min(X) - 100
# x = np.linspace(min_x, max_x, 1000)
y_1 = -a*X/b-c/b



# ###############################################
# Process: LMS
# ###############################################
# Calculating coefficient
# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)
# Total number of values
n = len(X)
# Using the formula to calculate b1 and b2
numerator = 0
denominator = 0
for i in range(n):
    numerator += (X[i] - mean_x) * (Y[i] - mean_y)
    denominator += (X[i] - mean_x) ** 2
b  = 1
a = - numerator / denominator
c =  (-a/b * mean_x - mean_y)*b   # printing the coefficient
a_b_c_2 = [a,b,c]
y_2 = y = -a*X/b-c/b
print('\n')
print('Virtical Distance Change:')
print('-a/b = ', -a/b, ', -c/b =', -c/b)
print('================================')
# print('d_virtical = ', abs(y_2 - Y))
for j in range(n):
    d_verticalDistance = abs(y_2[j] - Y[j])
    print('input= ', j, 'd_vertialDistance= ', d_verticalDistance)
print('================================')
print('-a/b = ', -a/b, ', -c/b =', -c/b)

# ###############################################
# Plot
# ###############################################
x = x_y[:,0]
y = x_y[:,1]
y_regression_line = get_y_from_a_b_c(x, a_b_c)
plt.figure()
plt.scatter(X, Y, color='blue', marker='o', label='Input Points')
plt.plot(X, y_1, 'g', label = 'MiniMax')
plt.plot(X, y_2, 'r-', label = 'LMS')
plt.yscale('linear')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Liner Regression Classifies Input Data')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()


# ###############################################
# Error
# ###############################################
# Calculating Root Mean Squares Error
rmse_1 = 0
rmse_2 = 0
for i in range(n):
    rmse_1 += (Y[i] - y_1[i])**2
    rmse_2 += (Y[i] - y_2[i])**2
rmse_1 = np.sqrt(rmse_1/len(Y))
rmse_2 = np.sqrt(rmse_2/len(Y))


# Calculating R2 Score
ss_tot_1 = 0
ss_res_1 = 0
ss_tot_2 = 0
ss_res_2 = 0
for i in range(n):
    ss_tot_1 += (Y[i] - np.mean(y_1)) ** 2
    ss_res_1 += (Y[i] - y_1[i]) ** 2
    ss_tot_2 += (Y[i] - np.mean(y_2)) ** 2
    ss_res_2 += (Y[i] - y_2[i]) ** 2
r2_1 = 1 - (ss_res_1/ss_tot_1)
r2_2 = 1 - (ss_res_2/ss_tot_2)
print('\n')
print('1. Least Maximum Normal Distance + Least Normal Distance')
print('2. Least Mean Square (LMS) of Virtual Distance')
print('Computation Cost :')
print('================================')
print('run_1 = ', N_epoch*N_iterations)
print('run_2 = ', n)
print('\n')
print("RMSE:")
print('================================')
print('rmse_1 = ', rmse_1)
print('rmse_2 = ', rmse_2)
print('\n')
print("R2 Coefficient of Determination")
print('================================')
print('r2_1 = ', r2_1)
print('r2_2 = ', r2_2)
# print("{0:.000000%}".format(1./3))
# print('r2_1 = ', "{0:.000000%}".format(r2_1))
# print('r2_2 = ', "{0:.000000%}".format(r2_2))


