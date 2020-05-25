# ###############################################
# Initialization, Input
# ###############################################

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
# initial value of a,b,c and input data x_y
# -----------------------------------------------
# Reading Data
data = pd.read_csv('headbrain.csv')
data.head()
# Collecting X and Y
x = data['Head Size(cm^3)'].values
y = data['Brain Weight(grams)'].values
x_y = np.zeros((len(x),2))
for i in range(len(x)):
    x_y[i] = [x[i], y[i]]
x = x_y[:,0]
y = x_y[:,1]
x_mean = np.mean(x_y[:,0])
y_mean = np.mean(x_y[:,1])



# ###############################################
# Functions
# ###############################################

# C1 related functions
# *******************************************************
# derivative of cost func 1: C_1(a,b,c) = SUM max(di)
# ========================================================
def df_cost1(a_b_c, x_y):
    [a, b, c] = a_b_c
    [x, y] = x_y
    d_a = (x * math.sqrt(a ** 2 + b ** 2) - a * (a * x + b * y + c)) / (a ** 2 + b ** 2)
    d_b = (y * math.sqrt(a ** 2 + b ** 2) - b * (a * x + b * y + c)) / (a ** 2 + b ** 2)
    d_c = (1 / math.sqrt(a ** 2 + b ** 2))
    df = np.asarray([d_a, d_b, d_c])
    return df

# adjust (a,b,c) based on h and df
# ========================================================
def update_coef(a_b_c, d_abc):
    a_updated = a_b_c[0] + d_abc[0]
    b_updated = a_b_c[1] + d_abc[1]
    c_updated = a_b_c[2] + d_abc[2]
    a_b_c_updated = [a_updated, b_updated, c_updated]
    return a_b_c_updated

# distance between point (x,y) to line, ax+by+c=0
# ========================================================
def normalDistance(a_b_c, x_y):
    a = a_b_c[0]
    b = a_b_c[1]
    c = a_b_c[2]
    x = x_y[0]
    y = x_y[1]
    # compute vertical distance between (x,y) and the line(a,b,c)
    a_b_c = np.asarray([a, b, c])
    d = (a * x + b * y + c) / math.sqrt(a ** 2 + b ** 2)
    return d

# maximum of d (in-batch)
# ============================
def sort_topN_max(d, N_top):
    d_sorted_big2small = np.sort(d)[::-1][0:N_top]
    return d_sorted_big2small


# C2 related function
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# derivative of cost func 2: C_2(a,b,c) = SUM (-ax/b-c/b-y)^2
# ========================================================
def df_cost2(a_b_c, x_y):
    [a, b, c] = a_b_c
    [x, y] = x_y
    da = (-x / b) * 2 * (-a * x / b - c / b - y)
    db = (a * x + c) / b * 2 * (-a * x / b - c / b - y)
    dc = (-1 / b) * 2 * (-a * x / b - c / b - y)
    df = np.asarray([da, db, dc])
    return df


# Top_N minimum of a numpy array
# ========================================================
def sort_topN_min(d, N_top):
    d_sorted_small2big = np.sort(d)[::-1][0:N_top]
    return d_sorted_small2big


# epsilon
# ========================================================
def epsilon_cost2(a_b_c, x_y):
    [a, b, c] = a_b_c
    [x, y] = x_y
    epsilon = (-a * x / b - c / b - y)
    return epsilon


# Plot
# ###############################################
# get y from ax+by+c=0
def get_y_from_a_b_c(x, a_b_c_final):
    # [a, b, c] = a_b_c_final
    a = a_b_c_final[0]
    b = a_b_c_final[1]
    c = a_b_c_final[2]
    # y = -ax/b -c/b
    y_regression_line = -a * x / b - c / b
    return y_regression_line


# Plot
# ========================================================
def plot_data(x_y, a_b_c_final):
    # [x, y] = x_y.reshape(N_batches*N_xy_perBatch,2)
    x = x_y.reshape(N_batches * N_xy_perBatch, 2)[:, 0]
    y = x_y.reshape(N_batches * N_xy_perBatch, 2)[:, 1]
    # [a, b, c] = a_b_c_final
    a = a_b_c_final[0]
    b = a_b_c_final[1]
    c = a_b_c_final[2]
    # get y on the regression line
    y_regression_line = get_y_from_a_b_c(x, a_b_c_final)
    # plot
    plt.figure()
    plt.plot(x, y, 'bo', x, y_regression_line, 'k')
    plt.yscale('linear')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Liner Regression Classifies Input Data')
    plt.grid(True)
    # plt.legend(loc='lower right')
    plt.show()

# Plot
# ========================================================
def plot_data_compare2line(x_y, a_b_c_final, a_b_c_final_2):
    # [x, y] = x_y.reshape(N_batches*N_xy_perBatch,2)
    x = x_y.reshape(N_batches * N_xy_perBatch, 2)[:, 0]
    y = x_y.reshape(N_batches * N_xy_perBatch, 2)[:, 1]
    # [a, b, c] = a_b_c_final
    a = a_b_c_final[0]
    b = a_b_c_final[1]
    c = a_b_c_final[2]
    # get y on the regression line
    y_regression_line = get_y_from_a_b_c(x, a_b_c_final)
    y_regression_line_2 = get_y_from_a_b_c(x, a_b_c_final_2)
    # plot
    plt.figure()
    plt.plot(x, y, 'bo', label='Input Points')
    plt.plot(x, y_regression_line, 'k', label='$Mean(\epsilon^2)$')
    plt.plot(x, y_regression_line_2, 'g*', label='Sum $\epsilon^2/N_perBatch$')
    plt.yscale('linear')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Liner Regression Classifies Input Data')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()




# ###############################################
# Least Mean Square Method
# ###############################################
# ==================begin=====================
numerator = 0
denominator = 0
for i in range(len(x)):
    x_ref = np.mean(x)
    y_ref = np.mean(y)
    numerator += (x[i] - x_ref) * (y[i] - y_ref)
    denominator += (x[i] - x_ref ) ** 2
b_ = 1
a_ = (-b_) * (numerator/denominator)
c_ = (-b_)* (-a_*x_ref - y_ref )
print('a=',a_)
print('b=',b_)
print('c=',c_)
a_b_c = [a_, b_, c_]
y_test = get_y_from_a_b_c(x, a_b_c)
# =================end======================
plt.figure()
plt.plot(x, y, 'bo', label='Input Points')
plt.plot(x, y_test, 'ro', label='$LSME$')
plt.yscale('linear')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Liner Regression Classifies Input Data')
plt.grid(True)
plt.legend(loc='lower right')
plt.show()



# ###############################################
# Processing
# ###############################################
N_batches = 3
N_xy_perBatch = 79
N_top = 9
N_iter = len(x)
N_epoch = 7
a_b_c_initial_value = np.random.rand(1, 3)
# x_y = np.random.rand(N_batches, N_xy_perBatch, 2)
x_y = x_y.reshape((N_batches, N_xy_perBatch,2))

# initialization of min c1
# *******************************************************
a_b_c_is = np.zeros((N_batches, N_xy_perBatch, 3))
d = np.zeros((N_batches, N_xy_perBatch, 1))
d_mean_list = []  # list
d_max_index = np.zeros((N_batches, 1))
d_sortedTopN_big2small = np.zeros((N_batches, N_top))
d_mean = np.zeros((N_batches, 1))
d_max_index_list = []
a_b_c_final = np.zeros((1, 3))
x_y_final = np.zeros((1, 2))

# Initialization of min c2
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
a_b_c_is_c2 = np.empty((N_batches, N_xy_perBatch, 3))
epsilon = np.zeros((N_batches, N_xy_perBatch, 1))
epsilon_min_index = np.zeros((N_batches, 1))
epsilon_min_index_list = []
epsilon_sortedTopN_small2big = np.zeros((N_batches, N_top))
epsilon_mean = np.zeros((N_batches, 1))
epsilon_mean_list = []
a_b_c_final_c2 = np.zeros((1, 3))
x_y_final_c2 = np.zeros((1, 2))


count = 0
print('Starting ........................................................')
for k in range(N_epoch):
    print('Epoch = ', k)
    numerator = 0
    denominator = 0
    # minimize max(d[i])
    # *******************************************************
    for ii in range(N_iter):
        print('Iteration = ', i)
        print('Iteration C1')
        print('======================================================')
        # Parallel processing cross all batches
        # ========================================================
        a_b_c = a_b_c_initial_value[0]
        for j in range(N_batches):
            # Serial Max Processing inside Each Batch
            # -------------------------------------------------------
            h_list = []
            for i in range(N_xy_perBatch):
                count = count + 1
                # derivative and based input(x,y) to update (a,b,c)
                df = df_cost1(a_b_c, x_y[j][i])
                a_b_c_is[j][i] = update_coef(a_b_c, df)
                # compute the normal distance
                d[j][i] = normalDistance(a_b_c_is[j][i], x_y[j][i])
                h_list.append(float(d[j][i]))
                # update a,b,c
                a_b_c = a_b_c_is[j][i]
            #  the index of maximum d per batch
            d_max_index[j] = np.argmax(np.asarray(h_list))
            d_max_index_list.append(int(d_max_index[j]))
            # pick up the N_top maximum elements in h_list
            # d_sortedTopN_big2small[j] = sort_topN_max(h_list, N_top)
            d_sortedTopN_big2small[j] = np.sort(h_list)[::-1][0:N_top]
            # mean of the top_N h
            d_mean[j] = np.mean(d_sortedTopN_big2small[j])
        d_mean_list.append(float(d_mean[j]))
        print('len(d_max_index_list) = ', len(d_max_index_list))
        print('d_max_index_list = ', d_max_index_list)
        # -------------------------------------------------------
        # index of the maximum of d_mean of all batches
        d_max_index_allBatches = np.argmax(np.asarray(d_mean_list))
        # minimum value of d
        d_max = max(np.asarray(d_mean_list))
        # based d_max_index_allBatches to reach the final a,b,c
        print('d_max_index_allBatches =', d_max_index_allBatches)
        a_b_c_final = a_b_c_is[d_max_index_allBatches][np.asarray(d_max_index_list)[d_max_index_allBatches]]
        # get the point (x,y)
        x_y_final = x_y[d_max_index_allBatches, np.asarray(d_max_index_list)[d_max_index_allBatches]]
        # *******************************************************
    [a,b,c] = a_b_c_final
    # Iteration: Dummy Test:
    # ==============begin================================
    x_ref = np.mean(x)
    y_ref = -a*x/b-c/b
    y_ref = np.mean(y)
    numerator += (x[i] - x_ref) * (y[i] - y_ref)
    denominator += (x[i] - x_ref ) ** 2
    # ==============end===================================

    print('Epoch C2')
    print('======================================================')
    a_b_c = a_b_c_final
    # minimize (-a*x[i]/b-c/b-y[i])^2
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    for jj in range(N_batches):
        epsilon_list = []
        epsilon_sum = 0
        for ii in range(N_xy_perBatch):
            # derivative and based input(x,y) to update (a,b,c)
            df_c2 = df_cost2(a_b_c, x_y[jj][ii])
            a_b_c_is_c2[jj][ii] = update_coef(a_b_c, df_c2)
            # epsilon of cost function 2
            epsilon[jj][ii] = epsilon_cost2(a_b_c, x_y[jj][ii])
            epsilon_list.append(float(epsilon[jj][ii]))
            # sum of epsilon
            epsilon_sum = epsilon_sum + epsilon[jj][ii]
            # update a_b_c_c2
            a_b_c = a_b_c_is_c2[jj][ii]
        # epsilon
        # ---------------------------------------------------------------------------
        # the index of minimum of epsilon per batch
        epsilon_min_index[jj] = np.argmin(np.asarray(epsilon_list))
        # save index
        epsilon_min_index_list.append(int(epsilon_min_index[jj][0]))
        # pick up the N_top minimum elements in epsilon_list
        epsilon_sortedTopN_small2big[jj] = np.sort(epsilon_list)[0:N_top]
        # mean of the top_N minimum epsilon
        epsilon_mean[jj] = np.mean(epsilon_sortedTopN_small2big[jj])
        epsilon_mean_list.append(float(epsilon_mean[jj]))

    # ---------------------------------------------------------------------------
    print('Epsilon')
    print('-----------------------------------------')
    print('len(epsilon_min_index_list) = ', len(epsilon_min_index_list))
    print('epsilon_min_index_list = ', epsilon_min_index_list)
    # index of the minimum of epsilon of all
    epsilon_min_index_allBatches = np.argmin(epsilon_mean)
    epsilon_min = min(epsilon_mean)
    a_b_c_final_c2 = a_b_c_is_c2[epsilon_min_index_allBatches][np.asarray(epsilon_min_index_list)[epsilon_min_index_allBatches]]
    x_y_final_c2 = x_y[epsilon_min_index_allBatches][np.asarray(epsilon_min_index_list)[epsilon_min_index_allBatches]]

    a_b_c_final_c2 = a_b_c_is_c2[epsilon_min_index_allBatches][int(epsilon_min_index[epsilon_min_index_allBatches])]
    x_y_final_c2 = x_y[epsilon_min_index_allBatches][int(epsilon_min_index[epsilon_min_index_allBatches])]

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print('Iteration End !!!\n')
print('a_b_c_final_c2 = ', a_b_c_final_c2)
a_b_c_finalResult = a_b_c_final_c2
# plot_data(x_y, a_b_c_finalResult)





# x = x_y.reshape(N_batches * N_xy_perBatch, 2)[:, 0]
# y = x_y.reshape(N_batches * N_xy_perBatch, 2)[:, 1]
# [a, b, c] = a_b_c_final
a = a_b_c_finalResult[0]
b = a_b_c_finalResult[1]
c = a_b_c_finalResult[2]
# get y on the regression line
y_regression_line = get_y_from_a_b_c(x, a_b_c_final)
y_regression_line_c2 = get_y_from_a_b_c(x, a_b_c_final_c2)
# plot
plt.figure()
plt.plot(x, y, 'bo', label='Input Points')
# plt.plot(x, y_regression_line, 'k-', label='$Mean(\epsilon^2)$')
plt.plot(x, y_regression_line_c2, 'g*', label='Sum $\epsilon^2/N$')
# Least Mean Square Method
# plt.plot(x, y_test, 'ro', label='$LSME$')
plt.yscale('linear')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Liner Regression Classifies Input Data')
plt.grid(True)
plt.legend(loc='lower right')
plt.show()



