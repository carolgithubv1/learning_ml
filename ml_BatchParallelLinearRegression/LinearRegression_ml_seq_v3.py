import numpy as np
import matplotlib.pyplot as plt
import math


# Initializing input and splitting it in batches
# =========================================================
# Initializing input and splitting it in batches
# =========================================================
def input(N_batches, N_xy_perBatch):
    a_b_c_initial_value = np.random.rand(1, 3)
    x_y = np.random.rand(N_batches, N_xy_perBatch, 2)
    return a_b_c_initial_value[0], x_y

# derivative of cost func 1: C_1(a,b,c) = SUM max(di)
# ========================================================
def df_cost1(a_b_c, x_y):
    [a,b,c] = a_b_c
    [x,y] = x_y
    d_a = (x*math.sqrt(a**2+b**2)-a*(a*x+b*y+c))/(a**2+b**2)
    d_b = (y*math.sqrt(a**2+b**2)-b*(a*x+b*y+c))/(a**2+b**2)
    d_c = (1/math.sqrt(a**2+b**2))
    df = np.asarray([d_a,d_b,d_c])
    return df

# derivative of cost func 2: C_2(a,b,c) = SUM (-ax/b-c/b-y)^2
# ========================================================
def df_cost2(a_b_c, x_y):
  [a,b,c] = a_b_c
  [x,y] = x_y
  da = (-x/b) * 2*(-a*x/b-c/b-y)
  db = (a*x+c)/b * 2*(-a*x/b-c/b-y)
  dc = (-1/b) * 2*(-a*x/b-c/b-y)
  df = np.asarray([da,db,dc])
  return df


# adjust (a,b,c) based on h and df
# ========================================================
def update_coef_p(a_b_c, d_abc):
    a_updated = a_b_c[0] + d_abc[0]
    b_updated = a_b_c[1] + d_abc[1]
    c_updated = a_b_c[2] + d_abc[2]
    a_b_c_updated = [a_updated, b_updated, c_updated]
    return a_b_c_updated

# Distance before an arbitrary point, (x,y), and an aritrary line, ax+by+c=0
def Distance_point_line(a_b_c, x_y):
    a = a_b_c[0]
    b = a_b_c[1]
    c = a_b_c[2]
    x = x_y[0]
    y = x_y[1]
    d = (a * x + b * y + c) /math.sqrt(a ** 2 + b ** 2)
    return d


# distance between point (x,y) to line, ax+by+c=0
# ========================================================
def normalDistance(a_b_c_initial_value, x_y):
    a_initial_value = a_b_c_initial_value[0]
    b_initial_value = a_b_c_initial_value[1]
    c_initial_value = a_b_c_initial_value[2]
    x = x_y[0]
    y = x_y[1]
    # compute vertical distance between (x,y) and the line(a,b,c)
    a_b_c = np.asarray([a_initial_value, b_initial_value, c_initial_value])
    h = Distance_point_line(a_b_c, x_y)
    return h

# maximum of d (in-batch)
# ============================
def sort_topN(d, N_top):
    # d_sorted_big2small = float(d)[N_top - 1:-1].sort
    d_sorted_big2small = np.sort(d)[::-1][0:N_top]
    d_mean = np.mean(d_sorted_big2small)
    return d_sorted_big2small, d_mean



# Processing:
# ===============================================
# Initializing
N_batches = 7
N_xy_perBatch = 5
N_top = 3
a_b_c_is = np.empty((N_batches, N_xy_perBatch,3))
d = np.zeros((N_batches, N_xy_perBatch))
d_sortedTopN = np.zeros((N_batches, N_top))
d_mean = np.zeros((N_batches, 1))
batch_d_mean = np.zeros((N_batches, 1))
a_b_c_final = np.zeros((1,3))
x_y_final = np.zeros((1,2))

# Input
a_b_c, x_y = input(N_batches, N_xy_perBatch)


N_epoch = 7
N_iter = 49
df_final = np.zeros((N_epoch, N_iter,3))
a_b_c_updated_final = np.zeros((N_epoch, N_iter, 3))



for k in range(N_epoch):
    print("Per Epoch:")
    print('============================================')
    for i in range(N_iter):
        # the final line: linear regression
        # a_b_c_is, a_b_c_final, x_y_final, d, d_sortedTopN, d_mean= maximum_crossBatches(a_b_c, x_y, N_batches, N_xy_perBatch, N_top)
        a_b_c_is = np.zeros((N_batches, N_xy_perBatch, 3))
        d = np.zeros((N_batches, N_xy_perBatch,1))
        d_sortedTopN = np.zeros((N_batches, N_top))
        d_mean = np.zeros((N_batches, 1))
        d_mean_list = [] # list
        d_max_index = np.zeros((N_batches,1))
        d_max_index_list = []
        for ii in range(N_batches):
            # a_b_c_is[ii], d[ii], d_max_index[ii], d_sortedTopN[ii], d_mean[ii] = maximum_perBatch(a_b_c, x_y[ii], N_xy_perBatch, N_top)
            a_b_c_is_perBatch = np.empty((N_xy_perBatch, 3))
            df_perBatch = np.empty((N_xy_perBatch,1))
            d_perBatch = np.empty((N_xy_perBatch, 1))
            h_perBatch = []
            x_y_perBatch = x_y[ii]
            for i in range(N_xy_perBatch):
                # compute derivative
                df_perBatch = df_cost1(a_b_c, x_y_perBatch[i])
                # based input(x,y) to update (a,b,c)
                a_b_c_is_perBatch[i] = update_coef_p(a_b_c, df_perBatch)
                # compute the normal distance
                d_perBatch[i] = normalDistance(a_b_c, x_y_perBatch[i])
                h_perBatch.append(d[i])
                # update a,b,c
                a_b_c = a_b_c_is_perBatch[i]
            d_perBatch = h_perBatch
            #  the index of maximum d of the batch
            d_max_index_perBatch = np.argmax(np.asarray(d))
            # pick up the N_top maximum elements in the two arrays
            d_sortedTopN_big2small, d_mean_perBatch = sort_topN(d, N_top)
        a_b_c_is[ii] = a_b_c_is_perBatch[i]
        d[ii] = np.asarray(d_perBatch)
        d_max_index[ii] = d_max_index_perBatch
        d_sortedTopN[ii] = d_sortedTopN_big2small
        d_mean[ii] = d_mean_perBatch


        d_mean_list.append(float(d_mean[ii]))
        d_max_index_list.append(int(d_max_index[ii]))
            # index of minimum value in d
        d_min_index = np.argmax(np.asarray(d_mean_list))
        # minimum value of d
        d_min = min(np.asarray(d_mean))
        # based d_min_index to reach the final a,b,c
        a_b_c_final = a_b_c_is[d_min_index, np.asarray(d_max_index_list)[d_min_index]]
        # get the point (x,y)
        x_y_final = x_y[d_min_index, np.asarray(d_max_index_list)[d_min_index]]


        # use a_b_c_final as a good choice of initial (a,b,c) to start update (a,b,c)
        a_b_c = a_b_c_final
    # minimize y
    x_y_ = x_y.reshape(N_batches*N_xy_perBatch,2)
    a_b_c_is = minimize_C2(a_b_c, x_y_)
    a_b_c = a_b_c_is
# plot the final a, b, c
plot_data(x_y, a_b_c_final)
