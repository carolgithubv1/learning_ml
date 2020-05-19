# Copyright:
# ==========
# Permission is required. 

# Algorithm Description: 
# ======================
# The input points (x,y) was categorized by a straight line, ax+by+c=0, which optimal parameters are approached by the minimax() algorithm. 

# Advantage of this design:
# =========================
# batch based processing to enable the following parallel processing for algorithm efficiency improvement purpose. 
# the minimax algorithm is picked up to minimize the upper bound of the vertical distance of between points and the optimal line. 


# input: [x,y] plus a random initial guess of [a,b,c]
# output: [a,b,c]



# Thoughts:
# =========
# It is proceeded in two cases 
# 'a': central cost function: find the maximum of d
# 'b': minimize (the central cost function)
# case "a" : maximum of dbased (x,y) and initial (a,b,c) to compute d as the prior info 

#   per point based linear regression (x)
#   per batch based processing: inputs are arrays (v)
#   a cluster of batches based d_max finding
#   a cluster of batches based min d processing to modify (a,b,c) by their gradient values
 
 
# Vertical Distance between a point and a line.
# ============================================
# Point(x,y)
# Line: ax+by+c=0
def linearRegression_minimax(x_y_h, a_b_c_initial_value):
# linear regression
# input: [x,y] plus a random initial guess of [a,b,c]
# output: [a,b,c]


#  ######################################
#  Per Batch Based Processing in Python
#  ######################################


# data structure
# =========================================================
def input(N_batches, N_xy_perBatch, N_top):
   N_batches = 7
   N_xy_perBatch = 5
   N_top = 3
   a_b_c_initial_value = np.random.rand(1, 3)
   x_y = np.random.rand(N_xy_perBatch, 2)
   return a_b_c_initial_value, x_y
 

# adjust (a,b,c) based on h and df
# ========================================================
def update_coef_n(a_b_c, d_abc):
    a_updated = a_b_c[:,0] - d_abc[:,0]
    b_updated = a_b_c[:,1] - d_abc[:,1]
    c_updated = a_b_c[:,2] - d_abc[:,2]
    a_b_c_updated = [a_updated, b_updated, c_updated]
    return a_b_c_updated


# adjust (a,b,c) based on h and df
# ========================================================
def update_coef_p(a_b_c, d_abc):
    a_updated = a_b_c[:,0] + d_abc[:,0]
    b_updated = a_b_c[:,1] + d_abc[:,1]
    c_updated = a_b_c[:,2] + d_abc[:,2]
    a_b_c_updated = [a_updated, b_updated, c_updated]
    return a_b_c_updated


# topN values of d per batch
# ========================================================
 def sort_topN(d, N_top):
    d_sorted_big2small = np.asarray(d)[N_top-1:-1].sort
    d_mean = np.mean(d_sorted_big2small)
    return list(d_sortedTopN_big2small), list(d_mean)


# d_abc
# ========================================================
def delta_abc(a_b_c_initial_value, x_y):
   a_initial_value = a_b_c_initial_value[:,0]
   b_initial_value = a_b_c_initial_value[:,1]
   c_initial_value = a_b_c_initial_value[:,2]
   x = x_y[:,0]
   y = x_y[:,1]

   # compute gradient for later (a,b,c) adjustment
   d_a = x/sqrt(a_initial_value**2 + b_initial_value**2)
   d_b = y/sqrt(a_initial_value**2 + b_initial_value**2)
   d_c = 1/sqrt(a_initial_value**2 + b_initial_value**2)
   d_abc = [d_a, d_b, d_c]
   return d_abc


# update a,b,c
# ========================================================
def update_abc(a_b_c_initial_value, x_y):
   d_abc = delta_abc(a_b_c_initial_value, x_y)
   # update a,b,c
   a_b_c_updated = update_coef_n(a_b_c, d_abc)   # ? should add or minus ???
   return a_b_c_updated


# Distance before an arbitrary point, (x,y), and an aritrary line, ax+by+c=0
def Distance_point_line(a_b_c, x_y):
   a = a_b_c[:,0]
   b = a_b_c[:,1]
   c = a_b_c[:,2]
   x = x_y[:,0]
   y = x_y[:,1]
   d = (a*x+b*y+c)/sqrt(a**2+b**2)
   return d


# distance between point (x,y) to line, ax+by+c=0
# ========================================================
def normalDistance(a_b_c_initial_value, x_y):
   a_initial_value = a_b_c_initial_value[:,0]
   b_initial_value = a_b_c_initial_value[:,1]
   c_initial_value = a_b_c_initial_value[:,2]
   x = x_y[:,0]
   y = x_y[:,1]
   # compute vertical distance between (x,y) and the line(a,b,c)
   a_b_c   = [a_initial_value, b_initial_value, c_initial_value]
   h       = Distance_point_line(a_b_c, x_y)
   return h

 

# maximum of d (in-batch)
# ============================
def sort_topN(d, N_top):
  d_sorted_big2small = np.asarray(d)[N_top-1:-1].sort
  d_mean = np.mean(d_sorted_big2small)
  return list(d_sortedTopN_big2small), list(d_mean)


# Serial Max Processing inside Each Batch
# ========================================================
def maximum_perBatch(a_b_c, x_y, N_xy_perBatch, N_top):
  a_b_c_is = np.empty((N_xy_perBatch,1))
  d        = np.empty((N_xy_perBatch,1))
  for i in range(N_xy_perBatch):
    # based input(x,y) to update (a,b,c)
    a_b_c_is[i] = update_coef_p(a_b_c, h, df)
    # compute the normal distance
    d[i] = normalDistance(a_b_c, x_y[i])
    # update a,b,c
    a_b_c = a_b_c_is[i]
  # pick up the N_top maximum elements in the two arrays
  d_sortedTopN_big2small, d_mean = sort_topN(d, N_top)
  return a_b_c_is, d, d_sortedTopN_big2small, d_mean


# Parallel processing cross all batches 
# ========================================================
def minimum_crossBatches(batches, N_batches, a_b_c, x_y, N_xy_perBatch, N_top):
  for ii in arange(N_batches):
    a_b_c_is[ii], d[ii], d_sortedTopN_big2small[ii], d_mean[ii] = maximum_perBatch(a_b_c, x_y, N_xy_perBatch, N_top) 
  # index of minimum value in d
  d_min_index = np.argmin(d_mean)
  # minimum value of d
  d_min = min(np.asarray(d_mean))
  # based d_min_index to reach the final a,b,c
  a_b_c_final = a_b_c_is[d_min_index]

  return a_b_c_final, a_b_c_is, d, d_sortedTopN_big2small, d_mean

# get y from ax+by+c=0
def get_y_from_a_b_c(x, a_b_c_final):
  [a,b,c] = a_b_c_final
  # y = -ax/b -c/b
  y_regression_line = -a*x/b-c/b
  return y_regression_line


# Plot
# ========================================================
def plot_data(x_y, a_b_c_final):
  [x,y]   = x_y
  [a,b,c] = a_b_c_final
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
  plt.legend(loc='lower right')
  plt.show()

# Processing: 
# ===============================================
import numpy as np
import matplotlib.pyplot as plt

# Initializing
N_batches     = 7
N_xy_perBatch = 5
N_top         = 3
a_b_c_is      = np.empty(N_batches,N_xy_perBatch)
d             = np.empty(N_batches,N_xy_perBatch)
d_sortedTopN  = np.empty(N_batches,N_top)
d_mean        = np.empty(N_batches,1)
batch_d_mean  = np.empty((N_batches,1))

# Input
a_b_c, x_y    = input(N_batches, N_xy_perBatch, N_top)

# the final line: linear regression 
a_b_c_final   = minimum_crossBatches(batches, N_batches, a_b_c, x_y, N_xy_perBatch)

# plot
plot_data(x_y, a_b_c_final)






