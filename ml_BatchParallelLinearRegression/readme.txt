Algorithm Description: 

The input points (x,y) was categorized by a straight line, ax+by+c=0, which optimal parameters are approached by the minimax() algorithm. 

Advantage of this design:

batch based processing to enable the following parallel processing for algorithm efficiency improvement purpose. 
the minimax algorithm is picked up to minimize the upper bound of the vertical distance of between points and the optimal line. 


input: [x,y] plus a random initial guess of [a,b,c]
output: [a,b,c]



Thoughts:
=========
It is proceeded in two cases 
 #  'a': central cost function: find the maximum of d
 # 'b': minimize (the central cost function)
 case "a" : maximum of dbased (x,y) and initial (a,b,c) to compute d as the prior info 

 #   per point based linear regression (x)
 #   per batch based processing: inputs are arrays (v)
 #   a cluster of batches based d_max finding
 #   a cluster of batches based min d processing to modify (a,b,c) by their gradient values
 
 
Vertical Distance between a point and a line.
============================================
Point(x,y)
Line: ax+by+c=0


def linearRegression_minimax(x_y_h, a_b_c_initial_value):
# linear regression
# input: [x,y] plus a random initial guess of [a,b,c]
# output: [a,b,c]
def linearRegression_minimax(x_y_h, a_b_c_initial_value):


a-0) data structure
===================

def input():
   N_batches = 7
   N_xy_perBatch = 5
   N_top = 3
   x_y = np.random.rand(N_xy_perBatch, 2)
   a_b_c_initial_value = np.random.rand(N_xy_perBatch, 3)
   return x_y, a_b_c_initial_value
 
 a-1) abstract variables which are column vectors (in-batch)
 
 def inputPrepare(x_y, a_b_c_initial_value):
   a_initial_value = a_b_c_initial_value[:,0]
   b_initial_value = a_b_c_initial_value[:,1]
   c_initial_value = a_b_c_initial_value[:,2]
   x = x_y[:,0]
   y = x_y[:,1]

   # compute vertical distance between (x,y) and the line(a,b,c)
   a_b_c = [a_initial_value, b_initial_value, c_initial_value]
   x_0_y_0 = [x,y]
   h = verticalDistance_point_line(a_b_c, x_0_y_0)

   # compute gradient for later (a,b,c) adjustment
   df_a = x/sqrt(a**2 + b**2)
   df_b = y/sqrt(a**2 + b**2)
   df_c = 1/sqrt(a**2 + b**2)

   # prepared data
   x_y_h = [x,y,h]
   df    = [df_a, df_b, df_c]
   return a_b_c, x_y_h, df
   
 a-2) maximum of d (in-batch)
 ============================
 
 def sort_topN(d, N_top):
    d_sorted_big2small = np.asarray(d)[N_top-1:-1].sort
    d_mean = np.mean(d_sorted_big2small)
    return list(d_sortedTopN_big2small), list(d_mean)
 
 a-3) adjust (a,b,c) based on h and df
 =====================================
 
 def update_coef_p(a_b_c, h, df):
    a_updated = a_b_c[:,0] - df[:,0]
    b_updated = a_b_c[:,1] - df[:,1]
    c_updated = a_b_c[:,2] - df[:,2]
    a_b_c_updated = [a_updated, b_updated, c_updated]
    return a_b_c_updated
    
    
a-4) parallel processing for maximum of d (cross-batch)
b-1) minimizing d




# processing
# =====================================================
N_batches, N_xy_perBatch, N_top, x_y, a_b_c_initial_value  = input()
turn = 'a'

# def parallel_Process_Batches_Minimax(N_batches, N_xy_perBatch, N_top, x_y, a_b_c_initial_value, d, df ):
    #  maximum of d (cross-batch)
    # -----------------------------------------------------------
if turn=='a':
   turn = 'b'
   a_b_c, x_y_h, df = inputPrepare(x_y, a_b_c_initial_value)
   h = x_y_h[2]
   d = h
   # all inputs to the parallel processing are batches
   for ii in range(N_batches):
        d_sortedTopN_big2small, d_sortedTopN_mean = sort_topN(d, N_top)
        batch_d_mean = np.empty((N_batches,1))
        batch_d_sortedTopN = np.empty((N_batches, N_top))
        batch_d_sortedTopN[ii] = d_sortedTopN_big2small
        batch_d_mean[ii] = np.mean(batch_d_sortedTopN)
        if ii>1:
            if(batch_d_mean[ii]>batch_d_mean[ii-1]):
               cluster_d_max_idx = ii
               a_b_c_is = update_coef_p(a_b_c, h, df)
            else:
               a_b_c_is = update_coef_m(a_b_c, h, df)
#  minimizing d
# -----------------------------------------------------------
if turn == 'b':
   turn = 'a'
   a_b_c, x_y_h, df = inputPrepare(x_y, a_b_c_initial_value)
   h = x_y_h[2]
   # all inputs to the parallel processing are batches
   for ii in range(N_batches):
       d_sortedTopN_big2small, d_mean = sort_topN(d, N_top)
       batch_d_sortedTopN[ii] = d_sortedTopN_big2small
       batch_d_mean[ii] = d_sortedTopN_mean
       if ii > 1:
           if (batch_d_mean[ii] < batch_d_mean[ii - 1]):
               cluster_d_max_idx = ii
               a_b_c_is = update_coef_m(a_b_c, h, df)
           else:
               a_b_c_is = update_coef_p(a_b_c, h, df)

a_is = a_b_c_is[0]
b_is = a_b_c_is[1]
c_is = a_b_c_is[2]


c-1) plotting 
