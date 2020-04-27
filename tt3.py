import numpy as np
import math
# from random import seed
# from random import gauss


# initialize Binary Tree
# ===================================
# value = [1,2,3]
class Node(object): 
	def __init__(self, value):
			self.a=value[0][0]
			self.b=value[0][1]
			self.left = None # Child left
			self.right= None # Child right
	def inorderTranversal(self):
		# traverse the bt via indexing the left node
		listLeftNode = []
		if self.left and self.right:
			listLeftNode = self.left.inorderTranversal()
			listLeftNode.append([self.a, self.b])
			listLeftNode = listLeftNode + self.right.inorderTranversal()
		return listLeftNode



# x = inorderTranversal(tree.root)


class BinaryTree(object):
	def __init__(self, root):
		self.root= Node(root)  # before creating a tree



# create a binary tree
tree = BinaryTree(np.random.rand(1,2))
tree.root.left = Node(np.random.rand(1,2)) 
tree.root.right = Node(np.random.rand(1,2))
tree.root.left.left = Node(np.random.rand(1,2))
tree.root.left.right = Node(np.random.rand(1,2))
tree.root.right.left = Node(np.random.rand(1,2))
tree.root.right.right = Node(np.random.rand(1,2))

# node in the tree
#                    1
#              2           3
#           4      5    6      7
#           x

# minimax basic 
# ===================================
# x provide values at d=0
def minimax(x, d, a, b, turn):
	a_max = -math.inf
	b_min = math.inf
	if(d==0):
		coef = []
		coef.append(a)
		coef.append(b)
		return coef 
	

	if(turn=='a'):
		x_max = math.inf
		d = d-1
		turn = 'b'
		[x_a, x_b] = x
		x_est, none = minimax(x, d, a, b, turn)
		a_max = max(x_est, a_max)
		x = [a_max, b]
		# prune 
		a = max(x_est, a_max)
		# if(a>=b): break
		# return a_max
		return x, a>=b

	if(turn=='b'):
		d = d-1
		turn = 'a'
		x_est = minimax(x, d, a, b, turn)
		b_min = min(x_est[1], b_min)
		x = [a, b_min]
		# prune
		b     = min(x_est[1], b)
		# if(b<=a): exit()
		# return b_min
		return x, b<=a

			

# Process: recursive update (a,b) upwards :
# ===================================
d = 3  # root is when d=1
breadth_BT = 2**(d-1) # breadth of the tree = 2**(d_max-1)
# x = np.random.rand(breadth_BT, 2)
# transver a note in the binary tree
x = tree.root.inorderTranversal()
turn='a';
a_max = -math.inf
b_min = math.inf
# x(a,b) was stored for each node on the binary tree
for i in range(len(x)):
	# each nodes contains values of x(a,b)
	a = x[i][0]
	b = x[i][1]
	# get updated value for each node 
	x, stop = minimax(x[i], d, a, b, turn)
	if stop:
		break



# for input_d0 in np.nditer(x):	
