import numpy as np
import math

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

class BinaryTree(object):
	def __init__(self, root):
		self.root= Node(root)  # before creating a tree


# minimax basic 
# ===================================
# x provide values at d=0
def minimax(x, d, a_b, turn):
	# x=[a,b]: input, an incoming [a,b]
	# d: depth of the binary tree
	# a_b = [a,b]: the current [a,b] value of the current node
	# turn: type of the current node, black or white
	if(d==0):
		return a_b 

	if(turn=='a'):
		d = d-1
		turn = 'b'
		a = a_b[0]
		b = a_b[1]
		a_b_new, flag = minimax(x, d, a_b, turn)
		a_new = a_b_new[0]
		b_new = a_b_new[1]
		a_max = max(a_new, a)
		a_b = [a_max, b]
		return a_b, a_max>=b

	if(turn=='b'):
		d = d-1
		turn = 'a'
		a = a_b[0]
		b = a_b[1]
		a_b_new, flag = minimax(x, d, a_b, turn)
		a_new = a_b_new[0]
		b_new = a_b_new[1]
		b_min = min(b_new, b)
		a_b = [a, b_min]
		return a_b, b_min<=a


# create a binary tree
tree = BinaryTree(np.random.rand(1,2))
tree.root.left = Node(np.random.rand(1,2)) 
tree.root.right = Node(np.random.rand(1,2))
tree.root.left.left = Node(np.random.rand(1,2))
tree.root.left.right = Node(np.random.rand(1,2))
tree.root.right.left = Node(np.random.rand(1,2))


# Process: recursive update (a,b) upwards :
# ===================================
d = 3  # root is when d=1
breadth_BT = 2**(d-1) # breadth of the tree = 2**(d_max-1)
# x = np.random.rand(breadth_BT, 2)
# transver a note in the binary tree
x = tree.root.inorderTranversal()
turn='a';
# x(a,b) was stored for each node on the binary tree
for i in range(len(x)):
	# each nodes contains values of x(a,b)
	a = -math.inf
	b = math.inf
	a_b = [a, b]
	# get updated value for each node 
	x, stop = minimax(x[i], d, a_b, turn)
	if stop :
		break

