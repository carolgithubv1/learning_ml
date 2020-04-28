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
		return a_b, 1

	if turn=='a':
		print('a begins: ')
		print('------------------------')
		print("d =", d, ", turn =", turn)
		turn = 'b'
		a_old = a_b[0]
		b_old = a_b[1]

	    # initialize value of nodes by a_b by recursively the func calls itself
		print('a initialize:')
		a_b_initialized, flag = minimax(x, d - 1, a_b, turn)
		print("x =", x)
		print("a_b_initialized =", a_b_initialized)
		a_initialized = a_b_initialized[0]
		b_initialized = a_b_initialized[1]

		# update value of nodes by comparing a_b with [a,b] stored in x
		print('a update:')
		a_new = x[0]
		b_new = x[1]
		a_is = max(a_new, a_initialized)
		print('a_is = ', a_is)
		b_is = b_initialized
		a_b_is = [a_is, b_is]
		print("a_b_is =", a_b_is)
		stop = a_is>=b_is
		return a_b_is, stop
	if turn=='b':
		print('b begins: ')
		print('------------------------')
		print("d =", d, ", turn =", turn)
		turn='a'
		a_old = a_b[0]
		b_old = a_b[1]

		# initialize node
		print('b initialize:')
		a_b_initialized, flag = minimax(x, d - 1, a_b, turn)
		print("x =", x)
		print("a_b_initialized =", a_b_initialized)
		a_initialized = a_b_initialized[0]
		b_initialized = a_b_initialized[1]

		# update
		print('b update:')
		a_new = x[0]
		b_new = x[1]
		b_is = min(b_new, b_initialized)
		print('b_is = ', b_is)
		a_is = a_initialized
		a_b_is = [a_is, b_is]
		print("a_b_is =", a_b_is)
		stop = b_is<=a_is
		return a_b_is, stop


# create a binary tree
tree = BinaryTree(np.random.rand(1,2))
tree.root.left = Node(np.random.rand(1,2))
tree.root.right = Node(np.random.rand(1,2))

tree.root.left.left = Node(np.random.rand(1,2))
tree.root.left.right = Node(np.random.rand(1,2))
tree.root.right.left = Node(np.random.rand(1,2))
tree.root.right.right = Node(np.random.rand(1,2))

tree.root.left.left.left = Node(np.random.rand(1,2))
tree.root.left.left.right = Node(np.random.rand(1,2))
tree.root.left.right.left = Node(np.random.rand(1,2))
tree.root.left.right.right = Node(np.random.rand(1,2))

tree.root.left.left.left.left = Node(np.random.rand(1,2))
tree.root.left.left.left.right = Node(np.random.rand(1,2))
tree.root.left.left.right.left = Node(np.random.rand(1,2))
tree.root.left.left.right.right = Node(np.random.rand(1,2))
tree.root.left.right.left.left = Node(np.random.rand(1,2))
tree.root.left.right.left.right = Node(np.random.rand(1,2))
tree.root.left.right.right.left = Node(np.random.rand(1,2))
tree.root.left.right.right.right = Node(np.random.rand(1,2))



# Process: recursive update (a,b) upwards :
# ===================================
d = 3  # root is when d=1
breadth_BT = 2**(d-1) # breadth of the tree = 2**(d_max-1)
# x = np.random.rand(breadth_BT, 2)
# transver a note in the binary tree
x = tree.root.inorderTranversal()
x_orig = x
x_updated = []
turn = 'a'
# a = float('-inf')
# b = float('inf')
a = -math.inf
b = math.inf
a_b = [a,b]
# x(a,b) was stored for each node on the binary tree
run=0
print("\nStarts:.......")
print('========================')
print('original tree value: x=', x)
for x_ in x:
	# each nodes contains values of x(a,b)
	run = run+1
	print("\nincoming input x---> ")
	print("run = ", run)
	print("d = ", d)
	# get updated value for each node 
	a_b_new, stop = minimax(x_, d, a_b, turn)
	a_b = a_b_new
	x_ = a_b_new
	x_updated.append(x_)
	print("Input: x_original =", x_orig)
	print("Result: x_updated =", x_updated)
	# prune binary tree if a_is>b_is
	if stop :
		break

