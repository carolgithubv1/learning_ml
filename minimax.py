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

	    # initialize value of nodes by a_b by recursively calling the func calls itself
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
# when d=5
# [[0.36239711090152626, 0.9505049969166952], [0.4404281911988689, 0.6607685025611586], [0.4147504722753387, 0.999669689737674], [0.10885687709147274, 0.7153656432244169], [0.20877758191111295, 0.98956706327814]
# when d = 4
# Input: x_original = [[0.08368198218939582, 0.8693755640400849], [0.43306721152658034, 0.8928105197076007], [0.718655261308865, 0.7553382307348262], [0.27217065359575277, 0.08416951385327631], [0.5984888686059232, 0.08868979298384316], [0.7791896890723393, 0.5571859543509925], [0.6319663144908938, 0.6622244529854451], [0.8904772523834654, 0.8328770540240288], [0.20935942270350938, 0.43490072290382165]]
# when d = 3
# Input: x_original = [[0.010283731869551005, 0.8717197217670714], [0.05672950834720958, 0.8205226736822478], [0.14161606562602913, 0.962547209682025], [0.4406515259763618, 0.35290595962059623], [0.2289099220747265, 0.09661409319308012], [0.2005329736882433, 0.9914223591002138], [0.7901435946156712, 0.8907198970978077], [0.4934339454932539, 0.09640680256470302], [0.3922750812768153, 0.5819657584109377]]
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

