# this file contains collections of proxes we learned in the class
import numpy as np
from scipy.optimize import bisect

# =============================================================================
# TODO Complete the following prox for simplex
# =============================================================================

# Prox of capped simplex
# -----------------------------------------------------------------------------
def prox_csimplex(z, k):
	"""
	Prox of capped simplex
		argmin_x 1/2||x - z||^2 s.t. x in k-capped-simplex.

	input
	-----
	z : arraylike
		reference point
	k : int
		positive number between 0 and z.size, denote simplex cap

	output
	------
	x : arraylike
		projection of z onto the k-capped simplex
	"""
	# safe guard for k
	assert 0<=k<=z.size, 'k: k must be between 0 and dimension of the input.'

	# TODO do the computation here
	# Hint: 1. construct the scalar dual object and use `bisect` to solve it.
	#		2. obtain primal variable from optimal dual solution and return it.
	#
	# dual problem:
	minz = np.min(z)-1
	maxz = np.max(z)
	x = 0

	def fun(x):
		a = -k
		b = np.sum(np.maximum(0, np.minimum(1, z - x)))
		return a+b

	a = bisect(fun, minz, maxz)

	print(a,z)
	x = np.zeros(z.shape[0])
	for i in range(z.shape[0]):
		if z[i] - a > 1:
			x[i] = 1
		elif z[i] - a < 0:
			x[i] = 0
		else:
			x[i] = z[i] - a

	return x


# def fun(x, k, z):
# 	return -k + np.sum(np.maximum(0, np.minimum(1, z - x)))

if __name__ == '__main__':
	np.random.seed(124)
	m = 5
	n = 2
	k = 1
	z = np.random.randn(m, n)
	x = np.zeros((m, n))

	for i in range(m):
		x[i] = prox_csimplex(z[i], k)

	print(x)