import numpy as np 

def M(i, j, I):
	x, y = np.meshgrid(
			np.arange(I.shape[-1]),
			np.arange(I.shape[-2]),
			indexing='xy')
	xi = np.power(x.astype(np.float), i)
	yj = np.power(y.astype(np.float), j)
	return np.sum(xi*yj*I, axis=(-1,-2), dtype=np.float)

def u(p, q, I, M00=None, M01=None, M10=None, x=None, y=None):
	if M00 is None:
		M00 = M(0, 0, I)
	if M01 is None:
		M01 = M(0, 1, I)
	if M10 is None:
		M10 = M(1, 0, I)

	xbar = M10 / M00 
	ybar = M01 / M00 

	if x is None or y is None:
		x, y = np.meshgrid(
			np.arange(I.shape[-1]),
			np.arange(I.shape[-2]),
			indexing='xy')

	if len(I.shape) > 2:
		xp = np.power(x.astype(np.float) - xbar[:, np.newaxis, np.newaxis], p)
		yq = np.power(y.astype(np.float) - ybar[:, np.newaxis, np.newaxis], q)
	else:
		xp = np.power(x.astype(np.float) - xbar, p)
		yq = np.power(y.astype(np.float) - ybar, q)

	return np.sum(xp*yq*I, axis=(-1, -2), dtype=np.float)

# TODO: Get these to pass tests

def scale_invariant_moments(I):
	# Precomputed values
	M00 = M(0, 0, I)
	M01 = M(0, 1, I)
	M10 = M(1, 0, I)

	x, y = np.meshgrid(
		np.arange(I.shape[-1]),
		np.arange(I.shape[-2]),
		indexing='xy')

	u20 = u(2, 0, I, M00, M01, M10, x, y)
	u02 = u(0, 2, I, M00, M01, M10, x, y)
	u11 = u(1, 1, I, M00, M01, M10, x, y)
	u12 = u(1, 2, I, M00, M01, M10, x, y)
	u30 = u(3, 0, I, M00, M01, M10, x, y)
	u03 = u(0, 3, I, M00, M01, M10, x, y)
	u21 = u(2, 1, I, M00, M01, M10, x, y)
	u22 = u(2, 2, I, M00, M01, M10, x, y)
	u00 = u(0, 0, I, M00, M01, M10, x, y)

	def denom(p, q):
		return np.power(u00, 1 + (p+q)/2.)

	return np.array([
		u20 / denom(2, 0), 
		u11 / denom(1, 1), 
		u02 / denom(0, 2),  
		u30 / denom(3, 0),
		u21 / denom(2, 1),
		u12 / denom(1, 2),
		u03 / denom(0, 3),
		u22 / denom(2, 2)
	])


def hu_moments(I):
	n = scale_invariant_moments(I)
	u20 = n[0]
	u11 = n[1]
	u02 = n[2]
	u30 = n[3]
	u21 = n[4]
	u12 = n[5]
	u03 = n[6]

	h1 = u20 + u02
	h2 = (u20 - u02)**2 + 4*(u11**2)
	h3 = (u30 - 3*u12)**2 + (3*u21 - u03)**2
	h4 = (u30 + u12)**2 + (u21 + u03)**2
	h5 = (u30 - 3*u12)*(u30 + u12)*((u30 + u12)**2 - 3*(u21+u03)**2) \
						+ (3*u21 - u03)*(u21 + u03) \
						* (3*(u30 + u12)**2 - (u21 + u03)**2)
	h6 = (u20 - u02)*((u30 + u12)**2 - (u21 + u03)**2) \
			+ 4*u11*(u30 + u12)*(u21 + u03)
	h7 = (3*u21 - u03)*(u30 + u12)*((u30 + u12)**2 - 3*(u21 + u03)**2) \
			- (u30 - 3*u12)*(u21 + u03)*(3*(u30 + u12)**2 - (u21 + u03)**2)

	return np.array([h1, h2, h3, h4, h5, h6, h7])


