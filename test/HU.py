from src.humoments import M, u, scale_invariant_moments, hu_moments
import numpy as np
import cv2


def test_M_all_ones():
	# Test standard mean --> x=1, y=2
	I = np.ones([5, 3])
	M00 = M(0, 0, I)
	M01 = M(0, 1, I)
	M10 = M(1, 0, I)

	xbar = M10 / M00 
	ybar = M01 / M00 

	if xbar != 1:
		raise ValueError("xbar is %d. Expected: %d" % (xbar, 1))

	if ybar != 2:
		raise ValueError("ybar is %d. Expected: %d" % (ybar, 2))

	return True

def test_u_all_ones():
	# All ones should *always* have 0 central moment

	I = np.ones([5, 3])
	u_val = u(1, 1, I)

	if u_val != 0:
		raise ValueError("u_val is %d. Expected %d." % (u_val, 0)) 

	return True

def test_moments_against_cv2():

	img = cv2.cvtColor(cv2.imread("images/test_still.png"), cv2.COLOR_RGB2GRAY)
	_, thresh = cv2.threshold(img, 127, 255, 0)

	moments_img_gt = cv2.moments(thresh)
	nus = ['nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']
	mus = ['mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03']
	mu_gt = [moments_img_gt[m] for m in mus]

	x = [2, 1, 0, 3, 2, 1, 0]
	y = [0, 1, 2, 0, 1, 2, 3]
	mu_exp = [u(xi, yi, thresh) for xi, yi in zip(x,y)]

	for i in range(len(mu_gt)):
		mu_gt_x = mu_gt[i]
		mu_exp_x = mu_exp[i]
		if np.abs(mu_gt_x - mu_exp_x) / np.abs(mu_gt_x) > 0.01:
			print('Translation invariant moment mismatch at index: %d' % i)
			print("Expected %f, got %f" % (mu_gt_x, mu_exp_x))
			return False

	nu_gt = [moments_img_gt[n] for n in nus]
	nu_exp = scale_invariant_moments(thresh)
	for i in range(len(nu_gt)):
		nu_gt_x = nu_gt[i]
		nu_exp_x = nu_exp[i]
		if np.abs(nu_gt_x - nu_exp_x) / np.abs(nu_gt_x) > 0.01:
			print("Nu mismatch at index: %d" % i)	
			print("Expected %f, got %f" % (nu_gt_x, nu_exp_x))
			return False

	h_exp = hu_moments(thresh)
	h_gt = cv2.HuMoments(cv2.moments(thresh)).flatten()
	for i in range(len(h_gt)):
		h_gt_x = h_gt[i]
		h_exp_x = h_exp[i]
		if np.abs(h_gt_x - h_exp_x) / np.abs(h_gt_x) > 0.01:
			print("h mismatch at index: %d" % i)	
			print("Expected %f, got %f" % (h_gt_x, h_exp_x))
			return False

	return True


def test_scale_invariance():
	img = cv2.cvtColor(cv2.imread("images/test_still.png"), cv2.COLOR_RGB2GRAY)
	_, thresh = cv2.threshold(img, 127, 255, 0)
	double_size = int(img.shape[0] * 2), int(img.shape[1] * 2)
	img_big = cv2.resize(img, double_size)
	_, thresh_big = cv2.threshold(img_big, 127, 255, 0)


	def test_against_h(h_test):
		error = np.linalg.norm(h - h_test) / np.linalg.norm(h)
		if error > 0.05:
			print("Error: inaccurate h: ")
			print(h)
			print(h_test)
			print("Error: %f" % error)
			return False 
		return True

	h = hu_moments(thresh)
	h_big = hu_moments(thresh_big)
	if not test_against_h(h_big):
		print("Hu moment not scale invariant.")
		return False 

	center = thresh.shape[0] / 2, thresh.shape[1] / 2
	scale = 1.0
	degrees = 90
	R = cv2.getRotationMatrix2D(center, degrees, scale)
	thresh_rotated = cv2.warpAffine(thresh, R, thresh.shape[::-1])
	h_rotated = hu_moments(thresh_rotated)
	if not test_against_h(h_rotated):
		print("Hu moment not rotationally invariant.")
		return False 




	return True