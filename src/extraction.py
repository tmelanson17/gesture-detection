from src.dataset import Dataset
from src.humoments import hu_moments
from src.image_io import get_image_collection
from src.mhi import MHI, valid

import numpy as np 

def convert_MHI_to_training(H):
	hu = hu_moments(H)
	sign = np.sign(hu)
	hu_abs = np.abs(hu)
	hu_floored = np.maximum(hu_abs, 1e-35)
	hu_scaled = sign*np.log(hu_floored)
	# hu_scaled = sign*np.array([
	# 	hu_floored[0],
	# 	np.power(hu_floored[1], 1/2.),
	# 	np.power(hu_floored[2], 1/3.),
	# 	np.power(hu_floored[3], 1/3.),
	# 	np.power(hu_floored[4], 1/5.),
	# 	np.power(hu_floored[5], 1/4.),
	# 	np.power(hu_floored[6], 1/6.)])
	 
	return hu_scaled 

def signed_roots(hu):
	sign = np.sign(hu)
	hu_abs = np.abs(hu)
	hu_floored = np.maximum(hu_abs, 1e-35)
	hu_scaled = sign*np.vstack([
		hu_floored[0],
		np.power(hu_floored[1], 1/2.),
		np.power(hu_floored[2], 1/3.),
		np.power(hu_floored[3], 1/3.),
		np.power(hu_floored[4], 1/6.),
		np.power(hu_floored[5], 1/4.),
		np.power(hu_floored[6], 1/6.)])
	 
	return hu_scaled / 1e-3

def signed_log(hu):
	sign = np.sign(hu)
	hu_abs = np.abs(hu+1e-20)
	hu_floored = np.maximum(hu_abs, 1e-30)
	hu_scaled = sign*np.log(hu_floored)
	return hu_scaled


def dataset_equil(X, y, H_values, img_values):
	counts = [np.sum(y==l) for l in np.unique(y)]
	counts.sort()
	max_allowed = (np.min(counts)*3)// 2
	X_equil = []
	y_equil = []
	H_equil = []
	img_equil = []
	for label in np.unique(y):
		X_label = X[y == label]
		y_label = y[y == label]
		img_label = img_values[y == label]
		H_label = H_values[y == label]
		indices = np.random.choice(len(y_label), max_allowed)
		X_equil.append(X_label[indices])
		y_equil += y_label[indices].tolist()
		img_equil += img_label[indices].tolist()
		H_equil += H_label[indices].tolist()
	return X_equil, y_equil, H_equil, img_equil

def extract_data(data_files, indices, tau=20, theta=0.04, use_all_dval=True):
	X_values = []
	y_values = []
	labels = data_files.get_labels()
	H_values = []
	img_values = []
	for idx in indices:
		X_values_i = []
		y_values_i = []
		H_values_i = []
		img_values_i = []
		for act in data_files.actions:
			for dval in data_files.images[idx][act]:
				if dval != 'd3' and not use_all_dval:
					continue
				images = get_image_collection(data_files.get(idx, act, dval))
				filename = data_files.get(idx, act, dval, full_path=False)
				if isinstance(tau, dict):
					mhi_t = MHI(tau=tau[act], theta=theta, use_open=True, use_close=True)
				else:
					mhi_t = MHI(tau=tau, theta=theta, use_open=True, use_close=True)
				H_images = []
				extracted_images = []
				for img in images:
					H = mhi_t.add_image(img)
					if H is not None:
						H_images.append(H)
						extracted_images.append(img)
				MEIs = mhi_t.get_MEI_sequence()
				for H, MEI, image in zip(H_images, MEIs, extracted_images):
					# if valid(H, MEI):
					# 	continue
					H_values_i.append(np.copy(H))
					img_values_i.append(np.copy(image))
					hu_mhi = convert_MHI_to_training(H)
					hu_mei = convert_MHI_to_training(MEI)
					X_values_i.append(np.concatenate([hu_mhi, hu_mei]))
					y_values_i.append(labels[act])
		X_unequil = np.vstack(X_values_i)
		y_unequil = np.array(y_values_i)
		H_unequil = np.array(H_values_i)
		img_unequil = np.array(img_values_i)
		X, y, Hs, imgs = dataset_equil(X_unequil, y_unequil, H_unequil, img_unequil)
		X_values += X 
		y_values += y 
		H_values += Hs	
		print("number of H values")
		print(len(H_values))
		img_values += imgs
	X = np.vstack(X_values)
	y = np.array(y_values)
	Hs = np.array(H_values)
	print("Size of Hs")
	print(Hs.shape)
	imgs = np.array(img_values)
	print("X shape", X.shape)
	print("y shape", y.shape)
	return X, y, Hs, imgs

