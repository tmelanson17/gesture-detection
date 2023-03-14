from src.image_io import get_image_collection
from src.extraction import extract_data
from src.dataset import Dataset 

import pickle

data = Dataset("images/first_dataset")
global THETA
# for THETA in [0.03, 0.05, 0.1]:
# 	for tau in [50]:
# 		for i in range(1, 26):
# 			if i==2:
# 				continue
# 			idxs = [i]
# 			# X, y, H_values, imgs, lens = extract_data(data, idxs, tau=tau, theta=THETA)
# 			# pickle.dump({'X': X, 'y': y, 'H': H_values, 'imgs':imgs}, open('total_data_theta_%d_tau_%d_%d.pkl' % (int(THETA*100), tau, idxs[0]), 'wb'))
# 			image_seqs = dict()
# 			for act in data.actions:
# 				image_seqs[act] = dict()
# 				for dval in data.images[i][act]:
# 					image_seqs[act][dval] = get_image_collection(data.get(i,act,dval))
# 			pickle.dump("image_seqs": image_seqs, open('image_sequences_%d.pkl' % i, 'wb'))
# 			X = None 
# 			y = None 
# 			imgs = None
# 			H_values = None
# THETA=0.01
# tau=10
for i in range(1, 26):
	idxs = [i]
	# X, y, H_values, imgs, lens = extract_data(data, idxs, tau=tau, theta=THETA)
	# pickle.dump({'X': X, 'y': y, 'H': H_values, 'imgs':imgs}, open('total_data_theta_%d_tau_%d_%d.pkl' % (int(THETA*100), tau, idxs[0]), 'wb'))
	image_seqs = dict()
	for act in data.actions:
		image_seqs[act] = dict()
		for dval in data.images[i][act]:
			image_seqs[act][dval] = get_image_collection(data.get(i,act,dval))
	pickle.dump({"image_seqs": image_seqs}, open('image_sequences_%d.pkl' % i, 'wb'))
