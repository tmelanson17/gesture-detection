from src import dataset, dataviz, humoments, image_io, mhi, extraction,  model
import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


data_dir="images/action_dataset"
pickle_file='data_viz_no_close_05.pkl'
data = dataset.Dataset(data_dir)

def plot_action_features():
	data = dataset.Dataset(data_dir)
	moments = []
	for act in data.actions:
		mhi_extractor = mhi.MHI()
		images_action = image_io.get_image_collection(data.get(1, act))
		images_H = mhi_extractor.process(images_action)[-1]
		cv2.imshow("H value for %s" % act, images_H / 255.)
		print("Min H", np.min(images_H))
		print("Max H", np.max(images_H))
		images_moments = humoments.hu_moments(images_H)
		moments.append(np.array(images_moments))
	moments = np.vstack(moments)

	dataviz.plot(moments)
	cv2.waitKey(0)

i=0
image_sets = []
img_ax = [None]*6
pickle_file='H_no_blur_no_open.pkl'
# optional arguments: blur, use_open, use_close, theta, tau
def display_mhi():
	global image_sets, img_ax, data

	for act in data.actions:
		images = image_io.get_image_collection(data.get(13, act), blur=True) 
		mhi_extractor = mhi.MHI(tau=20, theta=0.1, use_open=True, use_close=True)
		H_values = []
		for img in images:
			H_values.append(mhi_extractor.add_image(img))
		image_sets.append(H_values)

	# images = np.array(images)
	fig, axes = plt.subplots(2, 3, sharex=False, sharey=True)
	for j, act in enumerate(data.actions):
		ax = axes[j // 3, np.mod(j, 3)]
		ax.set_title(act)
		idx = 10
		img = np.zeros([120, 160]) if image_sets[j][idx] is None else image_sets[j][idx] / 255.
		img_ax[j] = ax.imshow(img)

	fig.show()
	def updatefig(*args):
	    global image_sets, data, img_ax, i
	    i+=1
	    for j, act in enumerate(data.actions):
	    	idx = np.minimum(i, len(image_sets[j])-1)
	    	# idx = 30
	    	# img = np.zeros([120, 160]) if image_sets[j][idx] is None else image_sets[j][idx] / 255.
	    	if image_sets[j][idx] is None:
	    		continue
	    	img_ax[j].set_array(image_sets[j][idx] / 255.)
	    return img_ax[0], img_ax[1], img_ax[2], img_ax[3], img_ax[4], img_ax[5]

	ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True)
	plt.show()

def display_mei_mhi():
	global image_sets, img_ax, data

	images = image_io.get_image_collection(data.get(13, 'boxing'), blur=True) 
	mhi_extractor = mhi.MHI(tau=20, theta=0.04, use_open=True, use_close=True)
	H_values = []
	for img in images:
		H = mhi_extractor.add_image(img)
		if H is not None:
			H_values.append(H)

	# images = np.array(images)
	fig, axes = plt.subplots(1, 2, sharex=False, sharey=True)
	idx = 30
	img_ax[0] = axes[0].imshow(H_values[idx] / 255.)
	img_ax[1] = axes[1].imshow(np.where(H_values[idx] > 0, 255, 0))

	fig.show()
	def updatefig(*args):
	    global image_sets, data, img_ax, i
	    i+=1
	    idx = np.minimum(i, len(H_values)-1)
	    img_ax[0].set_array(H_values[idx] / 255.)
	    img_ax[1].set_array(np.where(H_values[idx] > 0, 255, 0))
	    return img_ax[0], img_ax[1]

	ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True)
	plt.show()
# Plots a graph of the mean value of each feature for each label
def plot_features():
	if not os.path.exists(pickle_file):
		moments = []
		X, y, images, _= extraction.extract_data(data, [13, 14, 15, 19, 20])
		pickle.dump({'X': X, 'y': y, "images": images}, open(pickle_file, 'wb'))
	else:
		d = pickle.load(open(pickle_file, 'rb'))
		X = d['X']
		y = d['y']
		images = d['images']

	print("Data labels:")
	print(data.get_labels())
	dataviz.plot_feature_averages(X, y)
	dataviz.plot_feature_boxplot(X, y)

def display_labelled_images():
	# pickle_file='labelled_images_13.pkl'
	# if not os.path.exists(pickle_file):
	# 	moments = []
	# 	X, y, Hs, images = extraction.extract_data(data, [13])
	# 	pickle.dump({'X': X, 'y': y, "H": Hs, "images": images}, open(pickle_file, 'wb'))
	# else:
	# 	d = pickle.load(open(pickle_file, 'rb'))
	# 	X = d['X']
	# 	y = d['y']
	# 	images = d['images']
	idx = 8 
	print("This works, rihgt?")
	tau_dict = {'running': 30, 'handwaving': 30, 'walking': 22, 'jogging': 25, 'boxing': 15, 'handclapping': 25}
	trainer = model.Trainer(data_dir, tau=tau_dict, get_data_on_init=False)
	print("Ok, got here")
	trainer._classifier = pickle.load(open('trained_multiple_tau_adjusted.pkl', 'rb'))
	# Change this for every action
	act = data.actions[5]
	print("Loaded classifier")
	images = image_io.get_image_collection(data.get(idx, act))
	trainer.reset_predictor()
	print("Going into labels:")
	dataviz.show_label(images, lambda im: trainer.predict(im), data.actions, show_plot=True) 
	# plot.show()


def compare_moments():
	# Pick random indices from the training data
	pickle_file = "compare_moments.pkl"
	if not os.path.exists(pickle_file):
		indices = [13]
		X_values = []
		y_values = []
		labels = data.get_labels()
		for idx in indices:
			image_database = pickle.load(open('image_sequences_%d.pkl' % idx, 'rb'))
			for act in image_database['image_seqs']:
				images_action = image_database['image_seqs'][act]
				for dval in images_action:
					mhi_obj = mhi.MHI()
					images_dval = images_action[dval]
					for img in images_dval:
						mhi_obj.add_image(img)
					H = mhi_obj.get_H_sequence()
					# MEI = mhi.get_MEI_sequence()
					H_moments = humoments.hu_moments(np.array(H))
					H_labels = np.array([labels[act]] * H_moments.shape[1])
					X_values.append(H_moments.T)
					y_values.append(H_labels)
		X = np.vstack(X_values)

		X_sign_log = extraction.signed_log(X.T).T
		X_sign_root = extraction.signed_roots(X.T).T
		X /= np.mean(X, axis=1, keepdims=True)
		print("X shape")
		print(X.shape)

		y = np.concatenate(y_values)
		pickle.dump({"X": X, "X_sign_log": X_sign_log, "X_sign_root": X_sign_root, "y": y}, open(pickle_file, 'wb'))
	else:
		d = pickle.load(open(pickle_file, 'rb'))
		X = d['X']
		X_sign_log = d['X_sign_log']
		X_sign_root = d['X_sign_root']
		y = d['y']

	print("Shapes:")
	print(X.shape)
	print(y.shape)
	dataviz.plot_feature_boxplot(X, y)
	dataviz.plot_feature_boxplot(X_sign_log, y)
	dataviz.plot_feature_boxplot(X_sign_root, y)

	for X_in in [X, X_sign_log, X_sign_root]:
		X_train, X_test, y_train, y_test = train_test_split(X_in, y, test_size = 0.2)
		parameters = {'C':[1, 10, 100, 1000]}
		svm = SVC()
		cv = GridSearchCV(svm, parameters)
		cv.fit(X_train, y_train)
		print("Best Params")
		print(cv.best_params_)
		print("CV score")
		print(cv.score(X_test, y_test))
		print("Confusion matrix")
		y_pred = cv.predict(X_test)
		conf = confusion_matrix(y_test, y_pred)
		dataviz.plot_confusion_matrix(conf, data.actions)
		plt.show()

def best_tau_per_action():
	large_tau=50
	H = []
	y= []
	for i in [13, 14]:
		for j, act in enumerate(data.actions):
			images = image_io.get_image_collection(data.get(i, act), blur=True) 
			mhi_extractor = mhi.MHI(tau=large_tau, theta=0.04, use_open=True, use_close=True)
			H_values = []
			for img in images:
				mhi_extractor.add_image(img)
			H_values = mhi_extractor.get_H_sequence()
			H += H_values
			y+= [j] * len(H_values)

	H = np.array(H)
	y= np.array(y)
	pickle_file = 'best_tau.pkl'

	def test_H(MHI, y, tau):
		one_iter = 255. / large_tau
		H_normal =  MHI - (large_tau - tau) * one_iter
		H_normal = np.maximum(H_normal, 0) * 255. / tau
		MEI_normal = np.where(H_normal > 0, 255, 0)
		hu_h_moments = extraction.signed_roots(humoments.hu_moments(H_normal)).T
		hu_mei_moments = extraction.signed_roots(humoments.hu_moments(MEI_normal)).T
		X = np.hstack([hu_h_moments, hu_mei_moments])
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
		svm = SVC()
		svm.fit(X_train, y_train)
		scores = [None] * len(data.actions)
		for l in range(np.max(y)+1):
			scores[l] = svm.score(X_test[y_test==l], y_test[y_test==l])
		return scores

	score_matrix = []
	for j in range(1, 30):
		print(j)
		score_matrix.append(test_H(H, y, j))
	score_matrix = np.vstack(score_matrix)
	print("Score matrix")
	for row in score_matrix:
		print("%.03f %.03f %.03f %.03f %.03f %0.03f" % (row[0], row[1], row[2], row[3], row[4], row[5]))
	print(score_matrix)



if __name__ == "__main__":
	plot_features()
	display_mhi()
	display_mei_mhi()
	display_labelled_images()
	plot_action_features()
	compare_moments()
	best_tau_per_action()