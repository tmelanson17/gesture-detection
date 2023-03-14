import matplotlib
from matplotlib import pyplot as plt 
from matplotlib import animation
matplotlib.use('TkAgg')
import numpy as np
import itertools

class ObjectClass:
	# NOTE: Define X with each feature as a row,
	# so each variable over a set of observations is a column.
	def __init__(X):
		self._X = X 
		self._mean = np.mean(X, axis=0)
		self._std = np.mean(X, axis=0)
		self._cov = np.cov(X, rowvar=False)

def plot(feature_arrays, subplot=plt, show_plot=True):
	n_entries, n_variables = feature_arrays.shape
	x = np.arange(n_entries)*2 + 1
	w = (100 // n_variables) / 100.
	for i in range(-(n_variables // 2), np.ceil(n_variables / 2).astype('int')):
		# if i == -(n_variables // 2):
		# 	continue
		heights = feature_arrays[:, i + n_variables // 2]
		if i == 0:
			tick_label = x / 2 + 0.5
			subplot.bar(x+i*w, heights, width=w, align='center', tick_label=tick_label)
		else:
			subplot.bar(x+i*w, heights, width=w, align='center')
	if show_plot:
		subplot.show()
	else:
		return subplot

i=0
def show_label(imgs, y, actions, subplot=plt, show_plot=True):

	fig = subplot.gcf()
	im_ax = subplot.imshow(imgs[0])
	label_int = y(imgs[0])
	label = actions[label_int] if label_int is not None else "None"
	text = subplot.text(1, 8, label, fontsize=20)
	def updatefig(imgs, y, actions, img_ax, text):
		global i
		if i >= len(imgs):
			img_ax.set_array(imgs[-1])
			return [img_ax]
		img_ax.set_array(imgs[i])
		label_int = y(imgs[i])
		label = actions[label_int] if label_int is not None else "None"
		text.set_text(label)
		i+=1
		return img_ax, text
	ani = animation.FuncAnimation(fig, lambda *arg: updatefig(imgs, y, actions, im_ax, text), interval=50, blit=True)
	if show_plot:
		subplot.show()
	else:
		return subplot

	

def plot_feature_averages(X, y):
	labels = np.unique(y)
	i=0

	if X.shape[1] == 14:	
		fig, axes = plt.subplots(2, 7, sharex=False, sharey=True)
	else:
		fig, axes = plt.subplots(1, X.shape[1], sharex=False, sharey=True)

	for i in range(X.shape[1]):
		avgs = []	
		for l in labels:
			avgs.append(np.median(X[y == l, i]))
		avgs = np.array(avgs)
		if X.shape[1] == 14:
			ax = axes[i // 7, np.mod(i, 7)]
		else:
			ax = axes[0, i]
		ax.set_xticklabels([])
		subplot = plot(avgs[np.newaxis, :], ax, show_plot=False)
	plt.show()	

def plot_feature_boxplot(X, y):
	labels = np.unique(y)
	i=0

	if X.shape[1] == 14:	
		fig, axes = plt.subplots(2, 7, sharex=False, sharey=True)
	else:
		fig, axes = plt.subplots(1, X.shape[1], sharex=False, sharey=True)

	for i in range(X.shape[1]):
		vals = []
		for l in labels:
			vals.append(X[y==l, i])	
		if X.shape[1] == 14:
			ax = axes[i // 7, np.mod(i, 7)]
		else:
			ax = axes[i]
		ax.set_xticklabels([])
		subplot = ax.boxplot(vals, showfliers=False)
	plt.show()	
	
def plot_confusion_matrix(conf, classes):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]

    plt.imshow(conf, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' 
    thresh = conf.max() / 2.
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
        plt.text(j, i, format(conf[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
