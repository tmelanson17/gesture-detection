from test import IO, MHI, HU
from src.model import Trainer
import src.extraction
import pickle
import os
import sklearn
import numpy  as np
from src import dataviz
from matplotlib import pyplot as plt

data_dir="images/action_dataset"
print("Testing IO:")
# IO.test_image_collection()
if not IO.test_file_decompose() or not IO.test_dataset_creation():
	print("Image collection fails.")
else:
	print("Image collection passes.")
# if not MHI.write_mhi_output() or not MHI.test_mhi():
# 	print("MHI processing fails.")
# else:
# 	print("MHI processing passes.")

if HU.test_M_all_ones() and \
	HU.test_moments_against_cv2() and \
	HU.test_scale_invariance():
	print("Hu testing passes.")
else:
	print("Hu testing fails.")

pickle_file = 'untrained_test_final.pkl'
if os.path.exists(pickle_file):
	t = pickle.load(open(pickle_file, 'rb'))
	# t._classifier = sklearn.svm.SVC(C=1.0, gamma='scale', kernel='rbf', probability=True)
	# t._X_train /=  1e-3 
	# t._X_val /=  1e-3
	t._classifier = sklearn.neighbors.KNeighborsClassifier(49)
else:
	t = Trainer(data_dir)
	pickle.dump(t, open(pickle_file, 'wb'))
print("Training results:")
print(t.train())
print(t.validate())
print("Confusion matrix")
print(t.get_confusion_matrix())

pickle.dump(t, open('trained_test_final.pkl', 'wb'))

pickle_file = 'trained_multiple_tau_adjusted.pkl'
svm=True
if os.path.exists(pickle_file):
	multiple_tau = pickle.load(open(pickle_file, 'rb'))
	if svm:
		multiple_tau._classifier = sklearn.svm.SVC(C=10.0, probability=True)
		multiple_tau.train()
else:
	tau_dict = {'running': 30, 'handwaving': 30, 'walking': 22, 'jogging': 25, 'boxing': 15, 'handclapping': 25}
	multiple_tau = Trainer(data_dir, tau_dict)
	print(multiple_tau.train())
	# pickle.dump(multiple_tau, open(pickle_file, 'wb'))
print(multiple_tau.validate())
dataviz.plot_confusion_matrix(multiple_tau.get_confusion_matrix(), multiple_tau._data_files.actions)
plt.show()

multiple_tau._classifier = sklearn.svm.SVC(C=10.0, probability=True)
multiple_tau.train()

# multiple_tau = pickle.load(open(pickle_file, 'rb'))

print(multiple_tau.validate())
dataviz.plot_confusion_matrix(multiple_tau.get_confusion_matrix(), multiple_tau._data_files.actions)
plt.show()
pickle.dump(multiple_tau._classifier, open(pickle_file, 'wb'))
