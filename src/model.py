import numpy as np 
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from src.dataset import Dataset
from src.extraction import extract_data, convert_MHI_to_training
from src.mhi import MHI
import cv2

class Trainer:
	def __init__(self, image_dir, tau=20, get_data_on_init=True):
		self._data_files = Dataset(image_dir)	
		self._tau = tau
		self._train_indices = [13, 14 , 11, 12]
		self._val_indices = [19, 20] # , 3, 5] #, 6, 7, 8, 9, 10]
		if get_data_on_init:
			self.get_data()
		# self._X_test, self._y_test, self._im_test = self.extract_data(self._test_indices)
		self._classifier = KNeighborsClassifier(49)
		self._mhi = MHI(self._tau)

	def get_data(self):
		self._X_train, self._y_train, self._im_train, _ = extract_data(self._data_files, self._train_indices, tau=self._tau)
		self._X_val, self._y_val, self._im_val, _ = extract_data(self._data_files, self._val_indices, tau=self._tau)

	def train(self):
		self._classifier.fit(self._X_train, self._y_train)
		return self._classifier.score(self._X_train, self._y_train)

	def validate(self):
		return self._classifier.score(self._X_val, self._y_val)

	def get_confusion_matrix(self):
		y_pred = self._classifier.predict(self._X_val)
		conf = confusion_matrix(self._y_val, y_pred)
		return conf 

	def reset_predictor(self):
		self._mhis = []
		self._indices_actions = []
		for tau in [12, 25, 30]:
			self._mhis.append(MHI(tau))
		if isinstance(self._tau, dict):
			for label in self._tau:
				if self._tau[label] == 15:
					self._indices_actions.append(0)
				elif self._tau[label] == 30:
					self._indices_actions.append(2)
				else:
					self._indices_actions.append(1)
		self._indices_actions = np.array(self._indices_actions, dtype=np.int)
		# Add a majority filter
		self._last_5 = np.array([0] * 5)
		self._l5_idx = 0

	def predict(self, image):
		probs = []
		for mhi in self._mhis:
			H = mhi.add_image(image)
			if H is None:
				continue
			MEI = np.where(H > 0, 255, 0)
			H_moments = convert_MHI_to_training(H)
			MEI_moments = convert_MHI_to_training(MEI)
			feature = np.concatenate([H_moments, MEI_moments])
			probs.append(self._classifier.predict_proba(feature.reshape(1,-1)))
		if len(probs) < 3:
			return None
		probs = np.array(probs)
		candidates = [probs[i, j] for i,j in zip(self._indices_actions, np.arange(probs.shape[1]))]
		print(candidates)
		label = np.argmax(candidates)
		self._last_5[self._l5_idx] = label 
		self._l5_idx = np.mod(self._l5_idx + 1, 5)
		print(label, np.argmax(np.bincount(self._last_5)))
		return np.argmax(np.bincount(self._last_5)) #int(self._classifier.predict(feature.reshape(1,-1)))


