import numpy as np 
from cv2 import morphologyEx, MORPH_OPEN, MORPH_CLOSE
import cv2 


def D(cur, prev, theta=0.03):
	return np.logical_or(cv2.subtract(cur, prev) > theta,
			cv2.subtract(prev, cur) > theta)

MINIMUM_AREA=3000
MINIMUM_CURRENT_ACTION_PIXELS=5
def valid(H, MEI):
	return np.sum(MEI) < MINIMUM_AREA or np.sum(H == 255) < MINIMUM_CURRENT_ACTION_PIXELS

class MHI:
	def __init__(self, tau=-1, filename=None, theta=0.03, use_open=True, use_close=True):
		self._tau = tau
		self._D = []
		self._H = []
		self._last_image = None
		self._theta = theta
		self._use_open = use_open
		self._use_close = use_close

	def process(self, images):
		if len(images) < 2:
			raise ValueError("MHI must be given more than 1 image for MHI.")

		tau = self._tau if self._tau != -1 else len(images)
		if len(images) < tau:
			raise ValueError("Sequence length must be at least tau.")

		self._H = []
		for end in range(tau-1, len(images), 2):
			H = np.zeros_like(images[0])
			begin = end - tau + 1
			no_action=False
			for i in range(begin, end):
				H = np.maximum(H - 1, 0)
				D_curr = D(images[i], images[i-1], theta=self._theta)
				D_curr = morphologyEx(np.uint8(D_curr*255), MORPH_CLOSE, 255*np.ones([3,3])) > 0
				kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
				D_curr = morphologyEx(np.uint8(D_curr*255), MORPH_OPEN, kernel) > 0
				if np.sum(D_curr > 0) < MINIMUM_CURRENT_ACTION_PIXELS:
					no_action=True
					break
				H[D_curr] = tau 
			if no_action:
				continue
			self._H.append(H)
		return self.get_H_sequence() 

	def add_image(self, image):
		if self._last_image is None:
			self._last_image = np.copy(image)
			return
		tmp_img = np.copy(image)
		D_curr = D(tmp_img, self._last_image, theta=self._theta)
		if self._use_close:
			D_curr = morphologyEx(np.uint8(D_curr*255), MORPH_CLOSE, 255*np.ones([3,3])) > 0
		if self._use_open:
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
			D_curr = morphologyEx(np.uint8(D_curr*255), MORPH_OPEN, kernel) > 0
		if np.sum(D_curr > 0) < MINIMUM_CURRENT_ACTION_PIXELS:
			self._last_image = None
			return
		if len(self._H) != 0:
			H = np.maximum(self._H[-1] - 1, 0)
		else:
			H = np.zeros_like(image)
		tau = self._tau 
		H[D_curr] = tau
		self._H.append(H)
		self._last_image = np.copy(tmp_img)
		return H.astype(np.float) * (255. / tau)

	def get_H_sequence(self):
		if len(self._H) == 0:
			print("Warning: MHI not created.")
			return None 

		# H_normalized = self._H.astype(np.float) / np.max(self._H)
		H_out = [H.astype(np.float) * (255. / self._tau) for H in self._H]
		return H_out

	def get_MEI_sequence(self):
		if len(self._H) == 0:
			print("Warning: MHI not created.")
			return None 

		return [np.where(H > 0, 255, 0)	 for H in self._H]



# 
