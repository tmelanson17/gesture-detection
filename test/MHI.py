from src.mhi import MHI
from src.dataset import Dataset
from src.image_io import get_image_collection, write_video_output

import cv2
import numpy as np

data_dir="images/action_dataset"
output_dir="results/"

def test_mhi():
	data_files = Dataset(data_dir)

	images_handwaving = get_image_collection(data_files.get(1, "handwaving"))
	mhi = MHI()

	for img in images_handwaving:
		mhi.add_image(img)

	H = mhi.get_H_sequence()[-1]
	if not np.any(H < 1.0):
		print("Error: H is all ones.")
		return False
	if not np.any(H > 0.0):
		print("Error: H is all zeros.")
		return False

	# print("Min MHI:")
	# print(np.min(H))
	# print("Max MHI:")
	# print(np.max(H))
	# print("Mean")
	# print(np.mean(H))

	cv2.imwrite(output_dir + "H_" + "handwaving_" + "1.png", H)

	for action in ["handclapping", "walking"]:
		for t in range(2,50):
			images_handwaving = get_image_collection(data_files.get(1, action))
			mhi_t = MHI(tau=t)
			for img in images_handwaving:
				mhi_t.add_image(img)
			H = mhi_t.get_H_sequence()[-1]
			cv2.imwrite(output_dir + action + "/" + "H_" + "handwaving_" + "1_" + str(t) + ".png", H)

	for action in ["handwaving", "running", "boxing"]:
		images = get_image_collection(data_files.get(1, action))
		mhi_t = MHI()
		for img in images_handwaving:
			mhi_t.add_image(img)
		H = mhi_t.get_H_sequence()[-1]
		cv2.imwrite(output_dir + "H_" + action + "_" + "1.png", H)

	return True

def write_mhi_output():
	data_files = Dataset(data_dir)

	counter=0
	for idx in data_files.images:
		for act in data_files.images[idx]:
			print(act)
			for dval in data_files.images[idx][act]:
				images = get_image_collection(data_files.get(idx, act, dval))
				filename = data_files.get(idx, act, dval, full_path=False)
				mhi_t = MHI(tau=20)
				for img in images:
					mhi_t.add_image(img)
				H_images = mhi_t.get_H_sequence()
				# for H in H_images:
					# cv2.imshow('H', np.uint8(H))
					# cv2.waitKey(100)
				write_video_output(H_images, "results/train_data/" + data_files.create_video_file(
						idx, act, dval, filename))

				images = None 
				H_images = None
		counter += 1
		if counter > 2:
			break
	return True
