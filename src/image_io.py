import cv2
import numpy as np
import os
import imageio

from cv2 import VideoCapture, VideoWriter

# TODO: Include a size so that an input image can be resized


'''
	Preprocesses an image so it can be used for training / testing.

	\param[in] image Input image to be processed

	\return Image ready for input into the algorithm
'''
def preprocess(image, blur=True):
	tmp_img = np.copy(image)

	# Convert to grayscale if not currently grayscale
	if len(tmp_img.shape) > 2:
		tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2GRAY)

	tmp_img = tmp_img.astype(np.float) / 255.
	if blur:
		tmp_img = cv2.GaussianBlur(tmp_img, (3,3), 1)

	return tmp_img


'''
	Converts a video file into an array of images.

	\param[in] file Video file input.

	\return Array of images collected in order from file.
'''
def get_image_collection(file, blur=True):
	reader = imageio.get_reader(file)

	images_out = []
	for img in reader:
		images_out.append(preprocess(img, blur=blur))

	return images_out

'''
	Writes array of images into a video file

	\param[in] images Array of images to be exported into video

	\param[in] file Video file (with extension) to export images

	\param[in] fps FPS of output (default is 10 fps)
'''
def write_video_output(images, file, fps=10):
	if len(images) == 0:
		return
	dirname = os.path.dirname(file)
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	frame_width, frame_height = images[0].shape
	# # Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file. 
	# out = cv2.VideoWriter(file,
	# 		cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
	# 		10, 
	# 		(frame_width,frame_height),
	# 		False)

	writer = imageio.get_writer(file, fps=10)

	for img in images:
		# out.write(np.uint8(img))
		writer.append_data(np.uint8(img))
	images = None
	# out.release()
	writer.close()




