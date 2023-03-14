from src.image_io import preprocess, get_image_collection
from src.dataset import Dataset, decompose_file
import cv2

data_dir="images/action_dataset"

def test_image_collection():
	video_file = "images/handclapping/person01_handclapping_d1_uncomp.avi"
	images = get_image_collection(video_file)
	if len(images) <= 1:
		raise IOError("Images not created")

	if len(images[0].shape) > 2:
		raise IOError("Image not grayscale")

	return True

def test_file_decompose():
	for i in range(100):
		idx, action, dval = decompose_file("person%02d_handclapping_d1_uncomp.avi" % i)
		if idx != i: 
			print("Expected %02d. Got index %02d" % (i, idx))
			return False

		if action not in ("walking", "jogging", 'running', 'boxing', 'handwaving', 'handclapping'):
			print("Unknown action: " + action)
			return False

		if dval not in ('d1', 'd2', 'd3', 'd4'):
			print("Unknown dval: " + dval)
			return False

	print("File decompose passed.")
	return True

def test_dataset_creation():
	n_people=10
	data = Dataset(max_people=n_people)
	data.open(data_dir)

	if data.n_actions != 6:
		print("Expected 6 actions, got %d" % (data.n_actions))
		return False

	if len(data.images.keys()) != n_people:
		print("Expected %d people, got %d" % (n_people, len(data.images.keys())))
		return False

	if data.n_actions != len(data.images[1].keys()):
		print("Image actions not parsed correctly")
		return False

	if 4 != len(data.images[1]['walking']):
		print("Must have 4 image sets per action per person.")
		return False

	print("Dataset creation passed.")
	return True