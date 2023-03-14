import os

# TODO: Actually limit n_actions in a reasonable way (pure max limiting may lead
#		to different actions per person)

'''
	Specifically extracts the person number from the a file in 
	the Action Database (i.e. person[xx]_[action]_[d]_uncomp.avi, where 
	xx = index, action = action performed, d = video image edit)
'''
def decompose_file(filename):
	components = os.path.splitext(filename)[0].split('_')

	len_prefix = len("person")
	idx_string = components[0][len_prefix:len_prefix+2]

	action = components[1]

	dval = components[2]

	return int(idx_string), action, dval

'''
	Organizes images into needed files
'''
class Dataset:
	def __init__(self, dataset_dir = None, max_people=None, max_actions=None):
		self.actions = []
		self.images = dict()
		self.n_actions = 0
		self._max_people = max_people
		if max_actions is not None:
			print("Warning: maximum action count not supported")
		if dataset_dir is not None:
			self.open(dataset_dir)


	''' 
		Populates the image database with all the images 
		found in the given database. Assumes data is in the format 
		of the Action Database.

		\param[in] dataset_dir Readable path of the image dataset.
	''' 
	def open(self, dataset_dir):

		for act in os.listdir(dataset_dir):
			act_full_path = os.path.join(dataset_dir, act)
			if os.path.isdir(act_full_path):
				self.actions.append(act)
				for avi in os.listdir(act_full_path):
					suffix = os.path.splitext(avi)[1]
					if  suffix != ".avi":
						raise IOError("Files must be .avi, not " + suffix)
					idx, file_action, dval = decompose_file(avi)

					if self._max_people and idx > self._max_people:
						continue

					if act != file_action:
						raise IOError("Action %s found in directory of %s" % (file_action, act))

					if idx not in self.images:
						self.images[idx] = dict()

					if act not in self.images[idx]: 
						self.images[idx][act] = dict()

					self.images[idx][act][dval] = os.path.join(act_full_path, avi)

		# Update actions count
		self.n_actions = len(self.actions)
		print("N Actions")
		print(self.n_actions)

	def get(self, person_idx, action, dval='d1', full_path=True):
		filename = self.images[person_idx][action][dval]
		if not full_path:
			return os.path.basename(filename).split('.')[0]
		return filename

	def create_video_file(self, person_idx, action, dval='d1', video_filename='video'):
		return os.path.join(str(person_idx), action, dval, video_filename + '.avi')

	def get_labels(self):
		out_dict = dict()
		for i, l in enumerate(self.actions):
			out_dict[l] = i 
		return out_dict


