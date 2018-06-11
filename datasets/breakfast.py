import os
import h5py
from numpy import array

from utils.my_io_utils import load_from_json
from utils.misc import segments_to_frame_labels


class BreakfastDataset:
    def __init__(self, dataset_path, downsampling_factor,
                 annotations_json_file):
        self.dataset_path = dataset_path
        self.downsampling_factor = downsampling_factor
        self.annotations_json_file = annotations_json_file

        self.actions = self.get_action_names
        # annotations: list of dicts with keys
        #             'video_id_num': integer identifying video
        #             'video_name': video name
        #             'feature_filename': <feat_{video_name}>.h5
        #             'segs': list of tuples (start_fr, end_fr) corresponding
        #                     to segments with a single action.
        #             'seg_labels': list of labels (integers) per segment
        #             'nb_frames': number of frames of video
        self.annotations = load_from_json(annotations_json_file)
        self.feat_dir = os.path.join(dataset_path, 'feat',
                                     'dt_l2pn_c64_pc64', 'split00')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        """

        :param item: sample index (integer)
        :return: data_sample dictionary with keys: 'video_name', 'feat',
                 'labels', 'frame_indices'
        """

        annotation = self.annotations[item]
        video_name = annotation['video_name']
        feature_filename = annotation['feature_filename']
        feat_file = os.path.join(self.feat_dir, feature_filename)

        # feat: numpy array: nb_frames x feat_dim
        # Read hdf5 file
        # dt_l2pn_c64_pc64: dense trajectories, l2 power normalization
        # 426 -> 64 dim for feature descriptor
        # x -> 64 dim for fisher vector
        with h5py.File(feat_file, 'r') as hf:
            feat = array(hf['feat'])
            frame_indices = array(hf['frame_ind'])

        feat = feat[::self.downsampling_factor, :]
        frame_indices = frame_indices[::self.downsampling_factor] - 1

        _, labels = segments_to_frame_labels(
            segments=annotation['segs'],
            segment_labels=annotation['seg_labels'])
        labels = labels[::self.downsampling_factor]

        data_sample = {'video_name': video_name,
                       'feat': feat,
                       'labels': labels,
                       'frame_indices': frame_indices,
                       }
        return data_sample

    def get_orig_labels(self, item):
        annotation = self.annotations[item]

        _, labels = segments_to_frame_labels(
            segments=annotation['segs'],
            segment_labels=annotation['seg_labels'])

        return labels

    @staticmethod
    def get_action_names():
        actions = ['SIL', 'add_saltnpepper', 'add_teabag', 'butter_pan',
                   'crack_egg', 'cut_bun', 'cut_fruit', 'cut_orange',
                   'fry_egg', 'fry_pancake', 'peel_fruit',
                   'pour_cereals', 'pour_coffee', 'pour_dough2pan',
                   'pour_egg2pan', 'pour_flour', 'pour_juice', 'pour_milk',
                   'pour_oil', 'pour_sugar', 'pour_water',
                   'put_bunTogether',
                   'put_egg2plate', 'put_fruit2bowl', 'put_pancake2plate',
                   'put_toppingOnTop', 'smear_butter', 'spoon_flour',
                   'spoon_powder', 'spoon_sugar', 'squeeze_orange',
                   'stir_cereals', 'stir_coffee', 'stir_dough', 'stir_egg',
                   'stir_fruit', 'stir_milk', 'stir_tea', 'stirfry_egg',
                   'take_bowl', 'take_butter', 'take_cup', 'take_eggs',
                   'take_glass', 'take_knife', 'take_plate',
                   'take_squeezer',
                   'take_topping']
        return actions
