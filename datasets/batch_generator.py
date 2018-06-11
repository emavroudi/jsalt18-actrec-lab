from numpy import random, arange, ceil, append, array, amax
import logging

logging.basicConfig(level=logging.DEBUG)


class FrameSequenceBatchGenerator:
    def __init__(self, batch_size, dataset_obj,
                 preprocessor_obj, nb_classes,
                 shuffle=True, seed=42):
        """
        Batch generator class for getting batches of sequences.
        :param batch_size: batch size
        :param dataset_obj: instance of Dataset class,
                            implements __len__ and __getitem__
        :param preprocessor_obj: instance of Preprocessor object that
                                 has methods:
                set_max_nb_frames(): sets max_nb_frames (after downsampling)
                preprocess(): function that takes features_lst
                              and frame_labels_lst
                              as arguments and returns
                              feat, frame_labels, frame_sample_weights,
                              frame_sequence_lengths
        :param nb_classes: number of classes
        :param shuffle: whether to shuffle batches after seeing all data
        :param seed: numpy random seed
        """

        self.seed = seed
        random.seed(self.seed)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_obj = dataset_obj
        self.nb_classes = nb_classes
        self.batch_id = 0
        self.preprocessor_obj = preprocessor_obj

        nb_samples = self.dataset_obj.__len__()
        self.batches = self._make_batches(nb_samples, self.batch_size)

        # Get max_nb_frames, max_nb_segs and configure preprocessing
        self.configure_preprocessing()

    def __len__(self):
        nb_samples = self.dataset_obj.__len__()
        return nb_samples

    @property
    def steps(self):
        batch_count = int(ceil(self.__len__() / self.batch_size))
        return batch_count

    def batch_size(self):
        return self.batch_size

    def get_feat_dim(self):
        feat = self.dataset_obj[0]['feat']
        feat_dim = feat.shape[1]
        return feat_dim

    def get_frame_sequence_lengths(self):
        sequence_lengths = []
        for sample_ind in range(self.__len__()):
            feat = self.dataset_obj[sample_ind]['feat']
            sequence_lengths.append(feat.shape[0])
        return array(sequence_lengths)

    def get_max_nb_frames(self):
        frame_sequence_lengths = self.get_frame_sequence_lengths()
        max_nb_frames = amax(frame_sequence_lengths)
        return max_nb_frames

    def configure_preprocessing(self, _max_nb_frames=None, _max_nb_segs=None):
        if _max_nb_frames is None:
            max_nb_frames = self.get_max_nb_frames()
        else:
            max_nb_frames = _max_nb_frames

        self.preprocessor_obj.set_max_nb_frames(max_nb_frames)
        self.preprocessing_func = self.preprocessor_obj.preprocess

    def _shuffle_sample_indices(self):
        nb_samples = self.__len__()
        index_array = arange(nb_samples)
        index_array = self._batch_shuffle(index_array, self.batch_size)
        return index_array

    def __next__(self):
        """
        Return a batch of data. (X, frame_labels, frame_sample_weights,
                                 frame_sequence_lengths)
        When all data have been used, start over.
        :return:
        """
        data_indices = arange(self.__len__())
        if self.batch_id == self.steps:
            # Iterate over data from scratch, optionally reshuffling them
            self.batch_id = 0
            if self.shuffle:
                shuffled_indices = self._shuffle_sample_indices()
                data_indices = shuffled_indices

        batch_index = self.batch_id
        (batch_start, batch_end) = self.batches[batch_index]

        # TODO: refactor this, as it is it loads and preprocesses
        # each sequence at every epoch
        batch_indices = arange(batch_start, batch_end)
        data_indices = array(data_indices)[batch_indices]

        # Get features and ground truth labels
        features_lst = []
        frame_labels_lst = []
        for data_ind in data_indices:
            data_sample = self.dataset_obj[data_ind]
            feat = data_sample['feat']
            labels = data_sample['labels']
            features_lst.append(feat)
            frame_labels_lst.append(labels)

        x, frame_labels, frame_sample_weights, frame_sequence_lengths = \
            self.preprocessing_func(features_lst, frame_labels_lst)

        self.batch_id += 1

        return x, frame_labels, frame_sample_weights,\
            frame_sequence_lengths

    @staticmethod
    def _batch_shuffle(index_array, batch_size):
        """
        This shuffles an array in a batch-wise fashion.
        :param index_array:
        :param batch_size
        :return:
        """
        batch_count = int(len(index_array) / batch_size)
        # to reshape we need to be cleanly divisible by batch size
        # we stash extra items and reappend them after shuffling
        last_batch = index_array[batch_count * batch_size:]
        index_array = index_array[:batch_count * batch_size]
        index_array = index_array.reshape((batch_count, batch_size))
        random.shuffle(index_array)
        index_array = index_array.flatten()
        return append(index_array, last_batch)

    @staticmethod
    def _make_batches(size, batch_size):
        """
        Returns a list of batch indices (tuples of indices).
        :param size
        :param batch_size
        """
        # nb_batch = int(ceil(float(size) / float(batch_size)))
        # return [(i * batch_size, min(size, (i + 1) * batch_size))
        #        for i in range(0, nb_batch)]
        return [(ndx, min(ndx+batch_size, size)) for ndx in range(
            0, size, batch_size)]
