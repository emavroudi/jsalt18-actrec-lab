from numpy import ones, max, zeros
from abc import ABCMeta, abstractmethod


class DataPreprocessor(metaclass=ABCMeta):
    """
    Abstract base class used to build new data preprocessors
    """
    def __init__(self, params):
        """
        :param params: a dictionary of parameters
        """
        self.params = params

    @abstractmethod
    def preprocess(self, features_lst, labels_lst):
        """
        Args:
        features_lst: a list with nb_samples elements. Each element is a
             (nb_timesteps, feat_dim) numpy array
        labels_lst: a list with nb_samples elements. Each element is a
             (nb_timesteps,) numpy array or list

        :return: feat, labels, sample_weight
        """
        raise NotImplementedError

    def set_max_nb_frames(self, max_nb_frames):
        self.params['max_nb_frames'] = max_nb_frames


def pad_sequences(sequences, max_len=None, dtype='float32', value=0.):
    """

    :param sequences: list of nb_samples elements. Each element is an array
                      of shape (timesteps, feat_dim)
    :param max_len: if not None, pad all sequences to length max_len
    :param dtype: resulting numpy array dtype
    :param value: padding value
    :return:
            padded_sequences:
                numpy array with shape (nb_samples, max_len, feat_dim)
            padding_mask:
                numpy binary array with shape (nb_samples, max_len, 1),
                having zeroes at padded indices and ones elsewhere
    """

    nb_samples = len(sequences)
    # Get feature dimension from the first sequence
    feat_dim = sequences[0].shape[1]
    timesteps_per_sample = [s.shape[0] for s in sequences]
    max_timesteps = max(timesteps_per_sample)

    if max_len is None:
        max_len = max_timesteps
    else:
        if max_len < max_timesteps:
            raise ValueError("Max_len: {} less than max_timesteps: {}".format(
                max_len, max_timesteps))

    res = (ones((nb_samples, max_len, feat_dim)) * value).astype(dtype)
    padding_mask = zeros((nb_samples, max_len)).astype(dtype)
    for idx, s in enumerate(sequences):
        res[idx, :timesteps_per_sample[idx], :] = s
        padding_mask[idx, :timesteps_per_sample[idx]] = 1

    padded_sequences = res

    return padded_sequences, padding_mask


def pad_sequences_batch(sequences_batch, max_len, dtype='float32', value=0.):
    """

    :param sequences_batch: list of nb_batches elements.
        Each element is an array of shape (batch_size, timesteps, feat_dim)
    :param max_len: if not None, pad all sequences to length max_len
    :param dtype: resulting numpy array dtype
    :param value: padding value
    :return:
            padded_sequences:
                numpy array with shape (nb_samples, max_len, feat_dim)
    """

    nb_batches = len(sequences_batch)
    # Get info from the first sequence batch
    tensor_ndims = sequences_batch[0].ndim
    nb_samples = sum([sequences_batch[i].shape[0] for i in range(nb_batches)])
    timesteps_per_batch = [s.shape[1] for s in sequences_batch]
    if tensor_ndims == 2:
        res = (ones((nb_samples, max_len)) * value).astype(dtype)
    elif tensor_ndims == 3:
        feat_dim = sequences_batch[0].shape[-1]
        res = (ones((nb_samples, max_len, feat_dim)) * value).astype(dtype)
    else:
        raise ValueError('Not supported tensor shape')

    cnt = 0
    for (batch_ind, seq_batch) in enumerate(sequences_batch):
        batch_size = sequences_batch[batch_ind].shape[0]
        for seq_ind in range(batch_size):
            s = sequences_batch[batch_ind][seq_ind]
            res[cnt, :timesteps_per_batch[batch_ind]] = s
            cnt += 1

    padded_sequences = res

    return padded_sequences
