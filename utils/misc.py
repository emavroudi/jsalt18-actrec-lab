from numpy import ones, ceil, array


def get_batches(iterable, n=1):
    """

    :param iterable: list
    :param n: desired number of batches
    :return: exactly n batches of almost equal length
    """
    l = len(iterable)
    q = l // n
    r = l % n

    indices = [q*i + min(i, r) for i in range(n+1)]
    print(indices)
    return [iterable[indices[i]:indices[i+1]] for i in range(n)]


def frame_labels_to_segments(labels, start_fr=0):
    """

    :param labels: a vector of integers (labels), one per frame
    :param start_fr: frame index of the first label, labels are assumed to
    be continuous
    :return: segments: a list of tuples (segment_st, segment_e)
             segment_labels: a list of labels, one per segment
    """

    segment_starts = []
    segment_ends = []
    segment_labels = []

    frame_cnt = start_fr
    for frame_label in labels:
        if (len(segment_labels) == 0) or (segment_labels[-1] != frame_label):
            if len(segment_starts) != 0:
                segment_ends.append(frame_cnt - 1)
            segment_starts.append(frame_cnt)
            segment_labels.append(frame_label)
        frame_cnt += 1

    segment_ends.append(frame_cnt - 1)
    segments = array(list(zip(segment_starts, segment_ends)))
    segment_labels = array(segment_labels)
    return segments, segment_labels


def segments_to_frame_labels(segments, segment_labels):
    """

    :param segments: a list of tuples (segment_st, segment_e)
    :param segment_labels: a list of labels, one per segment
    :return: a vector of integers (labels), one per frame
    """

    nb_frames = segments[-1][1] - segments[0][0] + 1
    labels = -ones(nb_frames).astype(int)
    frame_ind = []

    seg_cnt = 0
    fr_cnt = 0
    for seg in segments:
        seg_len = seg[1] - seg[0] + 1
        labels[fr_cnt: fr_cnt + seg_len] = segment_labels[seg_cnt]
        frame_ind += range(seg[0], seg[1]+1)
        seg_cnt += 1
        fr_cnt += seg_len

#    if any(label < 0 for label in labels):
#        raise ValueError("Labeled segments should be contiguous without gaps")
    if len(labels) != len(frame_ind):
        raise ValueError("Length of labels is not the same as length of frame"
                         "indices")

    return frame_ind, labels


def make_batches(size, batch_size):
    """
    Returns a list of batch indices (tuples of indices).
    :param size
    :param batch_size
    """
    nb_batch = int(ceil(float(size) / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, nb_batch)]


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def pad_sequences(sequences, dtype='float64', value=-1.0):
    """

    :param sequences: list with nb_samples elements, which are numpy arrays
                      whose first dimension in nb_timesteps
    :param dtype: type of resulting numpy array
    :param value: float, value to pad the sequences to the desired value.

    :return: numpy array with dimensions (nb_samples, max_timesteps, ?)
    """

    nb_samples = len(sequences)
    timesteps_per_sample = [s.shape[0] for s in sequences]

    max_timesteps = max(timesteps_per_sample)

    item_shape = list(sequences[0].shape)
    padded_array_shape = [nb_samples] + item_shape
    padded_array_shape[1] = max_timesteps
    res = (ones(tuple(padded_array_shape)) * value).astype(dtype)

    for idx, s in enumerate(sequences):
        res[idx, :timesteps_per_sample[idx]] = s

    return res
