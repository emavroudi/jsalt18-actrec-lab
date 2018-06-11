import numpy as np
from sklearn.metrics import accuracy_score
from itertools import groupby
from utils.misc import frame_labels_to_segments

import logging
logging.basicConfig(level=logging.DEBUG)


def per_frame_accuracy(y_true, y_pred, sample_weights=None,
                       average_across_seq=True):
    """
    :param y_true: numpy array of ground truth labels for a sequence with shape
                   (nb_timesteps, ) or numpy array of ground truth labels
                   for nb_samples sequences (padded) with shape:
                   (nb_samples, max_len)
    :param y_pred: numpy array of predicted labels for a sequence with shape
                   (nb_timesteps, ) or  numpy array of predicted labels
                   for nb_samples sequences (padded) with shape:
                   (nb_samples, max_len)
    :param sample_weights: numpy array with binary indicators of padded/
                           non-padded elements (either (nb_timesteps,) for
                           one sequence or (nb_sequences, max_len) for multiple
                           sequences
    :param average_across_seq: If True return average accuracy across all
                               sequences. If False return per frame accuracy
                               for each sequence.
    :return:
        acc: per frame accuracy
    """
    if type(y_true) == list:
        acc = np.mean([accuracy_score(y_true[i], y_pred[i])
                    for i in range(len(y_true))])

    elif y_true.ndim == 1:

        acc = accuracy_score(y_true, y_pred, sample_weight=sample_weights,
                             normalize=True)
    elif y_true.ndim == 2:
        nb_sequences = y_true.shape[0]
        per_seq_acc = []
        for seq in range(nb_sequences):
            per_seq_acc.append(accuracy_score(y_true[seq], y_pred[seq],
                sample_weight=sample_weights[seq], normalize=True))
        if average_across_seq:
            acc = np.mean(per_seq_acc)
        else:
            acc = np.array(per_seq_acc)
    else:
        raise ValueError("Metric inputs should be 1 or 2 dimensional")

    return acc


def per_frame_accuracy_one_hot(y_true, y_pred, sample_weights=None,
                               average_across_seq=True):
    """
    :param y_true: numpy array of ground truth labels for a sequence with shape
                   (nb_timesteps, nb_classes) or
                   (nb_batches, nb_timesteps, nb_classes)
    :param y_pred: numpy array of predicted labels for a sequence with shape
                   (nb_timesteps, nb_classes) or
                   (nb_batches, nb_timesteps, nb_classes)
    :param sample_weights: numpy array with binary indicators of padded/
                           non-padded elements (either (nb_timesteps,) for
                           one sequence or (nb_sequences, max_len) for multiple
                           sequences
    :param average_across_seq: If True return average accuracy across all
                               sequences. If False return per frame accuracy
                               for each sequence.
    :return:
        acc: per frame accuracy
    """

    if y_true.ndim == 2:
        if sample_weights is not None:
            y_true = y_true[np.nonzero(sample_weights), :]
            y_pred = y_pred[np.nonzero(sample_weights), :]
        acc = 1 - np.sum(np.abs(y_true - y_pred)) / y_true.size

    elif y_true.ndim == 3:
        nb_sequences = y_true.shape[0]
        per_seq_acc = []
        for seq in range(nb_sequences):

            if sample_weights is not None:
                y_true_ = y_true[seq][np.nonzero(sample_weights[seq])]
                y_pred_ = y_pred[seq][np.nonzero(sample_weights[seq])]
            else:
                y_true_ = np.array(y_true, copy=True)
                y_pred_ = np.array(y_pred, copy=True)

            acc = 1 - np.sum(np.abs(y_true_ - y_pred_)) / y_true_.size
            per_seq_acc.append(acc)
        if average_across_seq:
            acc = np.mean(per_seq_acc)
        else:
            acc = np.array(per_seq_acc)
    else:
        raise ValueError("Metric inputs should be 1 or 2 dimensional")

    return acc


# Lea Metrics
def edit_score(y_true, y_pred, sample_weights=None,
               average_across_seq=True, norm=True, ignore_bg=True):
    if type(y_true) == list:
        score = np.mean([_edit_score(y_pred[i], y_true[i], norm=norm,
                                  ignore_bg=ignore_bg)
                    for i in range(len(y_true))])
    elif y_true.ndim == 1:
        if sample_weights is not None:
            y_true = y_true[np.nonzero(sample_weights)]
            y_pred = y_pred[np.nonzero(sample_weights)]
        score = _edit_score(y_pred, y_true, norm=norm, ignore_bg=ignore_bg)
    elif y_true.ndim == 2:

        nb_sequences = y_true.shape[0]
        per_seq_score = []
        for seq in range(nb_sequences):
            y_true_ = y_true[seq][np.nonzero(sample_weights[seq])]
            y_pred_ = y_pred[seq][np.nonzero(sample_weights[seq])]
            score = _edit_score(y_pred_, y_true_, norm=norm,
                                ignore_bg=ignore_bg)
            per_seq_score.append(score)
        if average_across_seq:
            score = np.mean(per_seq_score)
        else:
            score = np.array(per_seq_score)
    else:
        raise ValueError("Metric inputs should be 1 or 2 dimensional")

    return score


def overlap_f1(y_true, y_pred, n_classes, sample_weights=None,
               average_across_seq=True, bg_class=0, overlap=.1):
    if type(y_true) == list:
        score = np.mean([_overlap_f1(y_pred[i], y_true[i], n_classes,
                                     bg_class=bg_class,overlap=overlap)
                    for i in range(len(y_true))])

    elif y_true.ndim == 1:
        if sample_weights is not None:
            y_true = y_true[np.nonzero(sample_weights)]
            y_pred = y_pred[np.nonzero(sample_weights)]
        score = _overlap_f1(y_pred, y_true, n_classes, bg_class=bg_class,
                            overlap=overlap)
    elif y_true.ndim == 2:

        nb_sequences = y_true.shape[0]
        per_seq_score = []
        for seq in range(nb_sequences):
            y_true_ = y_true[seq][np.nonzero(sample_weights[seq])]
            y_pred_ = y_pred[seq][np.nonzero(sample_weights[seq])]
            score = _overlap_f1(y_pred_, y_true_, n_classes, bg_class=bg_class,
                                overlap=overlap)
            per_seq_score.append(score)
        if average_across_seq:
            score = np.mean(per_seq_score)
        else:
            score = np.array(per_seq_score)
    else:
        raise ValueError("Metric inputs should be 1 or 2 dimensional")

    return score


def edit_score_seg(seg_y_true, seg_y_pred, seg_sample_weights,
                   seg_pred_sequence_lengths, norm=True,
                   average_across_seq=True):
    if seg_y_pred.ndim == 2:
        nb_sequences = seg_y_true.shape[0]
        per_seq_score = []
        for seq in range(nb_sequences):
            seg_y_true_ = seg_y_true[seq][np.nonzero(seg_sample_weights[seq])]
            seg_y_pred_ = seg_y_pred[seq][:seg_pred_sequence_lengths[seq]]
            score = _levenshtein(seg_y_pred_, seg_y_true_, norm=norm)
            per_seq_score.append(score)
        if average_across_seq:
            score = np.mean(per_seq_score)
        else:
            score = np.array(per_seq_score)
    else:
        raise ValueError("Metric inputs should be 2 dimensional (padded)")
    return score


def _edit_score(y_pred, y_true, norm=True, ignore_bg=True):
    """
    Computes edit distance between y_pred, y_true
    Args:
        y_pred: predicted sequence of labels (nb_timesteps,)
        y_true: ground truth sequence of labels (nb_timesteps,)
        norm: normalize score
        ignore_bg: ignore background class (Always assume that background
                   class has label 0)
    """

    # Get sequence of action labels by removing consecutive duplicates
    y_pred_seg_labels = np.array([k for k, g in groupby(y_pred)], dtype='int')
    y_true_seg_labels = np.array([k for k, g in groupby(y_true)], dtype='int')

    if ignore_bg:
            y_pred_seg_labels = y_pred_seg_labels[np.nonzero(y_pred_seg_labels)]
            y_true_seg_labels = y_true_seg_labels[np.nonzero(y_true_seg_labels)]

    return _levenshtein(y_pred_seg_labels, y_true_seg_labels, norm)


def _levenshtein(p, y, norm=False):
    """
    Levenshtein distance, implementation by Colin Lea
    https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/
    metrics.py
    :param p: predicted segment labels
    :param y: ground truth segment labels
    :param norm: If True. return normalized distance
    :return:
    """
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], np.float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def _overlap_f1(p, y, n_classes, bg_class=0, overlap=.1):
    """
    Overlap f1 distance, implementation by Colin Lea
    https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/
    metrics.py
    """

    # true_intervals = np.array(utils.segment_intervals(y))
    # true_labels = utils.segment_labels(y)
    # pred_intervals = np.array(utils.segment_intervals(p))
    # pred_labels = utils.segment_labels(p)
    true_intervals, true_labels = frame_labels_to_segments(y)
    pred_intervals, pred_labels = frame_labels_to_segments(p)
    # Remove background labels
    if bg_class is not None:
        true_intervals = true_intervals[true_labels != bg_class]
        true_labels = true_labels[true_labels != bg_class]
        pred_intervals = pred_intervals[pred_labels != bg_class]
        pred_labels = pred_labels[pred_labels != bg_class]

    n_true = true_labels.shape[0]
    n_pred = pred_labels.shape[0]

    # We keep track of the per-class TPs, and FPs.
    # In the end we just sum over them though.
    tp = np.zeros(n_classes, np.float)
    fp = np.zeros(n_classes, np.float)
    true_used = np.zeros(n_true, np.float)

    for j in range(n_pred):
        # Compute IoU against all others
        intersection = np.minimum(pred_intervals[j, 1],
                                  true_intervals[:, 1]) - np.maximum(
            pred_intervals[j, 0], true_intervals[:, 0])
        union = np.maximum(pred_intervals[j, 1],
                           true_intervals[:, 1]) - np.minimum(
            pred_intervals[j, 0], true_intervals[:, 0])
        IoU = (intersection / union) * (pred_labels[j] == true_labels)

        # Get the best scoring segment
        idx = IoU.argmax()

        # If the IoU is high enough and the true segment isn't already used
        # Then it is a true positive. Otherwise is it a false positive.
        if IoU[idx] >= overlap and not true_used[idx]:
            tp[pred_labels[j]] += 1
            true_used[idx] = 1
        else:
            fp[pred_labels[j]] += 1

    tp = tp.sum()
    fp = fp.sum()
    # False negatives are any unused true segment (i.e. "miss")
    fn = n_true - true_used.sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    # If the prec+recall=0, it is a NaN. Set these to 0.
    f1 = np.nan_to_num(f1)

    return f1 * 100


def evaluate_nods(val_frame_y_pred, val_frame_sample_weights, nb_classes,
                  val_frame_y_true_nods, downsampling_factor):

    nb_val_sequences = len(val_frame_y_true_nods)
    val_frame_y_pred_nods_lst = []

    for seq in range(nb_val_sequences):
        val_frame_y_pred_seq = val_frame_y_pred[seq][
            np.nonzero(val_frame_sample_weights[seq])]
        nb_frames_nods = len(val_frame_y_true_nods[seq])
        val_frame_y_pred_seq_nods = np.repeat(
            val_frame_y_pred_seq, downsampling_factor)[:nb_frames_nods]
        nb_frames_nods_pred = len(val_frame_y_pred_seq_nods)
        if nb_frames_nods != nb_frames_nods_pred:
            logging.warning(
                    "Mismatch between predicted and true sequence lengths "
                    "for seq %d: (%d, %d)",
                    seq, nb_frames_nods_pred, nb_frames_nods)
            val_frame_y_true_nods[seq] = val_frame_y_true_nods[
                    seq][:nb_frames_nods_pred]
        val_frame_y_pred_nods_lst.append(val_frame_y_pred_seq_nods)

    val_frame_accuracy = per_frame_accuracy(
        y_true=val_frame_y_true_nods, y_pred=val_frame_y_pred_nods_lst,
        sample_weights=val_frame_sample_weights,
        average_across_seq=True)
    val_frame_edit_score_no_bg = edit_score(
        val_frame_y_true_nods, val_frame_y_pred_nods_lst,
        val_frame_sample_weights, average_across_seq=True,
        norm=True, ignore_bg=True)
    val_frame_edit_score = edit_score(
        val_frame_y_true_nods, val_frame_y_pred_nods_lst,
        val_frame_sample_weights, average_across_seq=True,
        norm=True, ignore_bg=False)
    val_frame_overlap_f1 = overlap_f1(
        val_frame_y_true_nods, val_frame_y_pred_nods_lst, nb_classes,
        val_frame_sample_weights,
        average_across_seq=True, bg_class=0, overlap=.1)

    return val_frame_accuracy, val_frame_edit_score_no_bg, \
        val_frame_edit_score, val_frame_overlap_f1
