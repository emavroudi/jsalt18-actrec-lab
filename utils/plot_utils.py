import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy import array, mean, unique, vstack
from os.path import join

mpl.rcParams.update({'font.size': 18})


def my_plot(vector, xlabel_str=None, ylabel_str=None, title_str=None,
            output_file=None):

    plt.plot(vector)
    if xlabel_str is not None:
        plt.xlabel(xlabel_str)
    if ylabel_str is not None:
        plt.ylabel(ylabel_str)
    if title_str is not None:
        plt.title(title_str)
    if output_file is not None:
        plt.savefig(output_file)


def imshow_(x, **kwargs):
    if x.ndim == 2:
        im = plt.imshow(x, interpolation="nearest", **kwargs)
    elif x.ndim == 1:
        im = plt.imshow(x[:,None].T, interpolation="nearest", **kwargs)
        plt.yticks([])
    plt.axis("tight")
    return im


def viz_sequence_predictions(nb_classes, split, y_pred, y_true, output_file):
    # # Output all truth/prediction pairs
    plt.figure(split, figsize=(20, 10))
    n_test = len(y_true)
    P_test_ = array(y_pred) / float(nb_classes - 1)
    y_test_ = array(y_true) / float(nb_classes - 1)
    values = []
    for i in range(len(y_true)):
        P_tmp = vstack([y_test_[i][:], P_test_[i][:]])
        plt.subplot(n_test, 1, i + 1)
        im = imshow_(P_tmp, vmin=0, vmax=1, cmap=plt.cm.jet)
        plt.xticks([])
        plt.yticks([])
        acc = mean(y_true[i] == y_pred[i]) * 100
        plt.ylabel("{:.01f}".format(acc))

        values.append(unique(P_tmp.ravel()))

        print("Visualized predictions")

    plt.savefig(output_file)
    plt.clf()


def plot_label_seq(label_seq, nb_classes, y_label=None, actions=None,
                   cmap='rainbow', output_file=None, title=None,
                   legend=None, figsize=None):

    if figsize is None:
        figsize = (20, 2)
    # Output all truth/prediction pairs
    actions_in_seq = unique(label_seq)
    fig = plt.figure(figsize=figsize)
    norm_label_seq = array(label_seq) / float(nb_classes-1)
    im = imshow_(norm_label_seq, vmin=0, vmax=1, cmap=plt.get_cmap(cmap))

    if y_label is not None:
        plt.ylabel("{}".format(y_label))

    if title is not None:
        plt.title(title)

    if legend is not None:
        values = unique(norm_label_seq.ravel())

        # get the colors of the values, according to the
        # colormap used by imshow
        colors = [im.cmap(im.norm(value)) for value in values]

        # create a patch (proxy artist) for every color
        if actions is None:
            patches = [
                mpatches.Patch(color=colors[i],
                               label="Action {}".format(values[i]))
                for i in range(len(values))]
        else:
            patches = [
                mpatches.Patch(color=colors[i],
                               label="{}".format(actions[actions_in_seq[i]]))
                for i in range(len(values))]

        # put those patched as legend-handles into the legend
        lgd = plt.legend(handles=patches, bbox_to_anchor=(1.2, 0.5),
                         loc='center right', borderaxespad=0.)

    if output_file is not None:
        if legend is not None:
            plt.savefig(output_file, dpi=300,
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')

    plt.clf()
    plt.close(fig)


def plot_optimization_log_frame(optimization_log, output_dir,
                                nb_epochs=None):

    # Plot frame loss
    output_file = join(output_dir, 'frame_loss.png')
    variables = ['train_frame_loss', 'val_frame_loss']
    linestyles = ['-', ':']
    colors = ['b', 'r']
    title = 'Frame loss'
    plot_lines(variables=variables,
               lines_dict=optimization_log, linestyles=linestyles,
               colors=colors, title=title,
               output_file=output_file, nb_epochs=nb_epochs)
    # Plot frame train loss
    output_file = join(output_dir, 'train_frame_loss.png')
    variables = ['train_frame_loss']
    linestyles = ['-']
    colors = ['b']
    title = 'Frame loss'
    plot_lines(variables=variables,
               lines_dict=optimization_log, linestyles=linestyles,
               colors=colors, title=title,
               output_file=output_file, nb_epochs=nb_epochs)
    # Plot frame validation loss
    output_file = join(output_dir, 'val_frame_loss.png')
    variables = ['val_frame_loss']
    linestyles = [':']
    colors = ['r']
    title = 'Frame loss'
    plot_lines(variables=variables,
               lines_dict=optimization_log, linestyles=linestyles,
               colors=colors, title=title,
               output_file=output_file, nb_epochs=nb_epochs)

    # Plot frame train metrics
    output_file = join(output_dir, 'train_frame_metric.png')
    variables = ['train_frame_metric']
    linestyles = ['-']
    colors = ['b']
    title = 'Frame metric'
    plot_lines(variables=variables,
               lines_dict=optimization_log, linestyles=linestyles,
               colors=colors, title=title,
               output_file=output_file, nb_epochs=nb_epochs)
    # Plot frame val metrics
    output_file = join(output_dir, 'val_frame_metric.png')
    variables = ['val_frame_metric']
    linestyles = [':']
    colors = ['r']
    title = 'Frame metric'
    plot_lines(variables=variables,
               lines_dict=optimization_log, linestyles=linestyles,
               colors=colors, title=title,
               output_file=output_file, nb_epochs=nb_epochs)


def plot_lines(variables, lines_dict, linestyles=None, colors=None,
               title=None, output_file=None, nb_epochs=None,
               xlabel=None):
    # Plot
    var_cnt = 0
    legends = []

    for variable in variables:
        x = lines_dict[variable]

        if nb_epochs is None:
            nb_epochs = len(x)
        else:
            nb_epochs = min(len(x), nb_epochs)

        if linestyles is None:
            linestyle = '-'
        else:
            linestyle = linestyles[var_cnt]

        if colors is None:
            color = 'b'
        else:
            color = colors[var_cnt]

        plt.plot(range(0, nb_epochs), x[:nb_epochs],
                 linestyle=linestyle, color=color)

        legends.append(variable)

        var_cnt += 1

    plt.title(title)
    if xlabel is None:
        xlabel = 'Epochs'

    plt.xlabel(xlabel)
    plt.legend(legends, loc='best')

    if output_file is not None:
        plt.savefig(output_file)

    # plt.show()
    plt.clf()
