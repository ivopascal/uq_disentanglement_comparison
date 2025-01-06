import numpy as np
from matplotlib import pyplot as plt


from disentanglement.datatypes import UncertaintyResults


def plot_ale_epi_acc_on_axes(ax, results: UncertaintyResults, accuracy_y_ax_to_share=None, is_final_column=False,
                             std=None):
    if not results:
        return

    plot_line_and_confidence_interval(ax, results, std, 'epistemic_uncertainties', label="Epi")
    plot_line_and_confidence_interval(ax, results, std, 'aleatoric_uncertainties', label="Ale")

    accuracy_axes = ax.twinx()
    plot_line_and_confidence_interval(accuracy_axes, results, std, 'accuracies', label="Accuracy", color='green')

    if is_final_column:
        accuracy_axes.set_ylabel("Accuracy", color='green')
    else:
        plt.setp(accuracy_axes.get_yticklabels(), visible=False)

    if accuracy_y_ax_to_share:
        accuracy_axes.sharey(accuracy_y_ax_to_share)

    return accuracy_axes


def plot_line_and_confidence_interval(ax, results, std, get_string, label, color=None):
    ax.plot(results.changed_parameter_values, results.__getattribute__(get_string), label=label, color=color)

    if std:
        plot_confidence_intervals(ax, results, std, get_string, label=f"_{label}", color=color)


def plot_confidence_intervals(ax, results, std, get_string, label, color=None):
    uncertainties_mean = np.array(results.__getattribute__(get_string))
    uncertainties_std = np.array(std.__getattribute__(get_string))
    ax.fill_between(results.changed_parameter_values,
                    uncertainties_mean - 1.96 * uncertainties_std,
                    uncertainties_mean + 1.96 * uncertainties_std,
                    alpha=0.3, label=label, color=color)


