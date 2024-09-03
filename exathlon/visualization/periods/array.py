"""Period ndarray visualization module.

Gathers functions for visualizing periods represented as ndarrays.
"""
import os
import re
import argparse
import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

from utils.spark.period_info import get_trace_type_title, get_period_title_from_info
from data.helpers import get_concatenated
from detection.metrics.helpers import extract_multiclass_ranges_ids
from detection.threshold_selectors import IQRSelector
from visualization.reporting import plot_boxplots


def period_wise_figure(
    plot_func,
    periods,
    periods_labels=None,
    periods_info=None,
    fig_title=None,
    full_output_path=None,
    **func_args,
):
    """Plots period-wise items within the same figure using the provided plotting function for each axis.

    Args:
        plot_func (func): plotting function to call for each period in the figure.
        periods (ndarray): period-wise arrays.
        periods_labels (ndarray|None): optional periods labels.
        periods_info (ndarray|None): optional periods information.
        fig_title (str|None): optional figure title.
        full_output_path (str|None): optional output path to save the figure to (including file name and extension).
        **func_args: optional keyword arguments to pass to the plotting function.
    """
    # create and setup new figure
    n_periods = len(periods)
    fontsizes = {"title": 25}
    fig, axs = plt.subplots(n_periods, 1, sharex="none")
    fig.set_size_inches(20, 5 * n_periods)
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=fontsizes["title"], fontweight="bold")

    # define axes, period labels and period titles to loop through
    if n_periods == 1:
        axs = [axs]
    looped = dict()
    for k, items in zip(["labels", "title"], [periods_labels, periods_info]):
        if items is None:
            looped[k] = np.repeat(None, n_periods)
        elif k == "labels":
            looped[k] = items
        else:
            looped[k] = [get_period_title_from_info(info) for info in items]

    # call the plotting function for each period separately
    for i, ax in enumerate(axs):
        plot_func(
            periods[i],
            period_labels=looped["labels"][i],
            period_title=looped["title"][i],
            ax=ax,
            **func_args,
        )

    # save the figure as an image if an output path was provided
    if full_output_path is not None:
        print(
            f"saving period-wise figure to {full_output_path}...", end=" ", flush=True
        )
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        fig.savefig(full_output_path)
        plt.close()
        print("done.")


def plot_score_boxes(
    periods_scores,
    periods_labels=None,
    periods_info=None,
    group_by="period",
    fig_title=None,
    full_output_path=None,
    return_medians=False,
):
    """Plots a set of outlier score boxplots for the provided `periods_scores`.

    Each group of boxplots will be plotted in a separate axis. Groups can be periods or, for spark data,
    applications or input rates (as extracted from `periods_info`).

    If `return_medians` is True, the median values for each boxplot will be returned as a pd.DataFrame,
    with boxplot groups as indices and boxplot names as columns (empty cells being NaN).

    Args:
        periods_scores (ndarray): periods scores of shape `(n_periods, period_length)`.
            Where `period_length` depends on the period.
        periods_labels (ndarray|None): optional multiclass periods labels of the same shape as `periods_scores`.
        periods_info (list|None): optional periods information.
        group_by (str): boxplots grouping criterion (either "period", "app" or "input_rate", the latter two
            only allowed for spark data).
        fig_title (str|None): optional figure title.
        full_output_path (str|None): optional output path to save the figure to (including file name and extension).
        return_medians (bool): whether to return the median values for the boxplots.

    Returns:
        pd.DataFrame|None: the boxplot medians DataFrame if `return_medians` is True, nothing otherwise.
    """
    # check the provided boxplot grouping
    allowed_groupings = ["period"]
    if USED_DATA == "spark":
        allowed_groupings += ["app", "input_rate"]
    assert (
        group_by in allowed_groupings
    ), f"the provided boxplot grouping must be in {allowed_groupings}"
    a_t = "periods information must be provided when grouping boxplots by application or input rate"
    assert not (periods_info is None and group_by in ["app", "input_rate"]), a_t

    # dictionary with as keys the axis names and values the DataFrames used to derive boxplots
    dfs_dict = dict()
    # period labels and group titles
    n_periods, looped = periods_scores.shape[0], dict()
    for k, items in zip(["labels", "title"], [periods_labels, periods_info]):
        if items is None:
            looped[k] = np.repeat(None, n_periods)
        elif k == "labels":
            # labels or group title set to the periods information
            looped[k] = items
        else:
            # group title extracted from the periods information
            if group_by == "period":
                get_title = get_period_title_from_info
            else:
                if group_by == "app":
                    info_idx, formatting_f = 0, lambda s: f"Application {s}"
                else:
                    info_idx, formatting_f = 2, lambda s: f"{int(s):,} records/sec"
                get_title = lambda info: formatting_f(info[0].split("_")[info_idx])
            looped[k] = [get_title(info) for info in items]

    for period_scores, period_labels, group_title in zip(
        periods_scores, looped["labels"], looped["title"]
    ):
        if group_title not in dfs_dict:
            dfs_dict[group_title] = dict()
        if period_labels is not None:
            # boxplot of outlier scores assigned to normal data
            if "NORMAL" not in dfs_dict[group_title]:
                dfs_dict[group_title]["NORMAL"] = []
            dfs_dict[group_title]["NORMAL"] = get_concatenated(
                dfs_dict[group_title]["NORMAL"], period_scores[period_labels == 0]
            )
            ranges_ids_dict = extract_multiclass_ranges_ids(period_labels)
            if group_by == "period":
                # boxplot of outlier scores assigned to each anomalous range (sorted by beginning index)
                flat_ranges = [
                    [beg, end, type_]
                    for type_, ranges in ranges_ids_dict.items()
                    for beg, end in ranges
                ]
                for range_idx, (beg, end, type_) in enumerate(
                    sorted(flat_ranges, key=lambda r: r[0])
                ):
                    dfs_dict[group_title][f"T{type_}#{range_idx + 1}"] = period_scores[
                        beg:end
                    ]
            else:
                # boxplot of outlier scores assigned to ranges of each anomaly type
                for type_ in ranges_ids_dict:
                    if f"T{type_}" not in dfs_dict[group_title]:
                        dfs_dict[group_title][f"T{type_}"] = []
                    dfs_dict[group_title][f"T{type_}"] = get_concatenated(
                        dfs_dict[group_title][f"T{type_}"],
                        period_scores[period_labels == type_],
                    )
        else:
            # boxplot of outlier scores assigned to the periods of each group
            if "SCORES" not in dfs_dict[group_title]:
                dfs_dict[group_title]["SCORES"] = []
            dfs_dict[group_title]["SCORES"] = get_concatenated(
                dfs_dict[group_title]["SCORES"], period_scores
            )

    # plot boxplots corresponding to the groupings
    dfs_dict = {
        ax_title: pd.DataFrame.from_dict(d, orient="index").transpose()
        for ax_title, d in dfs_dict.items()
    }
    plot_boxplots(dfs_dict, fig_title=fig_title, full_output_path=full_output_path)

    # return median values for the boxplots if specified
    if return_medians:
        for k in dfs_dict:
            # remove number of considered points from the columns so they can be shared
            dfs_dict[k].columns = [c[: c.index(" (")] for c in dfs_dict[k].columns]
            if group_by == "period":
                # remove type information from the anomalous ranges so as to better share columns
                dfs_dict[k].columns = [
                    re.sub("T\d{1}", "INSTANCE", c) for c in dfs_dict[k].columns
                ]
        return pd.DataFrame.from_dict(
            {k: dfs_dict[k].median() for k in dfs_dict}, orient="index"
        )


def plot_period_scores(
    period_scores,
    period_labels=None,
    period_title=None,
    threshold=None,
    threshold_label=None,
    *,
    ax=None,
    color="darkgreen",
):
    """Performs a time plot of `period_scores`, highlighting anomalous ranges if specified and any.

    Args:
        period_scores (ndarray): 1d-array of outlier scores to plot.
        period_labels (ndarray|None): optional multiclass labels for the period's records.
        period_title (list|None): optional title to use for the period.
        threshold (float|None): optional outlier score threshold to highlight in the figure.
        threshold_label (str|None): label to use for the threshold (defaulting to "TS=`rounded_value`").
        ax (AxesSubplot|None): optional plt.axis to plot the scores on if not in a standalone figure.
        color (str): color to use for the curve of outlier scores.

    Returns:
        AxesSubplot: the axis the period was plotted on, to enable further usage.
    """
    # setup font sizes, title and highlighted outlier scores
    fontsizes = {"title": 25, "axes": 25, "legend": 25, "ticks": 22}
    title = period_title if period_title is not None else "Outlier Scores"
    highlighted_scores, highlighted_score_colors = [], []

    # create standalone figure if needed and setup axis
    if ax is None:
        plt.figure(figsize=(20, 5))
        ax = plt.axes()
    ax.set_title(title, fontsize=fontsizes["title"], y=1.07)
    ax.set_xlabel("Time Index", fontsize=fontsizes["axes"])
    ax.set_ylabel("Outlier Score", fontsize=fontsizes["axes"])
    ax.tick_params(axis="both", which="major", labelsize=fontsizes["ticks"])
    ax.tick_params(axis="both", which="minor", labelsize=fontsizes["ticks"])

    # plot the period outlier scores
    ax.plot(period_scores, color=color)

    # highlight the anomalous ranges of the period if any
    rect_patches = []
    if period_labels is not None:
        rect_patches = plot_period_anomalies(ax, period_labels)
        # highlight the average outlier score of the normal and anomalous records of the period
        try:
            highlighted_scores += [
                np.mean(period_scores[period_labels == 0]),
                np.mean(period_scores[period_labels > 0]),
            ]
            highlighted_score_colors += ["green", "red"]
        # typically in case there are no anomalous records in the considered periods
        except ZeroDivisionError:
            pass
    else:
        # highlight the average outlier score of the period's records
        highlighted_scores.append(np.mean(period_scores))
        highlighted_score_colors.append(color)

    # highlight the specific outlier scores in the y-axis using dedicated colors
    for score, score_color in zip(highlighted_scores, highlighted_score_colors):
        # decrease alpha of the closest y tick
        set_closest_tick_alpha(ax, "y", score, alpha=0.2)
        ax.set_yticks(np.append(ax.get_yticks(), score))
        ax.get_yticklabels()[-1].set_color(score_color)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # highlight the outlier score threshold with a red dashed line if provided
    if threshold is not None:
        ax.axhline(threshold, color="red", lw=2, linestyle="--")
        if threshold_label is None:
            threshold_label = f"TS={threshold:.3f}"
        # place threshold label on the right
        right_ax = ax.secondary_yaxis("right")
        right_ax.set_yticks([threshold])
        right_ax.set_yticklabels([threshold_label])
        right_ax.get_yticklabels()[-1].set_color("red")
        right_ax.tick_params(axis="y", which="major", labelsize=fontsizes["ticks"])

    # remove duplicate labels for the legend
    handles, labels = ax.get_legend_handles_labels()
    label_dict = dict(zip(labels, handles))
    if len(label_dict) > 0:
        ax.legend(
            handles=rect_patches,
            labels=label_dict.keys(),
            loc="upper left",
            prop={"size": fontsizes["legend"]},
        )
    ax.grid()
    return ax


def plot_period_anomalies(ax, period_labels):
    """Plots any anomalous range(s) within a period on `ax` based on `periods_labels`.

    Args:
        ax (AxesSubplot): plt.axis on which to plot the anomalous ranges.
        period_labels (ndarray): 1d-array of multiclass labels for the period.

    Returns:
        list: colored rectangle patches to show before the legend labels to better describe ranges.
    """
    # extract contiguous ranges from the record-wise labels
    ranges_ids_dict = extract_multiclass_ranges_ids(period_labels)
    colors_dict, rect_dict, labels_dict = dict(), dict(), dict()
    for k in ranges_ids_dict:
        colors_dict[k] = "r"
        rect_dict[k] = mpatches.Patch(
            facecolor=(1, 0, 0, 0.05), edgecolor="r", linewidth=1
        )
        labels_dict[k] = "Real Anomaly"
    if USED_DATA == "spark":
        # show more details for the last three anomaly types
        anomaly_types, anomaly_colors = (
            DATA_CONFIG["anomaly_types"],
            VISUALIZATION_CONFIG["anomaly_colors"],
        )
        n_anomaly_types = len(anomaly_types)
        for i in range(3):
            type_idx = n_anomaly_types - 3 + i
            anomaly_type_str, k = anomaly_types[type_idx], type_idx + 1
            colors_dict[k] = anomaly_colors[anomaly_type_str]
            # colors are assumed to be (r, g, b) tuples
            rect_dict[k] = mpatches.Patch(
                facecolor=(*colors_dict[k], 0.05), edgecolor=colors_dict[k], linewidth=1
            )
            labels_dict[k] = get_trace_type_title(anomaly_type_str)

    # plot anomalous ranges by type
    for type_label in ranges_ids_dict:
        anomalous_ranges = ranges_ids_dict[type_label]
        for range_ in anomalous_ranges:
            # end of the range is exclusive
            beg, end = (range_[0], range_[1] - 1)
            ax.axvspan(beg, end, color=colors_dict[type_label], alpha=0.05)
            for range_idx in [beg, end]:
                ax.axvline(
                    range_idx,
                    label=labels_dict[type_label],
                    color=colors_dict[type_label],
                )

    # return rectangle patches list so they can be used in a legend
    return list(rect_dict.values())


def set_closest_tick_alpha(ax, axis, value, alpha=0.2):
    """Sets to `alpha` the opacity of the tick on `axis` that is closest to `value`.

    Args:
        ax (AxesSubplot): plt.axis containing the ticks of interest.
        axis (str): relevant axis of `ax` (either "x" or "y").
        value (float): value whose closest tick to modify the opacity.
        alpha (float): new opacity value to set for the closest tick (between 0 and 1).
    """
    assert axis in ["x", "y"], 'axis must be either "x" or "y"'
    if axis == "x":
        ticks, tick_values = ax.xaxis.get_major_ticks(), ax.get_xticks()
    else:
        ticks, tick_values = ax.yaxis.get_major_ticks(), ax.get_yticks()
    tick_dists = np.abs(value - tick_values)
    ticks[tick_dists.argmin()].label1.set_alpha(alpha)


def plot_scores_ridge(
    group_scores,
    group_labels,
    group_titles=None,
    restricted_types=None,
    threshold=None,
    threshold_label=None,
    fig_title=None,
    full_output_path=None,
):
    """Performs a ridge plot of outlier scores with one row per group and one KDE plot per record type.

    Args:
        group_scores (ndarray): outlier scores for each group, of shape `(n_groups, group_length)`.
            Where `group_length` depends on the group.
        group_labels (ndarray): either binary or multiclass labels for each outlier score, with the
            shape as `group_scores`.
        group_titles (list|None): optional titles to use as x labels for each group row
            (defaulting to "Group #`group_idx+1`").
        restricted_types (list|None): optional restriction on the record types to plot (either
            "normal" or in `anomaly_types`).
        threshold (float|None): optional (shared) outlier score threshold to highlight in the figure.
        threshold_label (str|None): label to use for the threshold (defaulting to "TS=`rounded_value`").
        fig_title (str|None): optional figure title.
        full_output_path (str|None): optional output path to save the figure to (including file name and extension).
    """
    fontsizes_dict, n_groups = {
        "title": 22,
        "axes": 17,
        "legend": 15,
        "ticks": 17,
    }, len(group_scores)
    fig = plt.figure(figsize=(19, 2 * n_groups))
    gs = GridSpec(n_groups, 1)

    # check optional type restrictions and derive restricted labels if relevant
    if restricted_types is not None:
        a_t = 'restricted types have to be either "normal" or in `anomaly_types`'
        assert (
            len(set(restricted_types) - set(["normal"] + DATA_CONFIG["anomaly_types"]))
            == 0
        ), a_t
        restricted_ano_types, restricted_labels = [
            t for t in restricted_types if t != "normal"
        ], []
        if "normal" in restricted_types:
            restricted_labels.append(0)
        restricted_labels += [
            DATA_CONFIG["anomaly_types"].index(t) + 1 for t in restricted_ano_types
        ]

    # set default group titles if they were not provided
    if group_titles is None:
        group_titles = [f"Group #{i+1}" for i in range(n_groups)]

    # loop through each group
    axs, legend_handles_dict, legend_labels_dict, min_x = [], dict(), dict(), np.inf
    for i, (title, scores, labels) in enumerate(
        zip(group_titles, group_scores, group_labels)
    ):
        axs.append(fig.add_subplot(gs[i, 0], sharex=(None if i == 0 else axs[-1])))
        # only consider scores of the restricted types if relevant
        if restricted_types is not None:
            type_mask = functools.reduce(
                np.logical_or, (labels == i for i in restricted_labels)
            )
            scores, labels = scores[type_mask], labels[type_mask]
        # for spark data, remove scores of type "unknown" if any
        if USED_DATA == "spark":
            type_mask = labels != DATA_CONFIG["anomaly_types"].index("unknown") + 1
            scores, labels = scores[type_mask], labels[type_mask]
        # perform a KDE plot for each outlier score type
        df = pd.DataFrame({"score": scores, "label": labels})
        for label in df.label.unique():
            c = VISUALIZATION_CONFIG["label_colors"][label]
            axs[-1] = df[df.label == label].score.plot.kde(ax=axs[-1], color=c, lw=1)
            # get x and y coordinates of the last plotted line to shade its inner area and update `min_x`
            x, y = np.split(axs[-1].lines[-1].get_path().vertices, 2, 1)
            axs[-1].fill_between(x[:, 0], y[:, 0], color=c, alpha=0.2)
            current_min_x = np.min(x)
            if current_min_x < min_x:
                min_x = current_min_x
            # set handle and label to show in the legend for the outlier score type
            legend_handles_dict[label] = mpatches.Patch(
                facecolor=(*c, 0.2), edgecolor=c, linewidth=1
            )
            legend_labels_dict[label] = VISUALIZATION_CONFIG["label_legends"][label]

        # update the group's axis to make it part of a ridge plot
        axs[-1].grid(False)
        # make background transparent
        axs[-1].patch.set_alpha(0)
        # remove ticks and labels of the y axis
        axs[-1].set_yticklabels([])
        axs[-1].set_ylabel("")
        axs[-1].yaxis.set_ticks_position("none")
        # remove axis borders, only setting an x label for the last row
        if i == n_groups - 1:
            axs[-1].set_xlabel("Outlier Score", fontsize=fontsizes_dict["axes"])
            axs[-1].tick_params(
                axis="x", which="major", labelsize=fontsizes_dict["ticks"]
            )
        for s in ["top", "right", "left", "bottom"]:
            axs[-1].spines[s].set_visible(False)
        # display group title on the left of the x axis
        axs[-1].text(
            min_x, 0, title, fontsize=fontsizes_dict["axes"], ha="right", va="center"
        )

    # highlight the outlier score threshold with a red dashed line across axes if provided
    if threshold is not None:
        # connection patch spanning all the rows
        axs[-1].add_artist(
            mpatches.ConnectionPatch(
                xyA=(threshold, 0),
                xyB=(threshold, 0),
                coordsA="data",
                coordsB="data",
                axesA=axs[-1],
                axesB=axs[0],
                color="red",
                lw=2,
                linestyle="--",
            )
        )
        if threshold_label is None:
            threshold_label = f"TS={threshold:.1f}"
        axs[-1].text(
            threshold,
            -0.03,
            threshold_label,
            color="red",
            transform=axs[-1].get_xaxis_transform(),
            ha="center",
            va="top",
            size=fontsizes_dict["axes"],
        )
        # decrease alpha of the x axis tick closest to the threshold
        set_closest_tick_alpha(axs[-1], "x", threshold, alpha=0.2)

    # plot legend on the first axis, sorting by integer labels and removing duplicates
    sorted_label_keys = sorted(legend_labels_dict.keys())
    axs[0].legend(
        handles=[legend_handles_dict[k] for k in sorted_label_keys],
        labels=[legend_labels_dict[k] for k in sorted_label_keys],
        prop={"size": fontsizes_dict["legend"]},
    )
    # pack axes together to constitute the ridge plot
    gs.update(hspace=-0.5)

    # set figure title in the same absolute position no matter the figure size
    if fig_title is None:
        fig_title = "Distributions of Outlier Scores"
    transform = mtransforms.blended_transform_factory(
        axs[0].transData, axs[0].transAxes
    )
    axs[0].text(
        0.0,
        1.0,
        fig_title,
        fontsize=fontsizes_dict["title"],
        ha="center",
        va="center",
        fontweight="bold",
        transform=transform,
    )

    # save the figure as an image if an output path was provided
    if full_output_path is not None:
        print(
            f"saving ridge plot of outlier scores to {full_output_path}...",
            end=" ",
            flush=True,
        )
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        fig.savefig(full_output_path, bbox_inches="tight")
        plt.close()
        print("done.")


def plot_scores_histograms(
    periods_scores,
    periods_labels,
    restricted_types=None,
    type_thresholds=None,
    type_threshold_labels=None,
    hist_type="equidepth",
    ax=None,
    title=None,
    full_output_path=None,
):
    """Plots a flattened histogram of outlier scores for each record type in `period_scores`.

    Args:
        periods_scores (ndarray): periods scores of shape `(n_periods, period_length)`.
            Where `period_length` depends on the period.
        periods_labels (ndarray): either binary or multiclass periods labels.
            With the same shape as `periods_scores`.
        restricted_types (list|None): optional restriction on the record types to plot (either
            "normal" or in `anomaly_types`).
        type_thresholds (dict|float|None): optional outlier score threshold(s) to highlight in the figure
            (either as a dict where keys are anomaly type indices or as a single value).
        type_threshold_labels (dict|str|None): label(s) to use for the threshold(s), in the same format
            as `type_thresholds`. For a single value, defaults to "TS=`rounded_value`".
        hist_type (str): type of histograms to plot (either "capped" or "equidepth").
        ax (AxesSubplot|None): optional plt.axis to plot the distributions on if not in a standalone figure
            (only allowed if the scores are not capped).
        title (str|None): optional figure title if `ax` is None, else optional title for `ax`.
        full_output_path (str|None): optional output path to save the figure to (including file name and extension).
    """
    # get anomaly types and colors
    anomaly_types, anomaly_colors = (
        DATA_CONFIG["anomaly_types"],
        VISUALIZATION_CONFIG["anomaly_colors"],
    )
    # check optional type restrictions
    if restricted_types is not None:
        a_t = 'restricted types have to be either "normal" or in `anomaly_types`'
        assert len(set(restricted_types) - set(["normal"] + anomaly_types)) == 0, a_t

    # check histogram type
    assert hist_type in [
        "capped",
        "equidepth",
    ], 'histogram type must be either "capped" or "equidepth"'

    # label classes list, in which the index corresponds to the integer label
    label_class_names = ["normal"]
    # label texts dictionary, mapping a label class to its displayed text
    label_texts_dict = {"normal": "Normal"}
    label_class_names += anomaly_types
    label_texts_dict.update({a_t: f"T{i+1}" for i, a_t in enumerate(anomaly_types)})
    a_t = "a color must be provided for every type in `anomaly_types`"
    assert len(set(anomaly_types) - set(anomaly_colors.keys())) == 0, a_t
    colors = dict(
        {label_class_names[0]: VISUALIZATION_CONFIG["normal_color"]}, **anomaly_colors
    )
    alphas = dict({label_class_names[0]: 0.6}, **{k: 0.5 for k in anomaly_colors})

    # histogram assignments
    flattened_scores = np.concatenate(periods_scores)
    flattened_labels = np.concatenate(periods_labels)
    # for spark data, remove scores of records of type "unknown" if any
    if USED_DATA == "spark":
        type_mask = flattened_labels != anomaly_types.index("unknown") + 1
        flattened_scores, flattened_labels = (
            flattened_scores[type_mask],
            flattened_labels[type_mask],
        )
    # integer label classes
    int_label_classes = np.unique(flattened_labels)

    # any values exceeding `thresholding_factor` * IQR are grouped in the last bin
    thresholding_args = argparse.Namespace(
        **{"thresholding_factor": 3, "n_iterations": 1, "removal_factor": 1}
    )
    selector = IQRSelector(thresholding_args, "")
    selector.select_threshold(flattened_scores)
    flattened_scores[flattened_scores >= selector.threshold] = selector.threshold
    capped = (
        hist_type == "capped"
        and len(flattened_scores[flattened_scores >= selector.threshold]) != 0
    )
    assert not (
        capped and ax is not None
    ), "cannot plot distributions on `ax` when capping outlier scores"
    # put the "normal" class last if it is there so that it is shown above
    if 0 in int_label_classes:
        int_label_classes = [cl for cl in int_label_classes if cl != 0]
        int_label_classes.insert(len(int_label_classes), 0)
    scores_dict = dict()
    for int_class in int_label_classes:
        # only consider scores that are of the restricted types if relevant
        if restricted_types is None or label_class_names[int_class] in restricted_types:
            scores_dict[label_class_names[int_class]] = flattened_scores[
                flattened_labels == int_class
            ]

    # setup figure and plot outlier scores histograms
    fontsizes_dict = {"title": 22, "axes": 17, "legend": 17, "ticks": 17}
    n_bins = 30
    if capped:
        # plot the histograms on a wider left axis
        subplots_args = {
            "nrows": 1,
            "ncols": 2,
            "gridspec_kw": {"width_ratios": [3, 1]},
        }
        bins = np.linspace(
            np.min(flattened_scores), np.max(flattened_scores), n_bins + 1
        )
    else:
        subplots_args = dict()
        bins = n_bins
    fig, axs = (
        plt.subplots(figsize=(15, 5), **subplots_args) if ax is None else (None, ax)
    )
    hist_ax = axs[0] if capped else axs
    hist_labels = []
    for k, scores in scores_dict.items():
        hist_scores = scores[scores < selector.threshold] if capped else scores
        hist_labels.append(label_texts_dict[k])
        # set equidepth bins depending on the scores to plot if relevant (cast as scores are of dtype object)
        hist_bins = (
            get_equidepth_edges(scores.astype(np.float32), bins)
            if hist_type == "equidepth"
            else bins
        )
        hist_ax.hist(
            hist_scores,
            bins=hist_bins,
            label=hist_labels[-1],
            color=colors[k],
            density=True,
            alpha=alphas[k],
            edgecolor="black",
            linewidth=1.2,
        )
    # highlight the outlier score threshold(s) if provided
    if type_thresholds is not None:
        a_t = "`type_thresholds` and `type_threshold_labels` should have the same data type"
        # either a single anomaly type or explicitly passed a single threshold
        if len(anomaly_types) == 1 or not isinstance(type_thresholds, dict):
            assert not isinstance(type_threshold_labels, dict), a_t
            title = highlight_threshold(
                hist_ax,
                selector.threshold,
                type_thresholds,
                type_threshold_labels,
                "r",
                fontsizes_dict["legend"],
                title,
            )
        else:
            assert isinstance(type_threshold_labels, dict), a_t
            for label_key, threshold in type_thresholds.items():
                a_t = "each type-wise threshold should be assigned an explicit label"
                assert (
                    label_key in type_threshold_labels
                    and type_threshold_labels[label_key] is not None
                ), a_t
                title = highlight_threshold(
                    hist_ax,
                    selector.threshold,
                    threshold,
                    type_threshold_labels[label_key],
                    colors[label_class_names[label_key]],
                    fontsizes_dict["legend"],
                    title,
                )
    # set figure or axis title
    if ax is None:
        fig.suptitle(title, size=fontsizes_dict["title"], y=0.96)
    else:
        ax.set_title(title, size=fontsizes_dict["title"])
    plt.grid()
    # re-order the legend labels if relevant
    handles, legend_labels = hist_ax.get_legend_handles_labels()
    # all possible anomaly labels might not be represented in the scores
    used_label_texts = {k: v for k, v in label_texts_dict.items() if v in legend_labels}
    for la, new_idx in zip(used_label_texts.values(), range(len(hist_labels))):
        current_idx = legend_labels.index(la)
        legend_labels[current_idx] = legend_labels[new_idx]
        handle_to_move = handles[current_idx]
        handles[current_idx] = handles[new_idx]
        legend_labels[new_idx] = la
        handles[new_idx] = handle_to_move
    hist_ax.legend(
        loc="best",
        prop={"size": fontsizes_dict["legend"]},
        labels=legend_labels,
        handles=handles,
    )
    hist_ax.set_xlabel("Outlier Score", fontsize=fontsizes_dict["axes"])
    hist_ax.set_ylabel("Density", fontsize=fontsizes_dict["axes"])
    hist_ax.tick_params(axis="both", which="major", labelsize=fontsizes_dict["ticks"])
    hist_ax.tick_params(axis="both", which="minor", labelsize=fontsizes_dict["ticks"])
    hist_ax.grid(True)

    # add capped scores bar chart to a narrower right axis if relevant
    if capped:
        bar_ax, bar_labels = axs[1], []
        # put outlier scores of normal records back as the first key
        reordered_scores = dict(
            {label_class_names[0]: scores_dict[label_class_names[0]]},
            **{k: v for k, v in scores_dict.items() if k != label_class_names[0]},
        )
        i = 0
        for k, scores in reordered_scores.items():
            # only consider the first letter of the "normal" label due to space constraints
            bar_labels.append(
                label_texts_dict[k]
                if k != label_class_names[0]
                else label_texts_dict[k][0]
            )
            capped_scores = scores[scores == selector.threshold]
            bar_ax.bar(
                [i],
                [len(capped_scores) / len(scores)],
                color=colors[k],
                alpha=alphas[k],
                edgecolor="black",
                linewidth=1.2,
            )
            i += 1
        plt.xticks(
            range(len(reordered_scores)),
            bar_labels,
            fontweight="light",
            fontsize="x-large",
        )
        bar_ax.yaxis.set_label_position("right")
        bar_ax.yaxis.tick_right()
        bar_ax.tick_params(
            axis="both", which="major", labelsize=fontsizes_dict["ticks"]
        )
        bar_ax.tick_params(
            axis="both", which="minor", labelsize=fontsizes_dict["ticks"]
        )
        bar_ax.grid(True)

    # save the figure as an image if is standalone and an output path was provided
    if not any([v is None for v in [ax, full_output_path]]):
        print(
            f"saving scores distributions figure to {full_output_path}...",
            end=" ",
            flush=True,
        )
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        fig.savefig(full_output_path)
        plt.close()
        print("done.")


def get_equidepth_edges(array, n_bins):
    """Returns bin edges for an `n_bins` equidepth histogram of `array`'s values."""
    array_length = len(array)
    return np.interp(
        np.linspace(0, array_length, n_bins + 1),
        np.arange(array_length),
        np.sort(array),
    )


def highlight_threshold(
    ax,
    capping_threshold,
    threshold,
    threshold_label=None,
    threshold_color=None,
    label_fontsize=None,
    fig_title=None,
):
    """Highlights `threshold`, either on `ax` or through `threshold_label` in `fig_title`.

    Args:
        ax (AxesSubplot): plt.axis to highlight `threshold` on if below `capping_threshold`.
        capping_threshold: value above which `threshold` should not be highlighted on `ax`, but
            appear in `fig_title` instead through its label.
        threshold (float): threshold value to highlight in either `ax` or `fig_title`.
        threshold_label (str|None): label to use for the threshold (defaults to "TS={rounded_value}").
        threshold_color (str|None): color to use if `threshold` is highlighted on `ax` (defaults to "r").
        label_fontsize (int|None): label fontsize to use if `threshold` is highlighted on `ax` (defaults to 15).
        fig_title (str|None): figure title to extend with `threshold_label` if `threshold` is
            below `capping_threshold`.

    Returns:
        str: the provided figure title, possibly extended with `threshold_label` if `threshold` was
            below `capping_threshold`.
    """
    if threshold_label is None:
        threshold_label, label_fontsize = f"TS={threshold:.2f}", 15
    if threshold_color is None:
        threshold_color = "r"
    if threshold <= capping_threshold:
        threshold_pos = (threshold, 0.60 * ax.get_ylim()[1])
        threshold_text_pos = (1.05 * threshold_pos[0], 1.05 * threshold_pos[1])
        ax.axvline(threshold, color=threshold_color, lw=3, linestyle="--")
        ax.annotate(
            threshold_label,
            threshold_pos,
            color=threshold_color,
            fontsize=label_fontsize,
            xytext=threshold_text_pos,
        )
    else:
        fig_title = (
            threshold_label if fig_title is None else f"{fig_title} ({threshold_label})"
        )
    return fig_title
