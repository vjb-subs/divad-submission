"""Reporting visualization module.
"""
import os

import matplotlib.pyplot as plt


def plot_boxplots(dfs_dict, *, fig_title=None, full_output_path=None):
    """Plots the set of boxplots corresponding to the provided `dfs_dict`.

    Args:
        dfs_dict: dictionary of pd.DataFrames from which to derive the boxplots, of the form `{ax_title: df}`.
        fig_title (str|None): optional title for the figure.
        full_output_path (str|None): path to save the figure to if specified (with file name and extension).
    """
    # setup font sizes, create figure and set title
    fontsizes_dict, n_boxplots = {"title": 15}, len(dfs_dict)
    fig, axs = plt.subplots(n_boxplots, 1, sharex="none")
    # set figure size according to the number of boxplots and maximum number of boxes
    max_n_boxes = max([df.max().notnull().sum() for df in dfs_dict.values()])
    fig.set_size_inches(max(7, int(max_n_boxes * 7 / 4)), 5 * n_boxplots)
    if fig_title is not None:
        fig.suptitle(fig_title, size=fontsizes_dict["title"], y=0.96, fontweight="bold")

    # plot boxplot(s)
    if n_boxplots == 1:
        plot_boxplot(list(dfs_dict.values())[0], title=list(dfs_dict.keys())[0], ax=axs)
    else:
        i = 0
        for title, df in dfs_dict.items():
            plot_boxplot(df, title=title, ax=axs[i])
            i += 1

    # save the figure as an image if specified
    if full_output_path is not None:
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        fig.savefig(full_output_path)
        plt.close()


def plot_boxplot(df, *, title=None, ax=None):
    """Plots the boxplot corresponding to the provided `df`.

    Args:
        df (pd.DataFrame): pd.DataFrame from which to derive the boxplot.
        title (str|None): optional title for the axis.
        ax (AxesSubplot): optional plt.axis to plot the boxplot on if not in a standalone figure.
    """
    # setup font sizes, create figure and set title
    fontsizes_dict = {"title": 15, "ticks": 13}
    if ax is None:
        # standalone figure
        plt.figure(figsize=(7, 5))
        ax = plt.axes()
    if title is not None:
        ax.set_title(title, fontsize=fontsizes_dict["title"], y=1.02)
    ax.tick_params(axis="both", which="major", labelsize=fontsizes_dict["ticks"])
    ax.tick_params(axis="both", which="minor", labelsize=fontsizes_dict["ticks"])

    # plot boxplot (update column names to include number of considered points)
    n_points_df = df.notnull().sum()
    df.columns = [f"{c} ({n_points_df[c]})" for c in df.columns]
    ax = df.boxplot(ax=ax)
    # derive constant x offset of texts based on the number of boxes
    n_boxes = df.max().notnull().sum()
    if n_boxes > 3:
        # slightly rotate x labels if more than 3 boxes
        for t in ax.get_xticklabels():
            t.set_rotation(20)
    max_text_x_offset = min_text_x_offset = 0.15 * n_boxes / 4
    median_text_x_offset = 0.25 * n_boxes / 4
    for box_x, box_max, box_median, box_min in zip(
        range(1, n_boxes + 1), df.max(), df.median(), df.min()
    ):
        if box_max is not None:
            # others will not be None either
            max_pos, median_pos, min_pos = (
                (box_x, box_max),
                (box_x, box_median),
                (box_x, box_min),
            )
            max_text_pos = (max_pos[0] + max_text_x_offset, max_pos[1])
            min_text_pos = (min_pos[0] + min_text_x_offset, min_pos[1])
            median_text_pos = (median_pos[0] - median_text_x_offset, median_pos[1])
            ax.annotate(
                f"{box_max:.2f}",
                max_pos,
                xytext=max_text_pos,
                weight="bold",
                color="red",
                fontsize=fontsizes_dict["ticks"],
                ha="left",
                va="center",
            )
            ax.annotate(
                f"{box_median:.2f}",
                median_pos,
                xytext=median_text_pos,
                weight="bold",
                color="green",
                fontsize=fontsizes_dict["ticks"],
                ha="right",
                va="center",
            )
            ax.annotate(
                f"{box_min:.2f}",
                min_pos,
                xytext=min_text_pos,
                weight="bold",
                color="blue",
                fontsize=fontsizes_dict["ticks"],
                ha="left",
                va="center",
            )
