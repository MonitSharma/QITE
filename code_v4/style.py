import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import rc
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, interp1d
import scipy.ndimage
import os
import math


class Style:
    """
    A class to encapsulate styling properties for plots.
    """

    def __init__(self):
        self.name = ''
        self.title = ''
        self.color = ''
        self.pt = ''  # Marker type
        self.ps = ''  # Marker size
        self.fs = ''  # Fill style
        self.lt = ''  # Line type
        self.lw = ''  # Line width
        self.lw2 = ''  # Additional line width

    def fill(self, name, title, color, pt, ps, fs, lt, lw, lw2):
        """
        Fill the style object with the given properties.
        """
        self.name = name
        self.title = title
        self.color = color
        self.pt = pt
        self.ps = ps
        self.fs = fs
        self.lt = lt
        self.lw = lw
        self.lw2 = lw2

    def dataplot(self, target, x, y, dx=None, dy=None, spline=False, connected=False, capsize=5, logscale=None):
        """
        Create a plot with the defined style.

        Parameters:
        - target: Axes object where the plot is drawn.
        - x, y: Data points for the plot.
        - dx, dy: Optional error values.
        - spline: Boolean, whether to use spline interpolation.
        - connected: Boolean, whether to connect points with lines.
        - capsize: Size of the error bar caps.
        - logscale: Set to 'xy' for log-log scaling.
        """
        if logscale == 'xy':
            target.set_xscale("log")
            target.set_yscale("log")

        marker_edge_color = 'black' if self.fs != 'none' else self.color

        if dy is None:
            if not spline:
                target.plot(
                    x, y, label=self.title, marker=self.pt, markersize=self.ps,
                    fillstyle=self.fs, linestyle=(' ' if not connected else self.lt),
                    color=self.color, markeredgecolor=marker_edge_color,
                    mew=self.lw, linewidth=self.lw2
                )
            else:
                spline_func = interp1d(x, y, kind='cubic')
                x_new = np.linspace(min(x), max(x), 2000)
                y_new = spline_func(x_new)
                target.plot(
                    x_new, y_new, label=self.title, linestyle=self.lt,
                    color=self.color, markeredgecolor=marker_edge_color,
                    mew=self.lw, linewidth=self.lw2
                )
        else:
            if not spline:
                errorbar_result = target.errorbar(
                    x, y, yerr=dy, label=self.title, marker=self.pt, markersize=self.ps,
                    fillstyle=self.fs, linestyle=(' ' if not connected else self.lt),
                    color=self.color, markeredgecolor=marker_edge_color,
                    mew=self.lw, linewidth=(0 if not connected else self.lw2),
                    capsize=capsize, elinewidth=self.lw2
                )
                for cap in errorbar_result[1]:  # Adjust caps
                    cap.set_markeredgewidth(self.lw2)
            else:
                spline_func = interp1d(x, y, kind='cubic')
                x_new = np.linspace(min(x), max(x), 2000)
                y_new = spline_func(x_new)
                target.plot(
                    x_new, y_new, label=self.title, linestyle=self.lt,
                    color=self.color, mew=self.lw, linewidth=self.lw2
                )
                errorbar_result = target.errorbar(
                    x, y, yerr=dy, label=self.title, marker=self.pt, markersize=self.ps,
                    fillstyle=self.fs, linestyle=' ', color=self.color,
                    markeredgecolor=marker_edge_color, mew=self.lw,
                    linewidth=self.lw2, capsize=capsize, elinewidth=self.lw2
                )
                for cap in errorbar_result[1]:  # Adjust caps
                    cap.set_markeredgewidth(self.lw2)


def make_legend(panel, order, ncol, bbox_to_anchor, loc, mode, borderaxespad, fontsize, handlelength, numpoints=None):
    """
    Create a legend for the plot.

    Parameters:
    - panel: Axes object where the legend is drawn.
    - order: Order of items in the legend.
    - ncol: Number of columns in the legend.
    - bbox_to_anchor: Legend's position.
    - loc: Location of the legend box.
    - mode: Scaling mode for the legend.
    - borderaxespad: Padding between the legend and the axes.
    - fontsize: Font size for the legend text.
    - handlelength: Length of the handles in the legend.
    - numpoints: Number of points in the legend markers.
    """
    handles, labels = panel.get_legend_handles_labels()
    ordered_handles = [handles[i] for i in order]
    ordered_labels = [labels[i] for i in order]

    return panel.legend(
        ordered_handles, ordered_labels, ncol=ncol, fancybox=True, shadow=True,
        bbox_to_anchor=bbox_to_anchor, loc=loc, mode=mode,
        borderaxespad=borderaxespad, fontsize=fontsize, handlelength=handlelength,
        numpoints=numpoints
    )


def set_canvas(panel, x_label, x_lim, x_ticks, x_tick_labels, y_label, y_lim, y_ticks, y_tick_labels, tick_fontsize, label_fontsize):
    """
    Configure the plot's axes and labels.
    """
    panel.set_xlabel(x_label, fontsize=label_fontsize)
    panel.set_xlim(x_lim)
    panel.set_xticks(x_ticks)
    panel.set_xticklabels(x_tick_labels)
    panel.set_ylabel(y_label, fontsize=label_fontsize)
    panel.set_ylim(y_lim)
    panel.set_yticks(y_ticks)
    panel.set_yticklabels(y_tick_labels)
    panel.tick_params(axis='both', labelsize=tick_fontsize, direction='in')


# Default LaTeX settings for matplotlib
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


colors = {
    "red": "#E42F0C",
    "orange": "#F28500",
    "yellow": "#FDB827",
    "green": "#78BB17",
    "blue": "#146EFF",
    "purple": "#A26BA2",
    "black": "#000000",
    "white": "#FFFFFF",
    "gray": "#999999"
}

# Palette for various plot styles
palette = [Style() for _ in range(200)]

# Filling palette styles
palette[0].fill('c0', r'$\displaystyle D=2$', colors["red"], '.', 0, 'full', '--', 0.2, 2.0)
palette[1].fill('c1', r'$\displaystyle D=4$', colors["green"], '.', 0, 'full', '-.', 0.2, 3.0)
palette[2].fill('c2', r'$\displaystyle D=6$', colors["blue"], '.', 0, 'full', ':', 0.2, 3.0)
palette[3].fill('c3', r'$\displaystyle D=8$', colors["yellow"], '.', 0, 'full', '--', 0.2, 2.0)
palette[4].fill('c3', r'$\displaystyle D=10$', colors["yellow"], '.', 0, 'full', '-.', 0.2, 2.0)
palette[5].fill('c4', r'$\displaystyle \mathrm{exact}$', colors["black"], '.', 0, 'full', '-', 0.2, 2.0)
palette[6].fill('c5', '', colors["black"], 'o', 9, 'full', '-', 0.2, 2.0)
palette[7].fill('c5', '', colors["white"], 'o', 9, 'full', '-', 0.2, 2.0)
palette[8].fill('c4', '', colors["gray"], '.', 0, 'full', ':', 0.2, 0.2)
palette[9].fill('c4', '', colors["black"], '.', 0, 'full', '-', 0.2, 0.2)
palette[10].fill('c4', '', colors["black"], '.', 0, 'full', '-', 0.4, 0.4)

# Vertex and link styles
vertex_in = 6
vertex_out = 7
link = 8
link_in = 9
link_between = 10

palette[11].fill('c0', r'$\displaystyle D=2, \mathrm{QLanczos}$', colors["red"], '.', 0, 'none', '-', 2.0, 2.0)
palette[12].fill('c1', r'$\displaystyle D=4, \mathrm{QLanczos}$', colors["green"], '.', 0, 'none', '-', 2.0, 2.0)
palette[13].fill('c2', r'$\displaystyle D=6, \mathrm{QLanczos}$', colors["blue"], '.', 0, 'none', '-', 2.0, 2.0)
palette[14].fill('c3', r'$\displaystyle D=8, \mathrm{QLanczos}$', colors["orange"], '.', 0, 'none', '-', 2.0, 2.0)
palette[15].fill('c4', r'$\displaystyle \mathrm{exact}$', colors["black"], '.', 0, 'none', '-', 2.0, 2.0)

# Additional Beta styles
palette[16].fill('c0', r'$\displaystyle \beta=0$', colors["red"], '.', 0, 'full', '--', 0.2, 2.0)
palette[17].fill('c1', r'$\displaystyle \beta=1$', colors["green"], '.', 0, 'full', '-.', 0.2, 3.0)
palette[18].fill('c2', r'$\displaystyle \beta=2$', colors["blue"], '.', 0, 'full', ':', 0.2, 3.0)
palette[19].fill('c3', r'$\displaystyle \beta=3$', colors["orange"], '.', 0, 'full', '--', 0.2, 2.0)
