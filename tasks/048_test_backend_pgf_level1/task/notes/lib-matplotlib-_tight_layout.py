"""
Routines to adjust subplot params so that subplots are
nicely fit in the figure. In doing so, only axis labels, tick labels, Axes
titles and offsetboxes that are anchored to Axes are currently considered.

Internally, this module assumes that the margins (left margin, etc.) which are
differences between ``Axes.get_tightbbox`` and ``Axes.bbox`` are independent of
Axes position. This may fail if ``Axes.adjustable`` is ``datalim`` as well as
such cases as when left or right margin are affected by xlabel.
"""

import numpy as np

import matplotlib as mpl
from matplotlib import _api, artist as martist
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox





