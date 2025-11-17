import codecs
import datetime
import functools
from io import BytesIO
import logging
import math
import os
import pathlib
import shutil
import subprocess
from tempfile import TemporaryDirectory
import weakref

from PIL import Image

import matplotlib as mpl
from matplotlib import cbook, font_manager as fm
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, RendererBase
)
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.backends.backend_pdf import (
    _create_pdf_info_dict, _datetime_to_pdf)
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib._pylab_helpers import Gcf

_log = logging.getLogger(__name__)


_DOCUMENTCLASS = r"\documentclass{article}"


# Note: When formatting floating point values, it is important to use the
# %f/{:f} format rather than %s/{} to avoid triggering scientific notation,
# which is not recognized by TeX.



# It's better to use only one unit for all coordinates, since the
# arithmetic in latex seems to produce inaccurate conversions.
latex_pt_to_in = 1. / 72.27
latex_in_to_pt = 1. / latex_pt_to_in
mpl_pt_to_in = 1. / 72.
mpl_in_to_pt = 1. / mpl_pt_to_in






















FigureManagerPgf = FigureManagerBase


@_Backend.export
class _BackendPgf(_Backend):
    FigureCanvas = FigureCanvasPgf

