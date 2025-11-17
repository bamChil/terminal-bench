"""
A PDF Matplotlib backend.

Author: Jouni K Seppänen <jks@iki.fi> and others.
"""

import codecs
from datetime import timezone
from datetime import datetime
from enum import Enum
from functools import total_ordering
from io import BytesIO
import itertools
import logging
import math
import os
import string
import struct
import sys
import time
import types
import warnings
import zlib

import numpy as np
from PIL import Image

import matplotlib as mpl
from matplotlib import _api, _text_helpers, _type1font, cbook, dviread
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase,
    RendererBase)
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.figure import Figure
from matplotlib.font_manager import get_font, fontManager as _fontManager
from matplotlib._afm import AFM
from matplotlib.ft2font import FT2Font, FaceFlags, Kerning, LoadFlags, StyleFlags
from matplotlib.transforms import Affine2D, BboxBase
from matplotlib.path import Path
from matplotlib.dates import UTC
from matplotlib import _path
from . import _backend_pdf_ps

_log = logging.getLogger(__name__)

# Overview
#
# The low-level knowledge about pdf syntax lies mainly in the pdfRepr
# function and the classes Reference, Name, Operator, and Stream.  The
# PdfFile class knows about the overall structure of pdf documents.
# It provides a "write" method for writing arbitrary strings in the
# file, and an "output" method that passes objects through the pdfRepr
# function before writing them in the file.  The output method is
# called by the RendererPdf class, which contains the various draw_foo
# methods.  RendererPdf contains a GraphicsContextPdf instance, and
# each draw_foo calls self.check_gc before outputting commands.  This
# method checks whether the pdf graphics state needs to be modified
# and outputs the necessary commands.  GraphicsContextPdf represents
# the graphics state, and its "delta" method returns the commands that
# modify the state.

# Add "pdf.use14corefonts: True" in your configuration file to use only
# the 14 PDF core fonts. These fonts do not need to be embedded; every
# PDF viewing application is required to have them. This results in very
# light PDF files you can use directly in LaTeX or ConTeXt documents
# generated with pdfTeX, without any conversion.

# These fonts are: Helvetica, Helvetica-Bold, Helvetica-Oblique,
# Helvetica-BoldOblique, Courier, Courier-Bold, Courier-Oblique,
# Courier-BoldOblique, Times-Roman, Times-Bold, Times-Italic,
# Times-BoldItalic, Symbol, ZapfDingbats.
#
# Some tricky points:
#
# 1. The clip path can only be widened by popping from the state
# stack.  Thus the state must be pushed onto the stack before narrowing
# the clip path.  This is taken care of by GraphicsContextPdf.
#
# 2. Sometimes it is necessary to refer to something (e.g., font,
# image, or extended graphics state, which contains the alpha value)
# in the page stream by a name that needs to be defined outside the
# stream.  PdfFile provides the methods fontName, imageObject, and
# alphaState for this purpose.  The implementations of these methods
# should perhaps be generalized.

# TODOs:
#
# * encoding of fonts, including mathtext fonts and Unicode support
# * TTF support has lots of small TODOs, e.g., how do you know if a font
#   is serif/sans-serif, or symbolic/non-symbolic?
# * draw_quad_mesh














# PDF strings are supposed to be able to include any eight-bit data, except
# that unbalanced parens and backslashes must be escaped by a backslash.
# However, sf bug #2708559 shows that the carriage return character may get
# read as a newline; these characters correspond to \gamma and \Omega in TeX's
# math font encoding. Escaping them fixes the bug.
_str_escapes = str.maketrans({
    '\\': '\\\\', '(': '\\(', ')': '\\)', '\n': '\\n', '\r': '\\r'})


















class FigureCanvasPdf(FigureCanvasBase):
    # docstring inherited

    fixed_dpi = 72
    filetypes = {'pdf': 'Portable Document Format'}

    def get_default_filetype(self):
        return 'pdf'

    def print_pdf(self, filename, *,
                  bbox_inches_restore=None, metadata=None):

        dpi = self.figure.dpi
        self.figure.dpi = 72  # there are 72 pdf points to an inch
        width, height = self.figure.get_size_inches()
        if isinstance(filename, PdfPages):
            file = filename._ensure_file()
        else:
            file = PdfFile(filename, metadata=metadata)
        try:
            file.newPage(width, height)
            renderer = MixedModeRenderer(
                self.figure, width, height, dpi,
                RendererPdf(file, dpi, height, width),
                bbox_inches_restore=bbox_inches_restore)
            self.figure.draw(renderer)
            renderer.finalize()
            if not isinstance(filename, PdfPages):
                file.finalize()
        finally:
            if isinstance(filename, PdfPages):  # finish off this page
                file.endStream()
            else:            # we opened the file above; now finish it off
                file.close()

    def draw(self):
        self.figure.draw_without_rendering()
        return super().draw()


FigureManagerPdf = FigureManagerBase


@_Backend.export
class _BackendPdf(_Backend):
    FigureCanvas = FigureCanvasPdf