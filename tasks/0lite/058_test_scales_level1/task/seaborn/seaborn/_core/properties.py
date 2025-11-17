from __future__ import annotations
import itertools
import warnings

import numpy as np
from numpy.typing import ArrayLike
from pandas import Series
import matplotlib as mpl
from matplotlib.colors import to_rgb, to_rgba, to_rgba_array
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path

from seaborn._core.scales import Scale, Boolean, Continuous, Nominal, Temporal
from seaborn._core.rules import categorical_order, variable_type
from seaborn.palettes import QUAL_PALETTES, color_palette, blend_palette
from seaborn.utils import get_color_cycle

from typing import Any, Callable, Tuple, List, Union, Optional

RGBTuple = Tuple[float, float, float]
RGBATuple = Tuple[float, float, float, float]
ColorSpec = Union[RGBTuple, RGBATuple, str]

DashPattern = Tuple[float, ...]
DashPatternWithOffset = Tuple[float, Optional[DashPattern]]

MarkerPattern = Union[
    float,
    str,
    Tuple[int, int, float],
    List[Tuple[float, float]],
    Path,
    MarkerStyle,
]

Mapping = Callable[[ArrayLike], ArrayLike]


# =================================================================================== #
# Base classes
# =================================================================================== #




# =================================================================================== #
# Properties relating to spatial position of marks on the plotting axes
# =================================================================================== #


class Coordinate(Property):
    """
    The position of visual marks with respect to the axes of the plot.
    """

    legend = False
    normed = False
    raise NotImplementedError('This class has been masked for testing')


# =================================================================================== #
# Properties with numeric values where scale range can be defined as an interval
# =================================================================================== #


class IntervalProperty(Property):
    """
    A numeric property where scale range can be defined as an interval.
    """

    legend = True
    normed = True
    _default_range = (0, 1)

    @property
    def default_range(self) -> tuple[float, float]:
        """
        Min and max values used by default for semantic mapping.
        """
        raise NotImplementedError('This function has been masked for testing')

    def _forward(self, values: ArrayLike) -> ArrayLike:
        """
        Transform applied to native values before linear mapping into interval.
        """
        raise NotImplementedError('This function has been masked for testing')

    def _inverse(self, values: ArrayLike) -> ArrayLike:
        """
        Transform applied to results of mapping that returns to native values.
        """
        raise NotImplementedError('This function has been masked for testing')

    def infer_scale(self, arg: Any, data: Series) -> Scale:
        """
        Given data and a scaling argument, initialize appropriate scale class.
        """
        raise NotImplementedError('This function has been masked for testing')

    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        """
        Return a function that maps from data domain to property range.
        """
        raise NotImplementedError('This function has been masked for testing')

    def _get_nominal_mapping(self, scale: Nominal, data: Series) -> Mapping:
        """
        Identify evenly-spaced values using interval or explicit mapping.
        """
        raise NotImplementedError('This function has been masked for testing')

    def _get_boolean_mapping(self, scale: Boolean, data: Series) -> Mapping:
        """
        Identify evenly-spaced values using interval or explicit mapping.
        """
        raise NotImplementedError('This function has been masked for testing')

    def _get_values(self, scale: Scale, levels: list) -> list:
        """
        Validate scale.values and identify a value for each level.
        """
        raise NotImplementedError('This function has been masked for testing')


class PointSize(IntervalProperty):
    """Size (diameter) of a point mark, in points, with scaling by area."""
    _default_range = 2, 8  # TODO use rcparams?

    def _forward(self, values):
        """Square native values to implement linear scaling of point area."""
        return np.square(values)

    def _inverse(self, values):
        """Invert areal values back to point diameter."""
        return np.sqrt(values)


class LineWidth(IntervalProperty):
    """Thickness of a line mark, in points."""
    @property
    def default_range(self) -> tuple[float, float]:
        """Min and max values used by default for semantic mapping."""
        base = mpl.rcParams["lines.linewidth"]
        return base * .5, base * 2


class EdgeWidth(IntervalProperty):
    """Thickness of the edges on a patch mark, in points."""
    @property
    def default_range(self) -> tuple[float, float]:
        """Min and max values used by default for semantic mapping."""
        base = mpl.rcParams["patch.linewidth"]
        return base * .5, base * 2


class Stroke(IntervalProperty):
    """Thickness of lines that define point glyphs."""
    _default_range = .25, 2.5


class Alpha(IntervalProperty):
    """
    Opacity of the color values for an arbitrary mark.
    """

    _default_range = (0.3, 0.95)
    raise NotImplementedError('This class has been masked for testing')
    # TODO validate / enforce that output is in [0, 1]


class Offset(IntervalProperty):
    """Offset for edge-aligned text, in point units."""
    _default_range = 0, 5
    _legend = False


class FontSize(IntervalProperty):
    """Font size for textual marks, in points."""
    _legend = False

    @property
    def default_range(self) -> tuple[float, float]:
        """Min and max values used by default for semantic mapping."""
        base = mpl.rcParams["font.size"]
        return base * .5, base * 2


# =================================================================================== #
# Properties defined by arbitrary objects with inherently nominal scaling
# =================================================================================== #


class ObjectProperty(Property):
    """
    A property defined by arbitrary an object, with inherently nominal scaling.
    """

    legend = True
    normed = False
    null_value = None

    def _default_values(self, n: int) -> list:
        raise NotImplementedError('This function has been masked for testing')

    def default_scale(self, data: Series) -> Scale:
        raise NotImplementedError('This function has been masked for testing')

    def infer_scale(self, arg: Any, data: Series) -> Scale:
        raise NotImplementedError('This function has been masked for testing')

    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        """
        Define mapping as lookup into list of object values.
        """
        raise NotImplementedError('This function has been masked for testing')

    def _get_values(self, scale: Scale, levels: list) -> list:
        """
        Validate scale.values and identify a value for each level.
        """
        raise NotImplementedError('This function has been masked for testing')


class Marker(ObjectProperty):
    """Shape of points in scatter-type marks or lines with data points marked."""
    null_value = MarkerStyle("")

    # TODO should we have named marker "palettes"? (e.g. see d3 options)

    # TODO need some sort of "require_scale" functionality
    # to raise when we get the wrong kind explicitly specified

    def standardize(self, val: MarkerPattern) -> MarkerStyle:
        return MarkerStyle(val)

    def _default_values(self, n: int) -> list[MarkerStyle]:
        """Build an arbitrarily long list of unique marker styles.

        Parameters
        ----------
        n : int
            Number of unique marker specs to generate.

        Returns
        -------
        markers : list of string or tuples
            Values for defining :class:`matplotlib.markers.MarkerStyle` objects.
            All markers will be filled.

        """
        # Start with marker specs that are well distinguishable
        markers = [
            "o", "X", (4, 0, 45), "P", (4, 0, 0), (4, 1, 0), "^", (4, 1, 45), "v",
        ]

        # Now generate more from regular polygons of increasing order
        s = 5
        while len(markers) < n:
            a = 360 / (s + 1) / 2
            markers.extend([(s + 1, 1, a), (s + 1, 0, a), (s, 1, 0), (s, 0, 0)])
            s += 1

        markers = [MarkerStyle(m) for m in markers[:n]]

        return markers


class LineStyle(ObjectProperty):
    """Dash pattern for line-type marks."""
    null_value = ""

    def standardize(self, val: str | DashPattern) -> DashPatternWithOffset:
        return self._get_dash_pattern(val)

    def _default_values(self, n: int) -> list[DashPatternWithOffset]:
        """Build an arbitrarily long list of unique dash styles for lines.

        Parameters
        ----------
        n : int
            Number of unique dash specs to generate.

        Returns
        -------
        dashes : list of strings or tuples
            Valid arguments for the ``dashes`` parameter on
            :class:`matplotlib.lines.Line2D`. The first spec is a solid
            line (``""``), the remainder are sequences of long and short
            dashes.

        """
        # Start with dash specs that are well distinguishable
        dashes: list[str | DashPattern] = [
            "-", (4, 1.5), (1, 1), (3, 1.25, 1.5, 1.25), (5, 1, 1, 1),
        ]

        # Now programmatically build as many as we need
        p = 3
        while len(dashes) < n:

            # Take combinations of long and short dashes
            a = itertools.combinations_with_replacement([3, 1.25], p)
            b = itertools.combinations_with_replacement([4, 1], p)

            # Interleave the combinations, reversing one of the streams
            segment_list = itertools.chain(*zip(list(a)[1:-1][::-1], list(b)[1:-1]))

            # Now insert the gaps
            for segments in segment_list:
                gap = min(segments)
                spec = tuple(itertools.chain(*((seg, gap) for seg in segments)))
                dashes.append(spec)

            p += 1

        return [self._get_dash_pattern(x) for x in dashes]

    @staticmethod
    def _get_dash_pattern(style: str | DashPattern) -> DashPatternWithOffset:
        """Convert linestyle arguments to dash pattern with offset."""
        # Copied and modified from Matplotlib 3.4
        # go from short hand -> full strings
        ls_mapper = {"-": "solid", "--": "dashed", "-.": "dashdot", ":": "dotted"}
        if isinstance(style, str):
            style = ls_mapper.get(style, style)
            # un-dashed styles
            if style in ["solid", "none", "None"]:
                offset = 0
                dashes = None
            # dashed styles
            elif style in ["dashed", "dashdot", "dotted"]:
                offset = 0
                dashes = tuple(mpl.rcParams[f"lines.{style}_pattern"])
            else:
                options = [*ls_mapper.values(), *ls_mapper.keys()]
                msg = f"Linestyle string must be one of {options}, not {repr(style)}."
                raise ValueError(msg)

        elif isinstance(style, tuple):
            if len(style) > 1 and isinstance(style[1], tuple):
                offset, dashes = style
            elif len(style) > 1 and style[1] is None:
                offset, dashes = style
            else:
                offset = 0
                dashes = style
        else:
            val_type = type(style).__name__
            msg = f"Linestyle must be str or tuple, not {val_type}."
            raise TypeError(msg)

        # Normalize offset to be positive and shorter than the dash cycle
        if dashes is not None:
            try:
                dsum = sum(dashes)
            except TypeError as err:
                msg = f"Invalid dash pattern: {dashes}"
                raise TypeError(msg) from err
            if dsum:
                offset %= dsum

        return offset, dashes


class TextAlignment(ObjectProperty):
    legend = False


class HorizontalAlignment(TextAlignment):

    def _default_values(self, n: int) -> list:
        vals = itertools.cycle(["left", "right"])
        return [next(vals) for _ in range(n)]


class VerticalAlignment(TextAlignment):

    def _default_values(self, n: int) -> list:
        vals = itertools.cycle(["top", "bottom"])
        return [next(vals) for _ in range(n)]


# =================================================================================== #
# Properties with  RGB(A) color values
# =================================================================================== #


class Color(Property):
    """
    Color, as RGB(A), scalable with nominal palettes or continuous gradients.
    """

    legend = True
    normed = True

    def standardize(self, val: ColorSpec) -> RGBTuple | RGBATuple:
        raise NotImplementedError('This function has been masked for testing')

    def _standardize_color_sequence(self, colors: ArrayLike) -> ArrayLike:
        """
        Convert color sequence to RGB(A) array, preserving but not adding alpha.
        """
        raise NotImplementedError('This function has been masked for testing')

    def infer_scale(self, arg: Any, data: Series) -> Scale:
        raise NotImplementedError('This function has been masked for testing')

    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        """
        Return a function that maps from data domain to color values.
        """
        raise NotImplementedError('This function has been masked for testing')

    def _get_nominal_mapping(self, scale: Nominal, data: Series) -> Mapping:
        raise NotImplementedError('This function has been masked for testing')

    def _get_boolean_mapping(self, scale: Boolean, data: Series) -> Mapping:
        raise NotImplementedError('This function has been masked for testing')

    def _get_values(self, scale: Scale, levels: list) -> ArrayLike:
        """
        Validate scale.values and identify a value for each level.
        """
        raise NotImplementedError('This function has been masked for testing')


# =================================================================================== #
# Properties that can take only two states
# =================================================================================== #


class Fill(Property):
    """
    Boolean property of points/bars/patches that can be solid or outlined.
    """

    legend = True
    normed = False

    def default_scale(self, data: Series) -> Scale:
        raise NotImplementedError('This function has been masked for testing')

    def infer_scale(self, arg: Any, data: Series) -> Scale:
        raise NotImplementedError('This function has been masked for testing')

    def standardize(self, val: Any) -> bool:
        raise NotImplementedError('This function has been masked for testing')

    def _default_values(self, n: int) -> list:
        """
        Return a list of n values, alternating True and False.
        """
        raise NotImplementedError('This function has been masked for testing')

    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        """
        Return a function that maps each data value to True or False.
        """
        raise NotImplementedError('This function has been masked for testing')

    def _get_values(self, scale: Scale, levels: list) -> list:
        """
        Validate scale.values and identify a value for each level.
        """
        raise NotImplementedError('This function has been masked for testing')


# =================================================================================== #
# Enumeration of properties for use by Plot and Mark classes
# =================================================================================== #
# TODO turn this into a property registry with hooks, etc.
# TODO Users do not interact directly with properties, so how to document them?


PROPERTY_CLASSES = {
    "x": Coordinate,
    "y": Coordinate,
    "color": Color,
    "alpha": Alpha,
    "fill": Fill,
    "marker": Marker,
    "pointsize": PointSize,
    "stroke": Stroke,
    "linewidth": LineWidth,
    "linestyle": LineStyle,
    "fillcolor": Color,
    "fillalpha": Alpha,
    "edgewidth": EdgeWidth,
    "edgestyle": LineStyle,
    "edgecolor": Color,
    "edgealpha": Alpha,
    "text": Property,
    "halign": HorizontalAlignment,
    "valign": VerticalAlignment,
    "offset": Offset,
    "fontsize": FontSize,
    "xmin": Coordinate,
    "xmax": Coordinate,
    "ymin": Coordinate,
    "ymax": Coordinate,
    "group": Property,
    # TODO pattern?
    # TODO gradient?
}

PROPERTIES = {var: cls(var) for var, cls in PROPERTY_CLASSES.items()}