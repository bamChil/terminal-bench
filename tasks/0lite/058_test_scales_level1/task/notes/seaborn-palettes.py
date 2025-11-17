import colorsys
from itertools import cycle

import numpy as np
import matplotlib as mpl

from .external import husl

from .utils import desaturate, get_color_cycle
from .colors import xkcd_rgb, crayons
from ._compat import get_colormap


__all__ = ["color_palette", "hls_palette", "husl_palette", "mpl_palette",
           "dark_palette", "light_palette", "diverging_palette",
           "blend_palette", "xkcd_palette", "crayon_palette",
           "cubehelix_palette", "set_color_codes"]


SEABORN_PALETTES = dict(
    deep=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
          "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"],
    deep6=["#4C72B0", "#55A868", "#C44E52",
           "#8172B3", "#CCB974", "#64B5CD"],
    muted=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4",
           "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"],
    muted6=["#4878D0", "#6ACC64", "#D65F5F",
            "#956CB4", "#D5BB67", "#82C6E2"],
    pastel=["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
            "#DEBB9B", "#FAB0E4", "#CFCFCF", "#FFFEA3", "#B9F2F0"],
    pastel6=["#A1C9F4", "#8DE5A1", "#FF9F9B",
             "#D0BBFF", "#FFFEA3", "#B9F2F0"],
    bright=["#023EFF", "#FF7C00", "#1AC938", "#E8000B", "#8B2BE2",
            "#9F4800", "#F14CC1", "#A3A3A3", "#FFC400", "#00D7FF"],
    bright6=["#023EFF", "#1AC938", "#E8000B",
             "#8B2BE2", "#FFC400", "#00D7FF"],
    dark=["#001C7F", "#B1400D", "#12711C", "#8C0800", "#591E71",
          "#592F0D", "#A23582", "#3C3C3C", "#B8850A", "#006374"],
    dark6=["#001C7F", "#12711C", "#8C0800",
           "#591E71", "#B8850A", "#006374"],
    colorblind=["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
                "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"],
    colorblind6=["#0173B2", "#029E73", "#D55E00",
                 "#CC78BC", "#ECE133", "#56B4E9"]
)


MPL_QUAL_PALS = {
    "tab10": 10, "tab20": 20, "tab20b": 20, "tab20c": 20,
    "Set1": 9, "Set2": 8, "Set3": 12,
    "Accent": 8, "Paired": 12,
    "Pastel1": 9, "Pastel2": 8, "Dark2": 8,
}


QUAL_PALETTE_SIZES = MPL_QUAL_PALS.copy()
QUAL_PALETTE_SIZES.update({k: len(v) for k, v in SEABORN_PALETTES.items()})
QUAL_PALETTES = list(QUAL_PALETTE_SIZES.keys())


class _ColorPalette(list):
    """Set the color palette in a with statement, otherwise be a list."""
    def __enter__(self):
        """Open the context."""
        from .rcmod import set_palette
        self._orig_palette = color_palette()
        set_palette(self)
        return self

    def __exit__(self, *args):
        """Close the context."""
        from .rcmod import set_palette
        set_palette(self._orig_palette)

    def as_hex(self):
        """Return a color palette with hex codes instead of RGB values."""
        hex = [mpl.colors.rgb2hex(rgb) for rgb in self]
        return _ColorPalette(hex)

    def _repr_html_(self):
        """Rich display of the color palette in an HTML frontend."""
        s = 55
        n = len(self)
        html = f'<svg  width="{n * s}" height="{s}">'
        for i, c in enumerate(self.as_hex()):
            html += (
                f'<rect x="{i * s}" y="0" width="{s}" height="{s}" style="fill:{c};'
                'stroke-width:2;stroke:rgb(255,255,255)"/>'
            )
        html += '</svg>'
        return html


def _patch_colormap_display():
    """Simplify the rich display of matplotlib color maps in a notebook."""
    def _repr_png_(self):
        """Generate a PNG representation of the Colormap."""
        import io
        from PIL import Image
        import numpy as np
        IMAGE_SIZE = (400, 50)
        X = np.tile(np.linspace(0, 1, IMAGE_SIZE[0]), (IMAGE_SIZE[1], 1))
        pixels = self(X, bytes=True)
        png_bytes = io.BytesIO()
        Image.fromarray(pixels).save(png_bytes, format='png')
        return png_bytes.getvalue()

    def _repr_html_(self):
        """Generate an HTML representation of the Colormap."""
        import base64
        png_bytes = self._repr_png_()
        png_base64 = base64.b64encode(png_bytes).decode('ascii')
        return ('<img '
                + 'alt="' + self.name + ' color map" '
                + 'title="' + self.name + '"'
                + 'src="data:image/png;base64,' + png_base64 + '">')

    mpl.colors.Colormap._repr_png_ = _repr_png_
    mpl.colors.Colormap._repr_html_ = _repr_html_


def color_palette(
        palette = None,
        n_colors = None,
        desat = None,
        as_cmap = False
    ):
    """
    Return a list of colors or continuous colormap defining a palette.

        Possible ``palette`` values include:
            - Name of a seaborn palette (deep, muted, bright, pastel, dark, colorblind)
            - Name of matplotlib colormap
            - 'husl' or 'hls'
            - 'ch:<cubehelix arguments>'
            - 'light:<color>', 'dark:<color>', 'blend:<color>,<color>',
            - A sequence of colors in any format matplotlib accepts

        Calling this function with ``palette=None`` will return the current
        matplotlib color cycle.

        This function can also be used in a ``with`` statement to temporarily
        set the color cycle for a plot or set of plots.

        See the :ref:`tutorial <palette_tutorial>` for more information.

        Parameters
        ----------
        palette : None, string, or sequence, optional
            Name of palette or None to return current palette. If a sequence, input
            colors are used but possibly cycled and desaturated.
        n_colors : int, optional
            Number of colors in the palette. If ``None``, the default will depend
            on how ``palette`` is specified. Named palettes default to 6 colors,
            but grabbing the current palette or passing in a list of colors will
            not change the number of colors unless this is specified. Asking for
            more colors than exist in the palette will cause it to cycle. Ignored
            when ``as_cmap`` is True.
        desat : float, optional
            Proportion to desaturate each color by.
        as_cmap : bool
            If True, return a :class:`matplotlib.colors.ListedColormap`.

        Returns
        -------
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

        See Also
        --------
        set_palette : Set the default color cycle for all plots.
        set_color_codes : Reassign color codes like ``"b"``, ``"g"``, etc. to
                          colors from one of the seaborn palettes.

        Examples
        --------

        .. include:: ../docstrings/color_palette.rst


    """
    raise NotImplementedError('This function has been masked for testing')














def diverging_palette(h_neg, h_pos, s=75, l=50, sep=1, n=6,  # noqa
                      center="light", as_cmap=False):
    """Make a diverging palette between two HUSL colors.

    If you are using the IPython notebook, you can also choose this palette
    interactively with the :func:`choose_diverging_palette` function.

    Parameters
    ----------
    h_neg, h_pos : float in [0, 359]
        Anchor hues for negative and positive extents of the map.
    s : float in [0, 100], optional
        Anchor saturation for both extents of the map.
    l : float in [0, 100], optional
        Anchor lightness for both extents of the map.
    sep : int, optional
        Size of the intermediate region.
    n : int, optional
        Number of colors in the palette (if not returning a cmap)
    center : {"light", "dark"}, optional
        Whether the center of the palette is light or dark
    as_cmap : bool, optional
        If True, return a :class:`matplotlib.colors.ListedColormap`.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    See Also
    --------
    dark_palette : Create a sequential palette with dark values.
    light_palette : Create a sequential palette with light values.

    Examples
    --------
    .. include: ../docstrings/diverging_palette.rst

    """
    palfunc = dict(dark=dark_palette, light=light_palette)[center]
    n_half = int(128 - (sep // 2))
    neg = palfunc((h_neg, s, l), n_half, reverse=True, input="husl")
    pos = palfunc((h_pos, s, l), n_half, input="husl")
    midpoint = dict(light=[(.95, .95, .95)], dark=[(.133, .133, .133)])[center]
    mid = midpoint * sep
    pal = blend_palette(np.concatenate([neg, mid, pos]), n, as_cmap=as_cmap)
    return pal




def xkcd_palette(colors):
    """Make a palette with color names from the xkcd color survey.

    See xkcd for the full list of colors: https://xkcd.com/color/rgb/

    This is just a simple wrapper around the `seaborn.xkcd_rgb` dictionary.

    Parameters
    ----------
    colors : list of strings
        List of keys in the `seaborn.xkcd_rgb` dictionary.

    Returns
    -------
    palette
        A list of colors as RGB tuples.

    See Also
    --------
    crayon_palette : Make a palette with Crayola crayon colors.

    """
    palette = [xkcd_rgb[name] for name in colors]
    return color_palette(palette, len(palette))


def crayon_palette(colors):
    """Make a palette with color names from Crayola crayons.

    Colors are taken from here:
    https://en.wikipedia.org/wiki/List_of_Crayola_crayon_colors

    This is just a simple wrapper around the `seaborn.crayons` dictionary.

    Parameters
    ----------
    colors : list of strings
        List of keys in the `seaborn.crayons` dictionary.

    Returns
    -------
    palette
        A list of colors as RGB tuples.

    See Also
    --------
    xkcd_palette : Make a palette with named colors from the XKCD color survey.

    """
    palette = [crayons[name] for name in colors]
    return color_palette(palette, len(palette))






def set_color_codes(palette="deep"):
    """Change how matplotlib color shorthands are interpreted.

    Calling this will change how shorthand codes like "b" or "g"
    are interpreted by matplotlib in subsequent plots.

    Parameters
    ----------
    palette : {deep, muted, pastel, dark, bright, colorblind}
        Named seaborn palette to use as the source of colors.

    See Also
    --------
    set : Color codes can be set through the high-level seaborn style
          manager.
    set_palette : Color codes can also be set through the function that
                  sets the matplotlib color cycle.

    """
    if palette == "reset":
        colors = [
            (0., 0., 1.),
            (0., .5, 0.),
            (1., 0., 0.),
            (.75, 0., .75),
            (.75, .75, 0.),
            (0., .75, .75),
            (0., 0., 0.)
        ]
    elif not isinstance(palette, str):
        err = "set_color_codes requires a named seaborn palette"
        raise TypeError(err)
    elif palette in SEABORN_PALETTES:
        if not palette.endswith("6"):
            palette = palette + "6"
        colors = SEABORN_PALETTES[palette] + [(.1, .1, .1)]
    else:
        err = f"Cannot set colors with palette '{palette}'"
        raise ValueError(err)

    for code, color in zip("bgrmyck", colors):
        rgb = mpl.colors.colorConverter.to_rgb(color)
        mpl.colors.colorConverter.colors[code] = rgb