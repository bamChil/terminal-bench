"""Control plot style and scaling using the matplotlib rcParams interface."""
import functools
import matplotlib as mpl
from cycler import cycler
from . import palettes


__all__ = ["set_theme", "set", "reset_defaults", "reset_orig",
           "axes_style", "set_style", "plotting_context", "set_context",
           "set_palette"]


_style_keys = [

    "axes.facecolor",
    "axes.edgecolor",
    "axes.grid",
    "axes.axisbelow",
    "axes.labelcolor",

    "figure.facecolor",

    "grid.color",
    "grid.linestyle",

    "text.color",

    "xtick.color",
    "ytick.color",
    "xtick.direction",
    "ytick.direction",
    "lines.solid_capstyle",

    "patch.edgecolor",
    "patch.force_edgecolor",

    "image.cmap",
    "font.family",
    "font.sans-serif",

    "xtick.bottom",
    "xtick.top",
    "ytick.left",
    "ytick.right",

    "axes.spines.left",
    "axes.spines.bottom",
    "axes.spines.right",
    "axes.spines.top",

]

_context_keys = [

    "font.size",
    "axes.labelsize",
    "axes.titlesize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
    "legend.title_fontsize",

    "axes.linewidth",
    "grid.linewidth",
    "lines.linewidth",
    "lines.markersize",
    "patch.linewidth",

    "xtick.major.width",
    "ytick.major.width",
    "xtick.minor.width",
    "ytick.minor.width",

    "xtick.major.size",
    "ytick.major.size",
    "xtick.minor.size",
    "ytick.minor.size",

]


def set_theme(context="notebook", style="darkgrid", palette="deep",
              font="sans-serif", font_scale=1, color_codes=True, rc=None):
    """
    Set aspects of the visual theme for all matplotlib and seaborn plots.

    This function changes the global defaults for all plots using the
    matplotlib rcParams system. The themeing is decomposed into several distinct
    sets of parameter values.

    The options are illustrated in the :doc:`aesthetics <../tutorial/aesthetics>`
    and :doc:`color palette <../tutorial/color_palettes>` tutorials.

    Parameters
    ----------
    context : string or dict
        Scaling parameters, see :func:`plotting_context`.
    style : string or dict
        Axes style parameters, see :func:`axes_style`.
    palette : string or sequence
        Color palette, see :func:`color_palette`.
    font : string
        Font family, see matplotlib font manager.
    font_scale : float, optional
        Separate scaling factor to independently scale the size of the
        font elements.
    color_codes : bool
        If ``True`` and ``palette`` is a seaborn palette, remap the shorthand
        color codes (e.g. "b", "g", "r", etc.) to the colors from this palette.
    rc : dict or None
        Dictionary of rc parameter mappings to override the above.

    Examples
    --------

    .. include:: ../docstrings/set_theme.rst

    """
    set_context(context, font_scale)
    set_style(style, rc={"font.family": font})
    set_palette(palette, color_codes=color_codes)
    if rc is not None:
        mpl.rcParams.update(rc)


def set(*args, **kwargs):
    """
    Alias for :func:`set_theme`, which is the preferred interface.

    This function may be removed in the future.
    """
    set_theme(*args, **kwargs)


def reset_defaults():
    """Restore all RC params to default settings."""
    mpl.rcParams.update(mpl.rcParamsDefault)


def reset_orig():
    """Restore all RC params to original settings (respects custom rc)."""
    from . import _orig_rc_params
    mpl.rcParams.update(_orig_rc_params)




def set_style(style=None, rc=None):
    """
    Set the parameters that control the general style of the plots.

    The style parameters control properties like the color of the background and
    whether a grid is enabled by default. This is accomplished using the
    matplotlib rcParams system.

    The options are illustrated in the
    :doc:`aesthetics tutorial <../tutorial/aesthetics>`.

    See :func:`axes_style` to get the parameter values.

    Parameters
    ----------
    style : dict, or one of {darkgrid, whitegrid, dark, white, ticks}
        A dictionary of parameters or the name of a preconfigured style.
    rc : dict, optional
        Parameter mappings to override the values in the preset seaborn
        style dictionaries. This only updates parameters that are
        considered part of the style definition.

    Examples
    --------

    .. include:: ../docstrings/set_style.rst

    """
    style_object = axes_style(style, rc)
    mpl.rcParams.update(style_object)




def set_context(context=None, font_scale=1, rc=None):
    """
    Set the parameters that control the scaling of plot elements.

    These parameters correspond to label size, line thickness, etc.
    Calling this function modifies the global matplotlib `rcParams`. For more
    information, see the :doc:`aesthetics tutorial <../tutorial/aesthetics>`.

    The base context is "notebook", and the other contexts are "paper", "talk",
    and "poster", which are version of the notebook parameters scaled by different
    values. Font elements can also be scaled independently of (but relative to)
    the other values.

    See :func:`plotting_context` to get the parameter values.

    Parameters
    ----------
    context : dict, or one of {paper, notebook, talk, poster}
        A dictionary of parameters or the name of a preconfigured set.
    font_scale : float, optional
        Separate scaling factor to independently scale the size of the
        font elements.
    rc : dict, optional
        Parameter mappings to override the values in the preset seaborn
        context dictionaries. This only updates parameters that are
        considered part of the context definition.

    Examples
    --------

    .. include:: ../docstrings/set_context.rst

    """
    context_object = plotting_context(context, font_scale, rc)
    mpl.rcParams.update(context_object)


class _RCAesthetics(dict):
    def __enter__(self):
        rc = mpl.rcParams
        self._orig = {k: rc[k] for k in self._keys}
        self._set(self)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._set(self._orig)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


class _AxesStyle(_RCAesthetics):
    """Light wrapper on a dict to set style temporarily."""
    _keys = _style_keys
    _set = staticmethod(set_style)


class _PlottingContext(_RCAesthetics):
    """Light wrapper on a dict to set context temporarily."""
    _keys = _context_keys
    _set = staticmethod(set_context)


def set_palette(palette, n_colors=None, desat=None, color_codes=False):
    """Set the matplotlib color cycle using a seaborn palette.

    Parameters
    ----------
    palette : seaborn color palette | matplotlib colormap | hls | husl
        Palette definition. Should be something :func:`color_palette` can process.
    n_colors : int
        Number of colors in the cycle. The default number of colors will depend
        on the format of ``palette``, see the :func:`color_palette`
        documentation for more information.
    desat : float
        Proportion to desaturate each color by.
    color_codes : bool
        If ``True`` and ``palette`` is a seaborn palette, remap the shorthand
        color codes (e.g. "b", "g", "r", etc.) to the colors from this palette.

    See Also
    --------
    color_palette : build a color palette or set the color cycle temporarily
                    in a ``with`` statement.
    set_context : set parameters to scale plot elements
    set_style : set the default parameters for figure style

    """
    colors = palettes.color_palette(palette, n_colors, desat)
    cyl = cycler('color', colors)
    mpl.rcParams['axes.prop_cycle'] = cyl
    if color_codes:
        try:
            palettes.set_color_codes(palette)
        except (ValueError, TypeError):
            pass