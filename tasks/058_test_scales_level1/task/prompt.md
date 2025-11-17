## Task
## Statistical Data Visualization Framework

**Core Functionality:**
Build a declarative plotting system that maps data variables to visual properties (position, color, size, etc.) through configurable scales and renders them using various mark types across single or multiple subplots.

**Main Features & Requirements:**
- **Declarative Interface**: Construct plots by specifying data mappings and visual layers rather than imperative drawing commands
- **Layered Architecture**: Support multiple visual layers with different marks, statistical transformations, and data sources
- **Flexible Data Mapping**: Map data columns to visual properties (coordinates, aesthetics) with automatic or custom scale inference
- **Subplot Generation**: Create subplot grids through faceting (conditional data subsets) or pairing (variable combinations)
- **Scale Management**: Handle various data types (continuous, categorical, temporal, boolean) with appropriate visual mappings
- **Property Standardization**: Convert diverse input formats (colors, markers, etc.) to consistent internal representations

**Key Challenges:**
- **Type Inference**: Automatically detect appropriate scales and mappings based on data characteristics
- **Flexible Input Handling**: Accept multiple data specification formats while maintaining consistent internal processing
- **Visual Property Management**: Standardize diverse visual property inputs (color names/codes, marker styles) into uniform formats
- **Layout Coordination**: Manage complex subplot arrangements with proper axis sharing, scaling, and legend placement
- **Backend Integration**: Seamlessly interface with matplotlib while providing higher-level abstractions

**NOTE**: 
- This test comes from the `seaborn` library, and we have given you the content of this code repository under `/testbed/`, and you need to complete based on this code repository and supplement the files we specify. Remember, all your changes must be in this codebase, and changes that are not in this codebase will not be discovered and tested by us.
- What's more, you need to install `pytest, pytest-timeout, pytest-json-report` in your environment, otherwise our tests won't run and you'll get **ZERO POINTS**!

Your available resources are listed below:
- `/workspace/task/black_links.txt`: Prohibited URLs (all other web resources are allowed)


## Precautions
- You may need to install some of the libraries to support you in accomplishing our task, some of the packages are already pre-installed in your environment, you can check them out yourself via `pip list` etc. For standard installs, just run `pip install <package>`. There's no need to add `--index-url`, the domestic mirrors are already set up unless you have special requirements.
- Please note that when running `pip install <package>`, you should not include the `--force-reinstall` flag, as it may cause pre-installed packages to be reinstalled.
- **IMPORTANT**: While you can install libraries using pip, you should never access the actual implementations in the libraries you install, as the tasks we give you originate from github, and if you look at the contents of the libraries, it could result in you being awarded 0 points directly for alleged cheating. Specifically, you cannot read any files under `/usr/local/lib/python3.x` and its subfolders (here python3.x means any version of python).
- **IMPORTANT**: Your installed python library may contain a real implementation of the task, and you are prohibited from directly calling the library's interface of the same name and pretending to package it as your answer, which will also be detected and awarded 0 points.
- **CRITICAL REQUIREMENT**: After completing the task, pytest will be used to test your implementation. **YOU MUST**:
    - Build proper code hierarchy with correct import relationships shown in **Test Description** (I will give you this later)
    - Match the exact interface shown in the **Interface Description** (I will give you this later)
- I will tell you details about **CRITICAL REQUIREMENT** below.

Your final deliverable should be code under the `/testbed/` directory, and after completing the codebase, we will evaluate your completion and it is important that you complete our tasks with integrity and precision
The final structure is like below, note that  your codebase's structure should match import structure in **Test Description**, which I will tell you later.
```
/workspace
├── task/
│   ├── prompt.md          # task statement
│   ├── black_links.txt    # black links you can't access
│   ├── ...
├── test/                  # you won't see this dir
│   ├── ...
/testbed                   # all your work should be put into this codebase and match the specific dir structure
├── dir1/
│   ├── file1.py
│   ├── ...
├── dir2/
```

## Test and Interface Descriptions

The **Test Description** will tell you the position of the function or class which we're testing should satisfy.
This means that when you generate some files and complete the functionality we want to test in the files, you need to put these files in the specified directory, otherwise our tests won't be able to import your generated.
For example, if the **Test Description** show you this:
```python
from seaborn._core.plot import Plot
```
This means that we will test one function/class: Plot.
And the defination and implementation of class Plot should be in `/testbed/seaborn/_core/plot.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
class Plot:
    """
    
        An interface for declaratively specifying statistical graphics.
    
        Plots are constructed by initializing this class and adding one or more
        layers, comprising a `Mark` and optional `Stat` or `Move`.  Additionally,
        faceting variables or variable pairings may be defined to divide the space
        into multiple subplots. The mappings from data values to visual properties
        can be parametrized using scales, although the plot will try to infer good
        defaults when scales are not explicitly defined.
    
        The constructor accepts a data source (a :class:`pandas.DataFrame` or
        dictionary with columnar values) and variable assignments. Variables can be
        passed as keys to the data source or directly as data vectors.  If multiple
        data-containing objects are provided, they will be index-aligned.
    
        The data source and variables defined in the constructor will be used for
        all layers in the plot, unless overridden or disabled when adding a layer.
    
        The following variables can be defined in the constructor:
            {known_properties}
    
        The `data`, `x`, and `y` variables can be passed as positional arguments or
        using keywords. Whether the first positional argument is interpreted as a
        data source or `x` variable depends on its type.
    
        The methods of this class return a copy of the instance; use chaining to
        build up a plot through multiple calls. Methods can be called in any order.
    
        Most methods only add information to the plot spec; no actual processing
        happens until the plot is shown or saved. It is also possible to compile
        the plot without rendering it to access the lower-level representation.
    
        
    """

    config = "PlotConfig()"
    _data: PlotData
    _layers: list[Layer]
    _scales: dict[str, Scale]
    _shares: dict[str, bool | str]
    _limits: dict[str, tuple[Any, Any]]
    _labels: dict[str, str | Callable[[str], str]]
    _theme: dict[str, Any]
    _facet_spec: FacetSpec
    _pair_spec: PairSpec
    _figure_spec: dict[str, Any]
    _subplot_spec: dict[str, Any]
    _layout_spec: dict[str, Any]

    def __init__(
        self,
        *args: DataSource | VariableSpec,
        **variables: VariableSpec
    ):
        """
        Initialize a Plot object with data and variable mappings.

        This constructor sets up the foundation for a declarative statistical visualization.
        It accepts a data source and variable assignments that will be used across all
        layers in the plot unless overridden when adding individual layers.

        Parameters
        ----------
        *args : DataSource or VariableSpec
            Positional arguments that can contain:
            - A data source (DataFrame, dict, or object with __dataframe__ method) as first argument
            - x variable as first argument (if no data source provided) or second argument
            - y variable as second or third argument
            Maximum of 3 positional arguments allowed: (data, x, y)
        data : DataSource, optional
            The primary data source for the plot. Can be a pandas DataFrame, dictionary
            with columnar values, or any object implementing the __dataframe__ protocol.
            If provided as a positional argument, this parameter should not be used.
        **variables : VariableSpec
            Keyword arguments defining variable mappings. Keys must correspond to valid
            plot properties (coordinate variables like 'x', 'y' or semantic variables
            like 'color', 'size', etc.). Values can be column names/keys referencing
            the data source, or direct data vectors.

        Raises
        ------
        TypeError
            If more than 3 positional arguments are provided, if both positional and
            keyword arguments are given for the same variable (data, x, or y), or if
            unknown variable names are provided that don't correspond to valid plot
            properties.

        Notes
        -----
        - The first positional argument is interpreted as a data source if it's a mapping,
          DataFrame, or has a __dataframe__ method; otherwise it's treated as the x variable
        - Variables defined here become the default for all plot layers but can be
          overridden or disabled when adding individual layers via the add() method
        - Multiple data-containing objects will be index-aligned if provided
        - The constructor only stores the plot specification; no actual processing
          occurs until the plot is rendered via show(), save(), or plot() methods
        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from seaborn._core.plot import Plot
from seaborn._core.properties import IntervalProperty
from seaborn._core.properties import ObjectProperty
from seaborn._core.properties import Coordinate
from seaborn._core.properties import Alpha
from seaborn._core.properties import Color
from seaborn._core.properties import Fill
from seaborn.palettes import color_palette
from seaborn.utils import _version_predates
```

### Interface Description 1
Below is **Interface Description 1** for file: seaborn-_core-plot.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
@build_plot_signature
class Plot:
    """
    
        An interface for declaratively specifying statistical graphics.
    
        Plots are constructed by initializing this class and adding one or more
        layers, comprising a `Mark` and optional `Stat` or `Move`.  Additionally,
        faceting variables or variable pairings may be defined to divide the space
        into multiple subplots. The mappings from data values to visual properties
        can be parametrized using scales, although the plot will try to infer good
        defaults when scales are not explicitly defined.
    
        The constructor accepts a data source (a :class:`pandas.DataFrame` or
        dictionary with columnar values) and variable assignments. Variables can be
        passed as keys to the data source or directly as data vectors.  If multiple
        data-containing objects are provided, they will be index-aligned.
    
        The data source and variables defined in the constructor will be used for
        all layers in the plot, unless overridden or disabled when adding a layer.
    
        The following variables can be defined in the constructor:
            {known_properties}
    
        The `data`, `x`, and `y` variables can be passed as positional arguments or
        using keywords. Whether the first positional argument is interpreted as a
        data source or `x` variable depends on its type.
    
        The methods of this class return a copy of the instance; use chaining to
        build up a plot through multiple calls. Methods can be called in any order.
    
        Most methods only add information to the plot spec; no actual processing
        happens until the plot is shown or saved. It is also possible to compile
        the plot without rendering it to access the lower-level representation.
    
        
    """

    config = "PlotConfig()"
    _data: PlotData
    _layers: list[Layer]
    _scales: dict[str, Scale]
    _shares: dict[str, bool | str]
    _limits: dict[str, tuple[Any, Any]]
    _labels: dict[str, str | Callable[[str], str]]
    _theme: dict[str, Any]
    _facet_spec: FacetSpec
    _pair_spec: PairSpec
    _figure_spec: dict[str, Any]
    _subplot_spec: dict[str, Any]
    _layout_spec: dict[str, Any]

    def __init__(
        self,
        *args: DataSource | VariableSpec,
        **variables: VariableSpec
    ):
        """
        Initialize a Plot object with data and variable mappings.

        This constructor sets up the foundation for a declarative statistical visualization.
        It accepts a data source and variable assignments that will be used across all
        layers in the plot unless overridden when adding individual layers.

        Parameters
        ----------
        *args : DataSource or VariableSpec
            Positional arguments that can contain:
            - A data source (DataFrame, dict, or object with __dataframe__ method) as first argument
            - x variable as first argument (if no data source provided) or second argument
            - y variable as second or third argument
            Maximum of 3 positional arguments allowed: (data, x, y)
        data : DataSource, optional
            The primary data source for the plot. Can be a pandas DataFrame, dictionary
            with columnar values, or any object implementing the __dataframe__ protocol.
            If provided as a positional argument, this parameter should not be used.
        **variables : VariableSpec
            Keyword arguments defining variable mappings. Keys must correspond to valid
            plot properties (coordinate variables like 'x', 'y' or semantic variables
            like 'color', 'size', etc.). Values can be column names/keys referencing
            the data source, or direct data vectors.

        Raises
        ------
        TypeError
            If more than 3 positional arguments are provided, if both positional and
            keyword arguments are given for the same variable (data, x, or y), or if
            unknown variable names are provided that don't correspond to valid plot
            properties.

        Notes
        -----
        - The first positional argument is interpreted as a data source if it's a mapping,
          DataFrame, or has a __dataframe__ method; otherwise it's treated as the x variable
        - Variables defined here become the default for all plot layers but can be
          overridden or disabled when adding individual layers via the add() method
        - Multiple data-containing objects will be index-aligned if provided
        - The constructor only stores the plot specification; no actual processing
          occurs until the plot is rendered via show(), save(), or plot() methods
        """
        <your code>

    def _resolve_positionals(
        self,
        args: tuple[DataSource | VariableSpec, ...],
        data: DataSource,
        variables: dict[str, VariableSpec]
    ) -> tuple[DataSource, dict[str, VariableSpec]]:
        """
        Handle positional arguments, which may contain data / x / y.
        """
        <your code>

    def __add__(self, other):
        """
        Implement addition operation for Plot objects.

        This method is called when the `+` operator is used with a Plot object on the left side.
        It provides helpful error messages to guide users away from ggplot-style syntax toward
        the correct seaborn objects interface.

        Parameters
        ----------
        other : object
            The object being added to the Plot instance. Common incorrect usage includes
            attempting to add Mark or Stat objects directly.

        Raises
        ------
        TypeError
            Always raised with specific guidance based on the type of `other`:
            - If `other` is a Mark or Stat instance, suggests using Plot.add() instead
            - For any other type, indicates that the operation is not supported

        Notes
        -----
        This method exists primarily for user experience, intercepting common mistakes
        from users familiar with ggplot2's `+` syntax. In seaborn's objects interface,
        layers should be added using the `add()` method rather than the `+` operator.

        The method intentionally never returns a valid result, always raising TypeError
        to redirect users to the correct API.

        Examples
        --------
        Incorrect usage that will raise TypeError:
            plot = Plot(data, x="x", y="y")
            plot + Mark()  # Raises TypeError with helpful message

        Correct usage:
            plot = Plot(data, x="x", y="y")
            plot.add(Mark())  # Proper way to add layers
        """
        <your code>

    def _repr_png_(self) -> tuple[bytes, dict[str, float]] | None:
        """
        Generate a PNG representation of the plot for Jupyter notebook display.

        This method is automatically called by Jupyter's rich display system when the plot
        object is the last expression in a cell and the display format is set to PNG.

        Returns
        -------
        tuple[bytes, dict[str, float]] | None
            A tuple containing the PNG image data as bytes and metadata dictionary with
            width and height information, or None if the display format is not set to PNG.
            The metadata dictionary contains:
            - "width": Image width in points, scaled according to display settings
            - "height": Image height in points, scaled according to display settings

        Notes
        -----
        - This method only returns data when Plot.config.display["format"] is "png"
        - The image resolution and scaling are controlled by Plot.config.display settings:
          - "hidpi": When True, doubles DPI while preserving display size
          - "scaling": Relative scaling factor for the embedded image
        - The method uses matplotlib's savefig functionality with tight bounding box
        - This is part of IPython's rich display protocol and should not typically be
          called directly by users

        See Also
        --------
        _repr_svg_ : SVG representation method for vector graphics display
        show : Method to explicitly display the plot using pyplot
        save : Method to save the plot to a file or buffer
        """
        <your code>

    def _repr_svg_(self) -> str | None:
        """
        Generate SVG representation of the plot for rich display in Jupyter notebooks.

        This method is automatically called by Jupyter's display system when the plot
        object is the last expression in a cell and the display format is configured
        for SVG output. It compiles the plot specification and returns the SVG string
        representation.

        Returns
        -------
        str or None
            SVG string representation of the plot if the display format is configured
            for "svg", otherwise None. The SVG includes proper scaling based on the
            Plot.config.display["scaling"] setting and uses "tight" bbox for optimal
            layout.

        Notes
        -----
        - This method is part of Jupyter's rich display protocol and should not
          typically be called directly by users
        - The method only returns SVG content when Plot.config.display["format"] 
          is set to "svg", otherwise it returns None to allow other display methods
          to handle the output
        - The SVG output respects the scaling factor defined in the plot configuration
        - Uses matplotlib's SVG backend with "tight" bounding box for clean output
        - The plot is compiled using the current theme settings via _theme_with_defaults()
        """
        <your code>

    def _clone(self) -> Plot:
        """
        Generate a new object with the same information as the current spec.
        """
        <your code>

    def _theme_with_defaults(self) -> dict[str, Any]:
        """
        Generate a theme dictionary with default values merged with instance-specific theme settings.

        This method combines the global default theme configuration from the Plot class
        configuration with any theme customizations that have been applied to this specific
        Plot instance.

        Returns
        -------
        dict[str, Any]
            A dictionary containing matplotlib rc parameters that define the plot's visual
            theme. This includes default styling parameters (axes style, plotting context,
            color cycles, etc.) merged with any custom theme settings specified for this
            Plot instance.

        Notes
        -----
        The returned dictionary contains matplotlib rc parameters that control various
        aspects of plot appearance including:
        - Axes styling (grid, spines, ticks)
        - Figure properties (size, DPI, background)
        - Font settings (family, size, weight)
        - Color cycles and palettes
        - Line and marker properties
        - Legend styling

        Instance-specific theme settings (from self._theme) take precedence over
        the default configuration values when there are conflicts.
        """
        <your code>

    @property
    def _variables(self) -> list[str]:
        """
        Property that returns a list of all variable names used in the plot.

        This property collects variable names from multiple sources within the plot
        specification, including the main data frame, pair specification variables,
        facet specification variables, and layer-specific variables.

        Returns
        -------
        list[str]
            A list of all variable names (as strings) used across all components
            of the plot. This includes:
            - Variables from the main data frame
            - Variables defined in pair specifications (for paired plots)
            - Variables defined in facet specifications (for faceted plots)
            - Layer-specific variables from all layers

            Variables are deduplicated, so each unique variable name appears only
            once in the returned list, regardless of how many plot components use it.

        Notes
        -----
        The return type is explicitly cast to list[str] to satisfy type checking,
        as DataFrame column names could theoretically be other types, but in practice
        will always be strings in the seaborn plotting context.

        This property is used internally for scale setup and other operations that
        need to know all variables present in the plot specification.
        """
        <your code>

    def on(
        self,
        target: Axes | SubFigure | Figure
    ) -> Plot:
        """

                Provide existing Matplotlib figure or axes for drawing the plot.

                When using this method, you will also need to explicitly call a method that
                triggers compilation, such as :meth:`Plot.show` or :meth:`Plot.save`. If you
                want to postprocess using matplotlib, you'd need to call :meth:`Plot.plot`
                first to compile the plot without rendering it.

                Parameters
                ----------
                target : Axes, SubFigure, or Figure
                    Matplotlib object to use. Passing :class:`matplotlib.axes.Axes` will add
                    artists without otherwise modifying the figure. Otherwise, subplots will be
                    created within the space of the given :class:`matplotlib.figure.Figure` or
                    :class:`matplotlib.figure.SubFigure`.

                Examples
                --------
                .. include:: ../docstrings/objects.Plot.on.rst


        """
        <your code>

    def add(
        self,
        mark: Mark,
        *transforms: Stat | Move,
        **variables: VariableSpec
    ) -> Plot:
        """

                Specify a layer of the visualization in terms of mark and data transform(s).

                This is the main method for specifying how the data should be visualized.
                It can be called multiple times with different arguments to define
                a plot with multiple layers.

                Parameters
                ----------
                mark : :class:`Mark`
                    The visual representation of the data to use in this layer.
                transforms : :class:`Stat` or :class:`Move`
                    Objects representing transforms to be applied before plotting the data.
                    Currently, at most one :class:`Stat` can be used, and it
                    must be passed first. This constraint will be relaxed in the future.
                orient : "x", "y", "v", or "h"
                    The orientation of the mark, which also affects how transforms are computed.
                    Typically corresponds to the axis that defines groups for aggregation.
                    The "v" (vertical) and "h" (horizontal) options are synonyms for "x" / "y",
                    but may be more intuitive with some marks. When not provided, an
                    orientation will be inferred from characteristics of the data and scales.
                legend : bool
                    Option to suppress the mark/mappings for this layer from the legend.
                label : str
                    A label to use for the layer in the legend, independent of any mappings.
                data : DataFrame or dict
                    Data source to override the global source provided in the constructor.
                variables : data vectors or identifiers
                    Additional layer-specific variables, including variables that will be
                    passed directly to the transforms without scaling.

                Examples
                --------
                .. include:: ../docstrings/objects.Plot.add.rst


        """
        <your code>

    def pair(
        self,
        x: VariableSpecList = None,
        y: VariableSpecList = None,
        wrap: int | None = None,
        cross: bool = True
    ) -> Plot:
        """

                Produce subplots by pairing multiple `x` and/or `y` variables.

                Parameters
                ----------
                x, y : sequence(s) of data vectors or identifiers
                    Variables that will define the grid of subplots.
                wrap : int
                    When using only `x` or `y`, "wrap" subplots across a two-dimensional grid
                    with this many columns (when using `x`) or rows (when using `y`).
                cross : bool
                    When False, zip the `x` and `y` lists such that the first subplot gets the
                    first pair, the second gets the second pair, etc. Otherwise, create a
                    two-dimensional grid from the cartesian product of the lists.

                Examples
                --------
                .. include:: ../docstrings/objects.Plot.pair.rst


        """
        <your code>

    def facet(
        self,
        col: VariableSpec = None,
        row: VariableSpec = None,
        order: OrderSpec | dict[str, OrderSpec] = None,
        wrap: int | None = None
    ) -> Plot:
        """

                Produce subplots with conditional subsets of the data.

                Parameters
                ----------
                col, row : data vectors or identifiers
                    Variables used to define subsets along the columns and/or rows of the grid.
                    Can be references to the global data source passed in the constructor.
                order : list of strings, or dict with dimensional keys
                    Define the order of the faceting variables.
                wrap : int
                    When using only `col` or `row`, wrap subplots across a two-dimensional
                    grid with this many subplots on the faceting dimension.

                Examples
                --------
                .. include:: ../docstrings/objects.Plot.facet.rst


        """
        <your code>

    def scale(self, **scales: Scale) -> Plot:
        """

                Specify mappings from data units to visual properties.

                Keywords correspond to variables defined in the plot, including coordinate
                variables (`x`, `y`) and semantic variables (`color`, `pointsize`, etc.).

                A number of "magic" arguments are accepted, including:
                    - The name of a transform (e.g., `"log"`, `"sqrt"`)
                    - The name of a palette (e.g., `"viridis"`, `"muted"`)
                    - A tuple of values, defining the output range (e.g. `(1, 5)`)
                    - A dict, implying a :class:`Nominal` scale (e.g. `{"a": .2, "b": .5}`)
                    - A list of values, implying a :class:`Nominal` scale (e.g. `["b", "r"]`)

                For more explicit control, pass a scale spec object such as :class:`Continuous`
                or :class:`Nominal`. Or pass `None` to use an "identity" scale, which treats
                data values as literally encoding visual properties.

                Examples
                --------
                .. include:: ../docstrings/objects.Plot.scale.rst


        """
        <your code>

    def share(self, **shares: bool | str) -> Plot:
        """

                Control sharing of axis limits and ticks across subplots.

                Keywords correspond to variables defined in the plot, and values can be
                boolean (to share across all subplots), or one of "row" or "col" (to share
                more selectively across one dimension of a grid).

                Behavior for non-coordinate variables is currently undefined.

                Examples
                --------
                .. include:: ../docstrings/objects.Plot.share.rst


        """
        <your code>

    def limit(
        self,
        **limits: tuple[Any, Any]
    ) -> Plot:
        """

                Control the range of visible data.

                Keywords correspond to variables defined in the plot, and values are a
                `(min, max)` tuple (where either can be `None` to leave unset).

                Limits apply only to the axis; data outside the visible range are
                still used for any stat transforms and added to the plot.

                Behavior for non-coordinate variables is currently undefined.

                Examples
                --------
                .. include:: ../docstrings/objects.Plot.limit.rst


        """
        <your code>

    def label(
        self,
        **variables: str | Callable[[str], str]
    ) -> Plot:
        """

                Control the labels and titles for axes, legends, and subplots.

                Additional keywords correspond to variables defined in the plot.
                Values can be one of the following types:

                - string (used literally; pass "" to clear the default label)
                - function (called on the default label)

                For coordinate variables, the value sets the axis label.
                For semantic variables, the value sets the legend title.
                For faceting variables, `title=` modifies the subplot-specific label,
                while `col=` and/or `row=` add a label for the faceting variable.

                When using a single subplot, `title=` sets its title.

                The `legend=` parameter sets the title for the "layer" legend
                (i.e., when using `label` in :meth:`Plot.add`).

                Examples
                --------
                .. include:: ../docstrings/objects.Plot.label.rst



        """
        <your code>

    def layout(self) -> Plot:
        """

                Control the figure size and layout.

                .. note::

                    Default figure sizes and the API for specifying the figure size are subject
                    to change in future "experimental" releases of the objects API. The default
                    layout engine may also change.

                Parameters
                ----------
                size : (width, height)
                    Size of the resulting figure, in inches. Size is inclusive of legend when
                    using pyplot, but not otherwise.
                engine : {{"tight", "constrained", "none"}}
                    Name of method for automatically adjusting the layout to remove overlap.
                    The default depends on whether :meth:`Plot.on` is used.
                extent : (left, bottom, right, top)
                    Boundaries of the plot layout, in fractions of the figure size. Takes
                    effect through the layout engine; exact results will vary across engines.
                    Note: the extent includes axis decorations when using a layout engine,
                    but it is exclusive of them when `engine="none"`.

                Examples
                --------
                .. include:: ../docstrings/objects.Plot.layout.rst


        """
        <your code>

    def theme() -> Plot:
        """

                Control the appearance of elements in the plot.

                .. note::

                    The API for customizing plot appearance is not yet finalized.
                    Currently, the only valid argument is a dict of matplotlib rc parameters.
                    (This dict must be passed as a positional argument.)

                    It is likely that this method will be enhanced in future releases.

                Matplotlib rc parameters are documented on the following page:
                https://matplotlib.org/stable/tutorials/introductory/customizing.html

                Examples
                --------
                .. include:: ../docstrings/objects.Plot.theme.rst


        """
        <your code>

    def save(self, loc, **kwargs) -> Plot:
        """

                Compile the plot and write it to a buffer or file on disk.

                Parameters
                ----------
                loc : str, path, or buffer
                    Location on disk to save the figure, or a buffer to write into.
                kwargs
                    Other keyword arguments are passed through to
                    :meth:`matplotlib.figure.Figure.savefig`.


        """
        <your code>

    def show(self, **kwargs) -> None:
        """

                Compile the plot and display it by hooking into pyplot.

                Calling this method is not necessary to render a plot in notebook context,
                but it may be in other environments (e.g., in a terminal). After compiling the
                plot, it calls :func:`matplotlib.pyplot.show` (passing any keyword parameters).

                Unlike other :class:`Plot` methods, there is no return value. This should be
                the last method you call when specifying a plot.


        """
        <your code>

    def plot(self, pyplot: bool = False) -> Plotter:
        """

                Compile the plot spec and return the Plotter object.

        """
        <your code>

    def _plot(self, pyplot: bool = False) -> Plotter:
        """
        Generate a plot visualization by compiling the plot specification.

        This is an internal method that handles the core compilation process of converting
        the declarative plot specification into a Plotter object containing the actual
        matplotlib figure and artists. It processes data transformations, applies scales,
        creates subplots, and renders all plot layers.

        Parameters
        ----------
        pyplot : bool, default False
            Whether to integrate the resulting plot with pyplot's state machine.
            When True, the plot will be managed by pyplot and can be displayed
            using pyplot.show(). When False, the plot exists independently of
            pyplot's global state.

        Returns
        -------
        Plotter
            A Plotter object containing the compiled matplotlib figure with all
            plot elements rendered. The Plotter provides methods for saving,
            displaying, and accessing the underlying matplotlib objects.

        Notes
        -----
        This method performs the complete plot compilation pipeline:
        1. Extracts and processes data from all layers
        2. Sets up the figure and subplot structure
        3. Configures scales for coordinate and semantic variables  
        4. Applies statistical transformations
        5. Renders marks for each layer
        6. Creates legends and finalizes figure layout

        The compilation is performed within the context of the plot's theme settings.
        This method is typically called internally by other Plot methods like show(),
        save(), and plot(), rather than being called directly by users.

        The pyplot parameter affects how the resulting figure integrates with
        matplotlib's global state, but does not change the visual output of the plot.
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 2
Below is **Interface Description 2** for file: seaborn-_core-properties.py

This file contains 4 top-level interface(s) that need to be implemented.

```python
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
        <your code>

    def _forward(self, values: ArrayLike) -> ArrayLike:
        """
        Transform applied to native values before linear mapping into interval.
        """
        <your code>

    def _inverse(self, values: ArrayLike) -> ArrayLike:
        """
        Transform applied to results of mapping that returns to native values.
        """
        <your code>

    def infer_scale(self, arg: Any, data: Series) -> Scale:
        """
        Given data and a scaling argument, initialize appropriate scale class.
        """
        <your code>

    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        """
        Return a function that maps from data domain to property range.
        """
        <your code>

    def _get_nominal_mapping(self, scale: Nominal, data: Series) -> Mapping:
        """
        Identify evenly-spaced values using interval or explicit mapping.
        """
        <your code>

    def _get_boolean_mapping(self, scale: Boolean, data: Series) -> Mapping:
        """
        Identify evenly-spaced values using interval or explicit mapping.
        """
        <your code>

    def _get_values(self, scale: Scale, levels: list) -> list:
        """
        Validate scale.values and identify a value for each level.
        """
        <your code>

class ObjectProperty(Property):
    """
    A property defined by arbitrary an object, with inherently nominal scaling.
    """

    legend = True
    normed = False
    null_value = None

    def _default_values(self, n: int) -> list:
        """
        Generate a list of n default values for this property.

        This method creates a sequence of default values that can be used when no explicit
        values are provided for a property scale. The specific values generated depend on
        the property type and are designed to be visually distinguishable when used in
        plots.

        Parameters
        ----------
        n : int
            The number of default values to generate. Must be a positive integer.

        Returns
        -------
        list
            A list containing n default values appropriate for this property type.
            The exact type and content of the values depends on the specific property
            subclass implementation.

        Raises
        ------
        NotImplementedError
            This base method must be implemented by subclasses. Each property type
            defines its own logic for generating appropriate default values.

        Notes
        -----
        This is an abstract method in the ObjectProperty base class that must be
        overridden by concrete property implementations. Different property types
        will generate different kinds of default values:

        - Marker properties generate distinguishable marker styles
        - LineStyle properties generate different dash patterns  
        - Color properties generate color palettes
        - Other properties generate appropriate sequences for their domain

        The generated values are typically designed to cycle through visually
        distinct options that work well for categorical data visualization.
        """
        <your code>

    def default_scale(self, data: Series) -> Scale:
        """
        Determine the appropriate scale type for the given data series.

        This method analyzes the data type and characteristics of the input series to automatically
        select the most suitable scale class for mapping data values to visual properties.

        Parameters
        ----------
        data : Series
            A pandas Series containing the data values to be analyzed. The data can be of
            various types including numeric, datetime, boolean, or categorical.

        Returns
        -------
        Scale
            An instance of the appropriate Scale subclass based on the data type:
            - Continuous: for numeric data
            - Temporal: for datetime data  
            - Boolean: for boolean data with strict boolean typing
            - Nominal: for categorical or other non-numeric data types

        Notes
        -----
        The scale selection is based on the variable_type function which performs type
        inference on the input data. Boolean data is handled with strict_boolean=True,
        meaning only actual boolean values (not boolean-like integers) are classified
        as boolean type.

        This method serves as the default scale inference mechanism when no explicit
        scale is provided by the user. The returned scale object can then be used to
        create appropriate mappings between data values and visual properties.
        """
        <your code>

    def infer_scale(self, arg: Any, data: Series) -> Scale:
        """
        Infer the appropriate scale type for a visual property based on the scaling argument and data characteristics.

        This method analyzes the provided scaling argument and determines the most suitable scale class to use for mapping data values to visual property values. It handles various types of scaling arguments including transformation strings and validates the argument type.

        Parameters
        ----------
        arg : Any
            The scaling argument that specifies how data should be mapped to the visual property.
            Can be a string specifying transformations (e.g., "log", "symlog", "logit", "pow", "sqrt")
            or other scale-specific arguments.
        data : Series
            The pandas Series containing the data values that will be scaled. Used to determine
            appropriate scale characteristics but may not be directly used in all cases.

        Returns
        -------
        Scale
            An initialized Scale object appropriate for the given argument and data type.
            Returns a Continuous scale with the specified transformation for valid transform
            strings.

        Raises
        ------
        ValueError
            If the scaling argument is a string that doesn't match any known transformation
            patterns. The error message indicates the unknown argument and the property
            variable name.
        TypeError
            If the scaling argument is not a string type. Only string arguments are accepted
            for magic scaling arguments.

        Notes
        -----
        This is the base implementation that handles transformation strings. Subclasses
        may override this method to handle property-specific scaling arguments (e.g.,
        color palettes for Color properties) before falling back to this base behavior.

        The supported transformation strings are: "log", "symlog", "logit", "pow", "sqrt".
        These can be used as prefixes (e.g., "log10", "pow2") for parameterized transforms.
        """
        <your code>

    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        """
        Define mapping as lookup into list of object values.
        """
        <your code>

    def _get_values(self, scale: Scale, levels: list) -> list:
        """
        Validate scale.values and identify a value for each level.
        """
        <your code>

class Color(Property):
    """
    Color, as RGB(A), scalable with nominal palettes or continuous gradients.
    """

    legend = True
    normed = True

    def standardize(self, val: ColorSpec) -> RGBTuple | RGBATuple:
        """
        Standardize a color specification to a consistent RGB or RGBA tuple format.

        This method converts various color input formats (color names, hex codes, RGB tuples, 
        RGBA tuples) into a standardized tuple representation that can be used consistently 
        throughout the plotting system.

        Parameters
        ----------
        val : ColorSpec
            A color specification that can be one of:
            - str: Named color (e.g., 'red', 'blue') or hex color code (e.g., '#FF0000')
            - RGBTuple: 3-element tuple of float values (r, g, b) in range [0, 1]
            - RGBATuple: 4-element tuple of float values (r, g, b, a) in range [0, 1]

        Returns
        -------
        RGBTuple or RGBATuple
            Standardized color representation as a tuple of float values in range [0, 1].
            Returns RGBTuple (3 elements) if the input color has no alpha channel or 
            alpha equals 1.0. Returns RGBATuple (4 elements) if the input color 
            explicitly specifies an alpha value different from 1.0.

        Notes
        -----
        The method preserves alpha channel information only when it is explicitly 
        specified and differs from full opacity (1.0). This design allows RGBA colors 
        to override the Alpha property in the plotting system while avoiding 
        unnecessary alpha channels for fully opaque colors.

        The conversion relies on matplotlib's color conversion functions (to_rgb, to_rgba) 
        which handle the parsing of various color specification formats and ensure 
        consistent output ranges.
        """
        <your code>

    def _standardize_color_sequence(self, colors: ArrayLike) -> ArrayLike:
        """
        Convert color sequence to RGB(A) array, preserving but not adding alpha.
        """
        <your code>

    def infer_scale(self, arg: Any, data: Series) -> Scale:
        """
        Infer the appropriate scale type for a visual property based on the scaling argument and data characteristics.

        This method analyzes the provided scaling argument and determines the most suitable scale class to use for mapping data values to visual property values. It handles various types of scaling arguments including transformation strings and validates the argument type.

        Parameters
        ----------
        arg : Any
            The scaling argument that specifies how data should be mapped to the visual property.
            Can be a string specifying transformations (e.g., "log", "symlog", "logit", "pow", "sqrt")
            or other scale-specific arguments.
        data : Series
            The pandas Series containing the data values that will be scaled. Used to determine
            appropriate scale characteristics but may not be directly used in all cases.

        Returns
        -------
        Scale
            An initialized Scale object appropriate for the given argument and data type.
            Returns a Continuous scale with the specified transformation for valid transform
            strings.

        Raises
        ------
        ValueError
            If the scaling argument is a string that doesn't match any known transformation
            patterns. The error message indicates the unknown argument and the property
            variable name.
        TypeError
            If the scaling argument is not a string type. Only string arguments are accepted
            for magic scaling arguments.

        Notes
        -----
        This is the base implementation that handles transformation strings. Subclasses
        may override this method to handle property-specific scaling arguments (e.g.,
        color palettes for Color properties) before falling back to this base behavior.

        The supported transformation strings are: "log", "symlog", "logit", "pow", "sqrt".
        These can be used as prefixes (e.g., "log10", "pow2") for parameterized transforms.
        """
        <your code>

    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        """
        Return a function that maps from data domain to color values.
        """
        <your code>

    def _get_nominal_mapping(self, scale: Nominal, data: Series) -> Mapping:
        """
        Generate a mapping function for nominal scale data to property values.

        This method creates a mapping function that converts nominal (categorical) data values
        to their corresponding property values. The mapping handles categorical ordering and
        provides appropriate fallback values for invalid/missing data.

        Parameters
        ----------
        scale : Nominal
            A nominal scale object that defines the mapping configuration, including
            any explicit value assignments and categorical ordering preferences.
        data : Series
            A pandas Series containing the categorical data to be mapped. Used to
            determine the categorical levels and their order.

        Returns
        -------
        Mapping
            A callable function that takes an array-like input of categorical indices
            and returns an array of corresponding property values. The returned function
            handles NaN/invalid values by filling them with NaN in the output array.

        Notes
        -----
        - The method uses categorical_order() to determine the proper ordering of levels
          based on the scale's order parameter and the data's natural ordering
        - Invalid or NaN input values are preserved as NaN in the output
        - The mapping function expects input values to be numeric indices corresponding
          to categorical levels, not the original categorical values
        - For nominal scales, values are typically evenly spaced across the property's
          range unless explicitly specified in the scale configuration

        See Also
        --------
        _get_boolean_mapping : Similar mapping function for boolean scales
        _get_values : Helper method to validate and extract values for each level
        """
        <your code>

    def _get_boolean_mapping(self, scale: Boolean, data: Series) -> Mapping:
        """
        Generate a mapping function for boolean-scaled data to property values.

        This method creates a mapping function that converts boolean data values to 
        corresponding property values based on the scale configuration. The mapping 
        handles True/False values and maps them to appropriate visual property values.

        Parameters
        ----------
        scale : Boolean
            A Boolean scale object that defines how boolean data should be mapped
            to property values. Contains configuration such as scale.values which
            can be a list, dict, or None to determine the mapping behavior.
        data : Series
            A pandas Series containing the boolean data to be mapped. Used to
            determine the levels/categories present in the data for mapping.

        Returns
        -------
        Mapping
            A callable function that takes an array-like input of boolean values
            and returns an array of corresponding property values. The returned
            function handles NaN values by setting them to NaN in the output,
            and uses np.where to efficiently map True/False values to their
            corresponding property values.

        Notes
        -----
        - The mapping function uses np.where(x[use], *values) where values is a
          2-element list corresponding to [True_value, False_value]
        - NaN or infinite input values are preserved as NaN in the output
        - The scale.values are processed through _get_values() which handles
          dict, list, or None value specifications
        - This method is specifically designed for IntervalProperty subclasses
          where boolean data needs to be mapped to numeric intervals
        """
        <your code>

    def _get_values(self, scale: Scale, levels: list) -> ArrayLike:
        """
        Validate scale.values and identify a value for each level.
        """
        <your code>

class Fill(Property):
    """
    Boolean property of points/bars/patches that can be solid or outlined.
    """

    legend = True
    normed = False

    def default_scale(self, data: Series) -> Scale:
        """
            Determine the appropriate scale type for the given data series.

            This method analyzes the data type of the input series to select the suitable scale class:
            - Boolean: For boolean data types (when identified by variable_type)
            - Nominal: For all other data types (including numeric, datetime, categorical, etc.)

            Parameters
            ----------
            data : Series
                A pandas Series containing the data values to be analyzed.

            Returns
            -------
            Scale
                An instance of Boolean scale if data is boolean type, otherwise Nominal scale.

            Notes
            -----
            The method uses `variable_type(data, boolean_type="boolean", strict_boolean=True)`
            to determine if the data is boolean. All non-boolean data types are mapped to Nominal scale.
    """
        <your code>

    def infer_scale(self, arg: Any, data: Series) -> Scale:
        """
        Infer the appropriate scale type for a visual property based on the scaling argument and data characteristics.

        This method analyzes the provided scaling argument and determines the most suitable scale class to use for mapping data values to visual property values. It handles various types of scaling arguments including transformation strings and validates the argument type.

        Parameters
        ----------
        arg : Any
            The scaling argument that specifies how data should be mapped to the visual property.
            Can be a string specifying transformations (e.g., "log", "symlog", "logit", "pow", "sqrt")
            or other scale-specific arguments.
        data : Series
            The pandas Series containing the data values that will be scaled. Used to determine
            appropriate scale characteristics but may not be directly used in all cases.

        Returns
        -------
        Scale
            An initialized Scale object appropriate for the given argument and data type.
            Returns a Continuous scale with the specified transformation for valid transform
            strings.

        Raises
        ------
        ValueError
            If the scaling argument is a string that doesn't match any known transformation
            patterns. The error message indicates the unknown argument and the property
            variable name.
        TypeError
            If the scaling argument is not a string type. Only string arguments are accepted
            for magic scaling arguments.

        Notes
        -----
        This is the base implementation that handles transformation strings. Subclasses
        may override this method to handle property-specific scaling arguments (e.g.,
        color palettes for Color properties) before falling back to this base behavior.

        The supported transformation strings are: "log", "symlog", "logit", "pow", "sqrt".
        These can be used as prefixes (e.g., "log10", "pow2") for parameterized transforms.
        """
        <your code>

    def standardize(self, val: Any) -> bool:
        """
        Coerce flexible property value to standardized representation.

        This method converts a property value from its flexible input format to a 
        standardized internal representation that can be consistently used throughout
        the visualization system. The base implementation performs no transformation
        and returns the value as-is, but subclasses override this method to implement
        property-specific standardization logic.

        Parameters
        ----------
        val : Any
            The input value to be standardized. Can be any type depending on the
            specific property implementation. For example, color values might accept
            strings, RGB tuples, or hex codes, while marker values might accept
            strings, numbers, or matplotlib MarkerStyle objects.

        Returns
        -------
        bool
            The standardized representation of the input value. Note that while this
            base method signature indicates a boolean return type, subclass 
            implementations may return different types appropriate for their specific
            property (e.g., RGBTuple for colors, MarkerStyle for markers).

        Notes
        -----
        - This is a base implementation that performs no transformation
        - Subclasses should override this method to implement property-specific
          standardization logic
        - The standardized format ensures consistent internal representation
          regardless of the input format used by the user
        - Invalid input values may raise appropriate exceptions depending on the
          property type

        Examples
        --------
        For a Color property, this might convert "red" to (1.0, 0.0, 0.0).
        For a Marker property, this might convert "o" to MarkerStyle("o").
        For a Fill property, this might convert 1 to True.
        """
        <your code>

    def _default_values(self, n: int) -> list:
        """
        Return a list of n values, alternating True and False.
        """
        <your code>

    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        """
        Return a function that maps each data value to True or False.
        """
        <your code>

    def _get_values(self, scale: Scale, levels: list) -> list:
        """
        Validate scale.values and identify a value for each level.
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 3
Below is **Interface Description 3** for file: seaborn-utils.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
def _version_predates(lib: ModuleType, version: str) -> bool:
    """
    Helper function for checking version compatibility.
    """
    <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 4
Below is **Interface Description 4** for file: seaborn-palettes.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
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
    <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.