## Task
## Task: Data Visualization Position Adjustment System

**Core Functionality:**
Implement a system for adjusting the positions of data visualization marks (points, bars, areas) to handle overlapping elements and improve visual clarity.

**Main Features:**
- **Categorical ordering** with customizable sequence rules
- **Position transformations** including:
  - Random jittering to reduce overplotting
  - Dodging to separate overlapping marks horizontally/vertically
  - Stacking to create cumulative displays
  - Uniform shifting for consistent displacement
  - Normalization for proportional scaling

**Key Requirements:**
- Process grouped data with flexible orientation (x/y axis)
- Maintain data integrity while transforming coordinates
- Support configurable parameters (gap sizes, magnitudes, aggregation functions)
- Handle both categorical and continuous data appropriately

**Main Challenges:**
- Coordinate system awareness and proper scaling
- Efficient grouping and transformation of large datasets
- Maintaining visual consistency across different mark types
- Balancing automatic positioning with user customization options

**NOTE**: 
- This test is derived from the `seaborn` library, but you are NOT allowed to view this codebase or call any of its interfaces. It is **VERY IMPORTANT** to note that if we detect any viewing or calling of this codebase, you will receive a ZERO for this review.
- What's more, you need to install `pytest, pytest-timeout, pytest-json-report` in your environment, otherwise our tests won't run and you'll get **ZERO POINTS**!
- **CRITICAL**: This task is derived from `seaborn`, but you **MUST** implement the task description independently. It is **ABSOLUTELY FORBIDDEN** to use `pip install seaborn` or some similar commands to access the original implementation—doing so will be considered cheating and will result in an immediate score of ZERO! You must keep this firmly in mind throughout your implementation.
- You are now in `/testbed/`, and originally there was a specific implementation of `seaborn` under `/testbed/` that had been installed via `pip install -e .`. However, to prevent you from cheating, we've removed the code under `/testbed/`. While you can see traces of the installation via the pip show, it's an artifact, and `seaborn` doesn't exist. So you can't and don't need to use `pip install seaborn`, just focus on writing your `agent_code` and accomplishing our task.
- Also, don't try to `pip uninstall seaborn` even if the actual `seaborn` has already been deleted by us, as this will affect our evaluation of you, and uninstalling the residual `seaborn` will result in you getting a ZERO because our tests won't run.

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

Your final deliverable should be code in the `/testbed/agent_code` directory, and after completing the codebase, we will use testfiles in workspace/test to evaluatate your codebase, and note that you won't see workspace/test.
The final structure is like below, note that all dirs and files under agent_code/ are just examples, your codebase's structure should match import structure in **Test Description**, which I will tell you later.
```
/workspace
├── task/                 
│   ├── prompt.md          # task statement
│   ├── black_links.txt    # black links you can't access
│   ├── ...
├── test/                 # you won't see this dir
│   ├── ...
/testbed
├── agent_code/           # all your code should be put into this dir and match the specific dir structure
│   ├── __init__.py       # agent_code/ folder and ALL folders under it should contain __init__.py
│   ├── dir1/
│   │   ├── __init__.py
│   │   ├── code1.py
│   │   ├── ...
├── setup.py              # after finishing your work, you MUST generate this file
```
After you have done all your work, you need to complete three CRITICAL things: 
1. When you have generated all files or folders under `agent_code/` that match the directory structure, you need to recursively generate `__init__.py` under `agent_code/` and in all subfolders under it to ensure that we can access all functions you generate.(you can simply generate empty `__init__.py`)
2. You need to generate `/testbed/setup.py` under `/testbed/` and place the following content exactly:
```python
from setuptools import setup, find_packages
setup(
    name="agent_code",
    version="1.0.0",
    packages=find_packages(include=["agent_code", "agent_code.*"]),
    install_requires=[],
)
```
3. After you have done above two things, you need to use `cd /testbed && pip install .` command to install your code.
Remember, these things are **VERY IMPORTANT**, as they will directly affect whether you can pass our tests.

## Test and Interface Descriptions

The **Test Description** will tell you the position of the function or class which we're testing should satisfy.
This means that when you generate some files and complete the functionality we want to test in the files, you need to put these files in the specified directory, otherwise our tests won't be able to import your generated.
For example, if the **Test Description** show you this:
```python
from agent_code.seaborn._core.moves import Dodge
```
This means that we will test one function/class: Dodge.
And the defination and implementation of class Dodge should be in `/testbed/agent_code/seaborn/_core/moves.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
class Dodge(Move):
    """
    
        Displacement and narrowing of overlapping marks along orientation axis.
    
        Parameters
        ----------
        empty : {'keep', 'drop', 'fill'}
        gap : float
            Size of gap between dodged marks.
        by : list of variable names
            Variables to apply the movement to, otherwise use all.
    
        Examples
        --------
        .. include:: ../docstrings/objects.Dodge.rst
    
        
    """

    empty = "keep"
    gap = 0
    by = None

    def __call__(
        self,
        data: DataFrame,
        groupby: GroupBy,
        orient: str,
        scales: dict[str, Scale]
    ) -> DataFrame:
        """
        Apply positional transformations to data based on grouping and orientation.

        This method is the main entry point for Move subclasses to transform data coordinates.
        It processes the input DataFrame by applying position-based transformations that are
        specific to each Move implementation (e.g., jittering, dodging, stacking, shifting, or
        normalizing).

        Parameters
        ----------
        data : DataFrame
            The input data containing coordinates and other variables to be transformed.
            Expected to have columns like 'x', 'y', and potentially 'width', 'baseline'
            depending on the specific Move subclass.
        groupby : GroupBy
            A GroupBy object that defines how the data should be grouped for applying
            transformations. Contains grouping variables and aggregation logic.
        orient : str
            The orientation axis for the transformation, typically 'x' or 'y'. This
            determines which axis is considered the "orientation axis" vs the "value axis"
            for moves that distinguish between them.
        scales : dict[str, Scale]
            A dictionary mapping variable names to Scale objects that define the
            coordinate system and scaling for each axis. Used by some moves to
            understand the data space.

        Returns
        -------
        DataFrame
            A new DataFrame with transformed coordinates. The structure matches the input
            DataFrame but with modified position values according to the specific Move
            transformation applied.

        Raises
        ------
        NotImplementedError
            This base implementation always raises NotImplementedError. Subclasses must
            override this method to provide specific transformation logic.

        Notes
        -----
        This is an abstract method in the base Move class. Each Move subclass (Jitter,
        Dodge, Stack, Shift, Norm) implements its own version of this method with
        transformation logic appropriate to that move type.

        The method typically returns a copy or modified version of the input data rather
        than modifying it in place, following pandas conventions for data transformation.
        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from agent_code.seaborn._core.moves import Dodge
from agent_code.seaborn._core.moves import Jitter
from agent_code.seaborn._core.moves import Shift
from agent_code.seaborn._core.moves import Stack
from agent_code.seaborn._core.moves import Norm
from agent_code.seaborn._core.rules import categorical_order
```

### Interface Description 1
Below is **Interface Description 1** for file: seaborn-_core-rules.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
def categorical_order(
    vector: Series,
    order: list | None = None
) -> list:
    """
    
        Return a list of unique data values using seaborn's ordering rules.
    
        Parameters
        ----------
        vector : Series
            Vector of "categorical" values
        order : list
            Desired order of category levels to override the order determined
            from the `data` object.
    
        Returns
        -------
        order : list
            Ordered list of category levels not including null values.
    
        
    """
    <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 2
Below is **Interface Description 2** for file: seaborn-_core-moves.py

This file contains 5 top-level interface(s) that need to be implemented.

```python
@dataclass
class Jitter(Move):
    """
    
        Random displacement along one or both axes to reduce overplotting.
    
        Parameters
        ----------
        width : float
            Magnitude of jitter, relative to mark width, along the orientation axis.
            If not provided, the default value will be 0 when `x` or `y` are set, otherwise
            there will be a small amount of jitter applied by default.
        x : float
            Magnitude of jitter, in data units, along the x axis.
        y : float
            Magnitude of jitter, in data units, along the y axis.
    
        Examples
        --------
        .. include:: ../docstrings/objects.Jitter.rst
    
        
    """

    width = "default"
    x = 0
    y = 0
    seed = None

    def __call__(
        self,
        data: DataFrame,
        groupby: GroupBy,
        orient: str,
        scales: dict[str, Scale]
    ) -> DataFrame:
        """
        Apply positional transformations to data based on grouping and orientation.

        This method is the main entry point for Move subclasses to transform data coordinates.
        It takes grouped data and applies position-based transformations such as jittering,
        dodging, stacking, shifting, or normalization depending on the specific Move implementation.

        Parameters
        ----------
        data : DataFrame
            The input data containing coordinates and other variables to be transformed.
            Expected to have columns like 'x', 'y', 'width', 'baseline', etc. depending
            on the specific Move transformation being applied.
        groupby : GroupBy
            A GroupBy object that defines how the data should be grouped for applying
            the transformation. Contains grouping variables and aggregation methods.
        orient : str
            The orientation axis for the transformation, typically 'x' or 'y'.
            Determines which axis is considered the "orientation" axis vs the "value" axis
            for transformations like dodging or stacking.
        scales : dict[str, Scale]
            A dictionary mapping variable names to Scale objects that define the
            coordinate system and scaling for each variable. Used to ensure transformations
            are applied in the correct coordinate space.

        Returns
        -------
        DataFrame
            A new DataFrame with the same structure as the input data but with
            transformed coordinates. The specific columns modified depend on the
            Move subclass implementation.

        Raises
        ------
        NotImplementedError
            This base class method must be overridden by subclasses. Calling this
            method directly on the Move base class will raise NotImplementedError.

        Notes
        -----
        This is an abstract method that must be implemented by all Move subclasses.
        Each subclass applies a different type of positional transformation:
        - Jitter: adds random displacement to reduce overplotting
        - Dodge: displaces overlapping marks along the orientation axis
        - Stack: stacks marks along the value axis
        - Shift: applies uniform displacement to all marks
        - Norm: applies divisive scaling after grouping

        The method typically returns a copy or modified version of the input data
        rather than modifying it in place.
        """
        <your code>

@dataclass
class Dodge(Move):
    """
    
        Displacement and narrowing of overlapping marks along orientation axis.
    
        Parameters
        ----------
        empty : {'keep', 'drop', 'fill'}
        gap : float
            Size of gap between dodged marks.
        by : list of variable names
            Variables to apply the movement to, otherwise use all.
    
        Examples
        --------
        .. include:: ../docstrings/objects.Dodge.rst
    
        
    """

    empty = "keep"
    gap = 0
    by = None

    def __call__(
        self,
        data: DataFrame,
        groupby: GroupBy,
        orient: str,
        scales: dict[str, Scale]
    ) -> DataFrame:
        """
        Apply positional transformations to data based on grouping and orientation.

        This method is the main entry point for Move subclasses to transform data coordinates.
        It processes the input DataFrame by applying position-based transformations that are
        specific to each Move implementation (e.g., jittering, dodging, stacking, shifting, or
        normalizing).

        Parameters
        ----------
        data : DataFrame
            The input data containing coordinates and other variables to be transformed.
            Expected to have columns like 'x', 'y', and potentially 'width', 'baseline'
            depending on the specific Move subclass.
        groupby : GroupBy
            A GroupBy object that defines how the data should be grouped for applying
            transformations. Contains grouping variables and aggregation logic.
        orient : str
            The orientation axis for the transformation, typically 'x' or 'y'. This
            determines which axis is considered the "orientation axis" vs the "value axis"
            for moves that distinguish between them.
        scales : dict[str, Scale]
            A dictionary mapping variable names to Scale objects that define the
            coordinate system and scaling for each axis. Used by some moves to
            understand the data space.

        Returns
        -------
        DataFrame
            A new DataFrame with transformed coordinates. The structure matches the input
            DataFrame but with modified position values according to the specific Move
            transformation applied.

        Raises
        ------
        NotImplementedError
            This base implementation always raises NotImplementedError. Subclasses must
            override this method to provide specific transformation logic.

        Notes
        -----
        This is an abstract method in the base Move class. Each Move subclass (Jitter,
        Dodge, Stack, Shift, Norm) implements its own version of this method with
        transformation logic appropriate to that move type.

        The method typically returns a copy or modified version of the input data rather
        than modifying it in place, following pandas conventions for data transformation.
        """
        <your code>

@dataclass
class Stack(Move):
    """
    
        Displacement of overlapping bar or area marks along the value axis.
    
        Examples
        --------
        .. include:: ../docstrings/objects.Stack.rst
    
        
    """

    def _stack(self, df, orient):
        """
        Stack data along the value axis by computing cumulative sums within groups.

        This method performs the core stacking operation for the Stack move by:
        1. Validating that all marks have homogeneous baselines within the group
        2. Computing cumulative sums of mark lengths (value - baseline) 
        3. Updating mark positions and baselines to create the stacked effect

        Parameters
        ----------
        df : DataFrame
            Input data containing mark information for a single group. Must contain
            columns for the orientation axis, value axis, and 'baseline'.
        orient : str
            The orientation axis ('x' or 'y') that defines the stacking direction.
            The stacking occurs along the opposite axis (the value axis).

        Returns
        -------
        DataFrame
            Modified dataframe with updated positions where:
            - Value axis coordinates are replaced with cumulative sums
            - Baseline coordinates are shifted by cumulative offsets to stack marks

        Raises
        ------
        RuntimeError
            If the input data contains heterogeneous baselines (baseline column has
            more than one unique value), as stacking requires consistent starting
            positions within each group.

        Notes
        -----
        The stacking operation transforms marks from overlapping to vertically/horizontally
        displaced positions. For each mark, the new baseline becomes the sum of all
        previous mark lengths, and the new value position extends from this new baseline.

        This is an internal helper method called by the main Stack.__call__ method
        after data has been properly grouped by position and grouping variables.
        """
        <your code>

    def __call__(
        self,
        data: DataFrame,
        groupby: GroupBy,
        orient: str,
        scales: dict[str, Scale]
    ) -> DataFrame:
        """
        Apply stacking transformation to grouped data based on orientation.

        This method groups the input data by column, row, and orientation axis,
        then applies the stacking transformation to each group using the _stack
        helper method. It handles the grouping logic and delegates the actual
        stacking computation to _stack.

        Parameters
        ----------
        data : DataFrame
            The input data containing coordinates and baseline information.
            Must include columns for 'x', 'y', and 'baseline'.
        groupby : GroupBy
            A GroupBy object (not used in this implementation, kept for interface
            consistency with other Move subclasses).
        orient : str
            The orientation axis ('x' or 'y') that defines the stacking direction.
            Stacking occurs along the perpendicular axis (value axis).
        scales : dict[str, Scale]
            A dictionary mapping variable names to Scale objects (not directly used
            in stacking, kept for interface consistency).

        Returns
        -------
        DataFrame
            Transformed data with stacked coordinates, where marks are displaced
            along the value axis to avoid overlap within each group.

        Notes
        -----
        - Groups are defined by the combination of 'col', 'row', and orientation axis
        - The input groupby parameter is not used, as stacking requires specific
          grouping by position variables
        - See the _stack method for details on the per-group transformation logic
        - TODO: Determine where to ensure proper sorting of semantic variables
          before stacking
        """
        <your code>

@dataclass
class Shift(Move):
    """
    
        Displacement of all marks with the same magnitude / direction.
    
        Parameters
        ----------
        x, y : float
            Magnitude of shift, in data units, along each axis.
    
        Examples
        --------
        .. include:: ../docstrings/objects.Shift.rst
    
        
    """

    x = 0
    y = 0

    def __call__(
        self,
        data: DataFrame,
        groupby: GroupBy,
        orient: str,
        scales: dict[str, Scale]
    ) -> DataFrame:
        """
        Apply a uniform shift transformation to the x and y coordinates of the data.

        This method performs a simple positional shift by adding fixed offset values
        to the 'x' and 'y' columns of the input data. It creates a shallow copy of
        the input data to avoid modifying the original dataframe in place.

        Parameters
        ----------
        data : DataFrame
            The input data containing 'x' and 'y' columns to be transformed.
        groupby : GroupBy
            A GroupBy object (not used in this implementation, retained for interface
            consistency with other Move subclasses).
        orient : str
            The orientation axis (not used in this implementation, retained for 
            interface consistency with other Move subclasses).
        scales : dict[str, Scale]
            A dictionary mapping variable names to Scale objects (not used in this
            implementation, retained for interface consistency).

        Returns
        -------
        DataFrame
            A new DataFrame with the same structure as the input, where:
            - 'x' column values are increased by self.x offset
            - 'y' column values are increased by self.y offset

        Notes
        -----
        - This is a basic shift transformation that applies the same offset to all
          data points regardless of grouping or orientation
        - Uses a shallow copy (deep=False) of the input data for efficiency
        - The method maintains compatibility with the Move subclass interface but
          does not utilize grouping, orientation, or scaling parameters
        - The offset values (self.x and self.y) are typically set during initialization
          of the Shift transform object
        """
        <your code>

@dataclass
class Norm(Move):
    """
    
        Divisive scaling on the value axis after aggregating within groups.
    
        Parameters
        ----------
        func : str or callable
            Function called on each group to define the comparison value.
        where : str
            Query string defining the subset used to define the comparison values.
        by : list of variables
            Variables used to define aggregation groups.
        percent : bool
            If True, multiply the result by 100.
    
        Examples
        --------
        .. include:: ../docstrings/objects.Norm.rst
    
        
    """

    func = "max"
    where = None
    by = None
    percent = False
    group_by_orient = False

    def _norm(self, df, var):
        """
        Normalize values within groups by dividing by an aggregated comparison value.

        This is a helper method for the Norm class that performs the actual normalization
        operation on a DataFrame. It divides all values in the specified variable by an
        aggregated value (computed using the specified function) from either the entire
        group or a filtered subset.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the data to be normalized. Must contain the column
            specified by the 'var' parameter.
        var : str
            The name of the column/variable to normalize. This should be a numeric column
            as it will be divided by the computed denominator value.

        Returns
        -------
        pandas.DataFrame
            A modified copy of the input DataFrame where the specified variable has been
            normalized. The normalization is performed by dividing all values in the
            variable by the aggregated comparison value. If percent=True, the normalized
            values are multiplied by 100.

        Notes
        -----
        - The denominator for normalization is computed by applying self.func to either
          all values in the variable (if self.where is None) or to a filtered subset
          (if self.where is specified as a query string).
        - The normalization preserves the original structure of the DataFrame while only
          modifying the values in the specified variable column.
        - If self.percent is True, the normalized values are converted to percentages
          by multiplying by 100.
        """
        <your code>

    def __call__(
        self,
        data: DataFrame,
        groupby: GroupBy,
        orient: str,
        scales: dict[str, Scale]
    ) -> DataFrame:
        """
        Apply group-wise normalization to data based on orientation.

        This method normalizes values along the value axis (perpendicular to the orientation axis)
        within each group defined by the groupby parameter. It delegates the actual normalization
        computation to the _norm helper method.

        Parameters
        ----------
        data : DataFrame
            The input data containing coordinates to be normalized. Must include columns
            for both axes ('x' and 'y').
        groupby : GroupBy
            A GroupBy object that defines how the data should be grouped for normalization.
            Normalization is performed independently within each group.
        orient : str
            The orientation axis ('x' or 'y'). The value axis to be normalized is determined
            as the opposite axis (e.g., 'y' for orient='x').
        scales : dict[str, Scale]
            A dictionary mapping variable names to Scale objects (not directly used in
            normalization, retained for interface consistency with other Move subclasses).

        Returns
        -------
        DataFrame
            A new DataFrame with the same structure as the input, where values along the
            value axis have been normalized within each group.

        Notes
        -----
        - The value axis to normalize is determined by inverting the orientation axis:
          if orient='x', normalize 'y' values; if orient='y', normalize 'x' values.
        - Grouping is handled by the provided GroupBy object, with normalization applied
          separately to each group via the _norm method.
        - The normalization divides each value by an aggregated value (from self.func)
          from the same group, optionally filtered by self.where.
        - See the _norm method for details on the per-group normalization logic.
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.