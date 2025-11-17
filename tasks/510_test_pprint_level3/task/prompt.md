## Task
**Task: Implement a Configurable Pretty Printer for Python Objects**

**Core Functionalities:**
- Format complex Python data structures (dictionaries, lists, tuples, sets, dataclasses, etc.) into human-readable, indented text representations
- Provide intelligent line wrapping and layout control based on configurable width constraints
- Handle nested objects with proper indentation levels and depth limiting

**Main Features & Requirements:**
- **Configurable formatting**: Support customizable indentation spacing, maximum line width, and maximum nesting depth
- **Type-specific formatting**: Implement specialized formatters for different object types (dicts, lists, strings, bytes, collections, etc.)
- **Intelligent layout**: Break long lines appropriately, wrap text at word boundaries, and maintain readable structure
- **Circular reference detection**: Safely handle self-referencing objects without infinite recursion
- **Stream-based output**: Write formatted results directly to output streams for memory efficiency

**Key Challenges:**
- **Recursion management**: Track object IDs to detect and handle circular references gracefully
- **Width calculation**: Balance line length constraints with readability across nested structures
- **Type dispatch**: Route different object types to appropriate specialized formatting methods
- **Context preservation**: Maintain proper indentation and formatting state across recursive calls
- **Performance optimization**: Handle large or deeply nested data structures efficiently

The task requires building a robust, extensible pretty-printing system that produces consistently formatted, readable output while handling edge cases like circular references and deeply nested structures.

**NOTE**: 
- This test is derived from the `pytest` library, but you are NOT allowed to view this codebase or call any of its interfaces. It is **VERY IMPORTANT** to note that if we detect any viewing or calling of this codebase, you will receive a ZERO for this review.
- What's more, you need to install `pytest, pytest-timeout, pytest-json-report` in your environment, otherwise our tests won't run and you'll get **ZERO POINTS**!
- **CRITICAL**: This task is derived from `pytest`, but you **MUST** implement the task description independently. It is **ABSOLUTELY FORBIDDEN** to use `pip install pytest` or some similar commands to access the original implementation—doing so will be considered cheating and will result in an immediate score of ZERO! You must keep this firmly in mind throughout your implementation.
- You are now in `/testbed/`, and originally there was a specific implementation of `pytest` under `/testbed/` that had been installed via `pip install -e .`. However, to prevent you from cheating, we've removed the code under `/testbed/`. While you can see traces of the installation via the pip show, it's an artifact, and `pytest` doesn't exist. So you can't and don't need to use `pip install pytest`, just focus on writing your `agent_code` and accomplishing our task.
- Also, don't try to `pip uninstall pytest` even if the actual `pytest` has already been deleted by us, as this will affect our evaluation of you, and uninstalling the residual `pytest` will result in you getting a ZERO because our tests won't run.

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
from agent_code._pytest._io.pprint import PrettyPrinter
```
This means that we will test one function/class: PrettyPrinter.
And the defination and implementation of class PrettyPrinter should be in `/testbed/agent_code/_pytest/_io/pprint.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
class PrettyPrinter:
    """
    A configurable pretty printer for Python objects that formats complex data structures
    in a human-readable way with proper indentation and line wrapping.
    
    The PrettyPrinter class provides sophisticated formatting capabilities for nested data
    structures like dictionaries, lists, tuples, sets, dataclasses, and other Python objects.
    It intelligently handles line wrapping, indentation, and recursion detection to produce
    clean, readable output that respects specified width constraints.
    
    Attributes:
        _indent_per_level (int): Number of spaces to indent for each nesting level
        _width (int): Maximum number of columns attempted in output formatting  
        _depth (int | None): Maximum depth to traverse when formatting nested structures
    
    Methods:
        __init__(indent=4, width=80, depth=None): 
            Initialize the pretty printer with formatting parameters.
            
            Args:
                indent: Spaces per indentation level (must be >= 0)
                width: Target maximum line width (must be != 0) 
                depth: Maximum nesting depth to format (must be > 0 if specified)
                
        pformat(object):
            Format an object and return the result as a string.
            
            Args:
                object: Any Python object to format
                
            Returns:
                str: Pretty-formatted string representation
                
        _format(object, stream, indent, allowance, context, level):
            Internal method that handles the core formatting logic. Writes formatted
            output directly to a stream, managing indentation, recursion detection,
            and dispatching to specialized formatters for different object types.
            
        _pprint_dataclass(object, stream, indent, allowance, context, level):
            Specialized formatter for dataclass instances. Formats dataclasses with
            their class name followed by field names and values in a readable layout.
            
        _pprint_dict(object, stream, indent, allowance, context, level):
            Specialized formatter for dictionary objects. Handles key-value pair
            formatting with proper indentation and sorting for consistent output.
    
    Usage Examples:
        # Basic usage with default settings
        pp = PrettyPrinter()
        data = {'users': [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]}
        formatted = pp.pformat(data)
        
        # Custom indentation and width
        pp = PrettyPrinter(indent=2, width=60)
        nested_list = [[1, 2, [3, 4]], [5, 6, [7, 8, [9, 10]]]]
        print(pp.pformat(nested_list))
        
        # Limit formatting depth
        pp = PrettyPrinter(depth=2)
        deep_structure = {'a': {'b': {'c': {'d': 'too deep'}}}}
        formatted = pp.pformat(deep_structure)  # Will truncate at depth 2
    """

    _dispatch = {}

    def __init__(
        self,
        indent: int = 4,
        width: int = 80,
        depth: int | None = None
    ) -> None:
        """
        Handle pretty printing operations onto a stream using a set of
                configured parameters.

                indent
                    Number of spaces to indent for each level of nesting.

                width
                    Attempted maximum number of columns in the output.

                depth
                    The maximum depth to print out nested structures.


        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from agent_code._pytest._io.pprint import PrettyPrinter
```

### Interface Description 1
Below is **Interface Description 1** for file: src-_pytest-_io-pprint.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
class PrettyPrinter:
    """
    A configurable pretty printer for Python objects that formats complex data structures
    in a human-readable way with proper indentation and line wrapping.
    
    The PrettyPrinter class provides sophisticated formatting capabilities for Python objects,
    including built-in types (lists, dicts, tuples, sets, strings, bytes), collections types
    (OrderedDict, defaultdict, Counter, deque, ChainMap), dataclasses, and SimpleNamespace
    objects. It handles circular references gracefully and respects maximum depth limits.
    
    Attributes:
        _indent_per_level (int): Number of spaces to indent for each nesting level
        _width (int): Target maximum number of columns in output lines
        _depth (int | None): Maximum depth to traverse nested structures (None for unlimited)
    
    Parameters:
        indent (int, optional): Number of spaces per indentation level. Defaults to 4.
            Must be >= 0.
        width (int, optional): Attempted maximum line width in characters. Defaults to 80.
            Must be != 0.
        depth (int | None, optional): Maximum nesting depth to print. Defaults to None
            (unlimited depth). If specified, must be > 0.
    
    Main Methods:
            pformat(object): Returns a formatted string representation of any Python object.Serves as the primary public interface, handling all types of input throughinternal dispatching to appropriate formatters. The output respects configuredindentation, width, and depth constraints.
            _format(object, stream, indent, allowance, context, level): Core internal formatting engine.Manages type-specific dispatching, circular reference detection (via context tracking),and stream-based output writing. Coordinates indentation levels, line wrapping,and recursion handling for all object types.
            _pprint_dataclass(object, stream, indent, allowance, context, level): Dataclass-specific formatter.Renders dataclass instances with class name followed by formatted field names and values,only including fields marked for representation (repr=True). Maintains consistentindentation for nested dataclass structures.
            _pprint_dict(object, stream, indent, allowance, context, level): Dictionary formatter with sorted output.Displays dictionaries with keys sorted for consistent rendering, formatting key-value pairswith proper indentation. Handles nested dictionaries through recursive formatting andincludes trailing commas for clean multi-line output.
            _pprint_list(object, stream, indent, allowance, context, level): List and sequence formatter.Renders list elements with appropriate line breaks and indentation, handling nestedsequences through recursive processing. Maintains square bracket syntax with trailingcommas for multi-line lists.
            _pprint_tuple(object, stream, indent, allowance, context, level): Tuple formatter with syntax preservation.Formats tuples with parentheses, including trailing commas for single-element tuples.Handles nested tuples and maintains consistent indentation for complex structures.
            _format_items(items, stream, indent, allowance, context, level): Collection element formatter.Shared helper for formatting elements in sequences (lists, tuples, sets), handlingindentation, line breaks, and recursive formatting of individual items. Ensuresconsistent rendering across all sequence types.
            _safe_repr(object, context, maxlevels, level): Safe string representation generator.Creates fallback string representations for objects without specialized formatters,handling depth limits and circular references to prevent infinite recursion.
    
    The class uses a dispatch table (_dispatch) to route different object types to their
    appropriate formatting methods, enabling extensible and type-specific pretty printing.
    
    Usage Examples:
        # Basic usage with default settings
        pp = PrettyPrinter()
        formatted = pp.pformat({'key': [1, 2, {'nested': 'value'}]})
        
        # Custom indentation and width
        pp = PrettyPrinter(indent=2, width=60)
        formatted = pp.pformat(complex_data_structure)
        
        # Limit nesting depth to prevent deep recursion
        pp = PrettyPrinter(depth=3)
        formatted = pp.pformat(deeply_nested_object)
    
    Raises:
        ValueError: If indent < 0, depth <= 0 (when not None), or width == 0
    """

    _dispatch = {}

    def __init__(
        self,
        indent: int = 4,
        width: int = 80,
        depth: int | None = None
    ) -> None:
        """
        Handle pretty printing operations onto a stream using a set of
                configured parameters.

                indent
                    Number of spaces to indent for each level of nesting.

                width
                    Attempted maximum number of columns in the output.

                depth
                    The maximum depth to print out nested structures.


        """
        <your code>

    def pformat(self, object: Any) -> str:
        """
        Format an object into a pretty-printed string representation.

        This method converts any Python object into a formatted string with proper
        indentation and line breaks according to the PrettyPrinter's configuration
        settings (width, depth, and indent).

        Parameters
        ----------
        object : Any
            The Python object to be formatted. Can be any type including built-in
            types (dict, list, tuple, set, str, bytes), custom classes, dataclasses,
            or nested structures.

        Returns
        -------
        str
            A formatted string representation of the object with appropriate
            indentation, line breaks, and structure formatting. The output respects
            the PrettyPrinter's width limit and will break long lines when possible.

        Notes
        -----
        - The formatting behavior varies by object type:
          * Dictionaries: Keys are sorted and displayed with proper indentation
          * Lists/Tuples: Items are formatted with line breaks for readability
          * Strings: Long strings are wrapped across multiple lines
          * Sets: Items are sorted before formatting
          * Dataclasses: Fields are displayed in a structured format
          * Custom objects: Falls back to their __repr__ method

        - Circular references are detected and displayed as recursion indicators
          to prevent infinite loops

        - The output format is influenced by the PrettyPrinter's configuration:
          * width: Maximum line width before wrapping
          * depth: Maximum nesting level to display
          * indent: Number of spaces per indentation level

        - For objects at the maximum depth limit, abbreviated representations
          (like "...") are used instead of full formatting
        """
        <your code>

    def _format(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Format an object for pretty printing by writing its representation to a stream.

        This is the core formatting method that handles the pretty printing of any Python object.
        It determines the appropriate formatting strategy based on the object's type and handles
        circular reference detection.

        Parameters:
            object (Any): The Python object to be formatted and written to the stream.
            stream (IO[str]): The text stream where the formatted representation will be written.
            indent (int): The current indentation level in spaces from the left margin.
            allowance (int): The number of characters to reserve at the end of the current line
                            for closing brackets, parentheses, or other delimiters.
            context (set[int]): A set containing the object IDs of all objects currently being
                               processed, used for detecting and handling circular references.
            level (int): The current nesting depth level, used for depth limiting and formatting
                        decisions.

        Returns:
            None: This method writes directly to the provided stream and returns nothing.

        Important Notes:
            - Circular references are detected using the context set and are represented with
              a special recursion marker instead of causing infinite loops.
            - The method uses a dispatch table (_dispatch) to find specialized formatting
              functions for different object types based on their __repr__ method.
            - Special handling is provided for dataclasses that have auto-generated repr methods.
            - If no specialized formatter is found, the method falls back to using the object's
              standard repr() representation.
            - The object ID is temporarily added to the context set during processing to detect
              circular references, then removed when formatting is complete.
        """
        <your code>

    def _pprint_dataclass(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Pretty print a dataclass object to a stream with proper formatting and indentation.

        This method handles the pretty printing of dataclass instances by extracting their
        field names and values, then formatting them in a readable structure. It writes
        the class name followed by parentheses containing the formatted field assignments.

        Parameters:
            object (Any): The dataclass instance to be pretty printed. Must be a valid
                dataclass object with repr-enabled fields.
            stream (IO[str]): The output stream where the formatted representation will
                be written.
            indent (int): The current indentation level in spaces from the left margin.
            allowance (int): The number of characters reserved at the end of the line
                for closing brackets, parentheses, or other structural elements.
            context (set[int]): A set containing object IDs currently being processed,
                used to detect and handle circular references.
            level (int): The current nesting depth level, used for recursion control
                and depth limiting.

        Returns:
            None: This method writes directly to the provided stream and returns nothing.

        Notes:
            - Only processes dataclass fields that have repr=True in their field definition
            - Writes output in the format: ClassName(field1=value1, field2=value2, ...)
            - Handles nested structures by delegating to _format_namespace_items for proper
              indentation and line breaking
            - The object ID is managed in the context set by the calling _format method
              to prevent infinite recursion on circular references
            - This method is part of the internal pretty printing dispatch system and
              should not be called directly
        """
        <your code>

    def _pprint_dict(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Pretty print a dictionary object to a stream with proper formatting and indentation.

        This method formats dictionary objects by writing an opening brace, sorting the dictionary
        items by key using safe comparison, formatting each key-value pair with proper indentation,
        and writing a closing brace. The dictionary items are sorted to ensure consistent output
        regardless of the original insertion order.

        Parameters:
            object (Any): The dictionary object to be pretty printed. Expected to be a dict-like
                         object that supports .items() method.
            stream (IO[str]): The output stream where the formatted dictionary representation
                             will be written.
            indent (int): The current indentation level in spaces from the left margin.
            allowance (int): The number of characters to reserve at the end of the line for
                            closing brackets/braces of parent containers.
            context (set[int]): A set containing object IDs currently being processed, used
                               to detect and handle circular references.
            level (int): The current nesting depth level, used for recursion control and
                        depth limiting.

        Returns:
            None: This method writes directly to the stream and does not return a value.

        Notes:
            - Dictionary items are sorted using _safe_tuple to handle cases where keys may not
              be directly comparable (e.g., mixed types).
            - The method delegates the actual formatting of key-value pairs to _format_dict_items.
            - This method is part of the dispatch table and is called automatically for dict
              objects during pretty printing.
            - Circular references are handled by the parent _format method before this method
              is called.
        """
        <your code>

    def _pprint_ordered_dict(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Pretty print an OrderedDict object to a stream with proper formatting and indentation.

        This method handles the pretty printing of collections.OrderedDict objects by writing
        the class name followed by the dictionary contents in a formatted manner. For empty
        OrderedDict objects, it uses the standard repr() output. For non-empty objects, it
        writes the class name and delegates to _pprint_dict for formatting the contents.

        Parameters:
            object (Any): The OrderedDict object to be pretty printed. Expected to be an
                         instance of collections.OrderedDict.
            stream (IO[str]): The output stream where the formatted representation will be
                             written. Must support write() method for string output.
            indent (int): The current indentation level in spaces. Used to maintain proper
                         nesting alignment in the output.
            allowance (int): The number of characters to reserve at the end of the line for
                            closing brackets or other structural elements.
            context (set[int]): A set containing object IDs currently being processed, used
                               to detect and handle circular references during formatting.
            level (int): The current nesting depth level, used for recursion control and
                        depth limiting if configured.

        Returns:
            None: This method writes directly to the provided stream and returns nothing.

        Notes:
            - Empty OrderedDict objects are formatted using their standard string representation
            - Non-empty OrderedDict objects are formatted as "OrderedDict({...})" where the
              dictionary contents are pretty printed with proper indentation
            - This method is part of the dispatch system and is automatically called for
              OrderedDict objects during pretty printing
            - The method relies on _pprint_dict to handle the actual dictionary formatting
        """
        <your code>

    def _pprint_list(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Pretty print a list object to the output stream.

        This method handles the formatting of list objects by writing the opening bracket,
        delegating the formatting of list items to the _format_items method, and writing
        the closing bracket. The output follows Python's list syntax with proper indentation
        and line breaks for nested structures.

        Parameters:
            object (Any): The list object to be pretty printed. Expected to be a list type.
            stream (IO[str]): The output stream where the formatted representation will be written.
            indent (int): The current indentation level in spaces from the left margin.
            allowance (int): The number of characters reserved at the end of the line for
                            closing brackets, parentheses, or other delimiters.
            context (set[int]): A set containing object IDs currently being processed, used
                               to detect and handle circular references during formatting.
            level (int): The current nesting depth level, used for recursion control and
                        depth limiting if configured.

        Returns:
            None: This method writes directly to the stream and returns nothing.

        Notes:
            - This method is part of the internal dispatch mechanism and is automatically
              called when formatting list objects through the _format method
            - The actual formatting of individual list items is delegated to _format_items
            - Circular reference detection is handled by the parent _format method before
              this method is called
            - The method writes the list brackets directly and relies on _format_items
              to handle proper spacing, indentation, and comma placement for the contents
        """
        <your code>

    def _pprint_tuple(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Pretty print a tuple object to the output stream with proper formatting and indentation.

        This method handles the pretty printing of tuple objects by writing the opening parenthesis,
        formatting all items within the tuple using the configured indentation and width settings,
        and then writing the closing parenthesis. The method is part of the PrettyPrinter's dispatch
        system for handling different object types.

        Parameters:
            object (Any): The tuple object to be pretty printed. Should be a tuple instance.
            stream (IO[str]): The output stream where the formatted tuple representation will be written.
            indent (int): The current indentation level in spaces from the left margin.
            allowance (int): The number of characters reserved at the end of the line for closing
                            delimiters and punctuation.
            context (set[int]): A set containing the object IDs of objects currently being processed,
                               used to detect and handle circular references.
            level (int): The current nesting depth level, used for recursion control and depth limiting.

        Returns:
            None: This method writes directly to the stream and does not return a value.

        Notes:
            - This method is automatically called through the PrettyPrinter's dispatch mechanism
              when a tuple object is encountered during formatting
            - The actual item formatting is delegated to the _format_items helper method
            - Circular references are handled by the context parameter tracking
            - The method respects the PrettyPrinter's configured width, indent, and depth settings
            - For single-item tuples, the trailing comma will be handled by the _format_items method
        """
        <your code>

    def _pprint_set(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Pretty-print a set or frozenset object to the output stream.

        This method handles the formatted output of set and frozenset objects, providing
        proper indentation and line breaks for improved readability. Empty sets are
        handled specially to maintain their standard representation.

        Parameters:
            object (Any): The set or frozenset object to be pretty-printed.
            stream (IO[str]): The output stream where the formatted representation will be written.
            indent (int): The current indentation level in spaces.
            allowance (int): The number of characters to reserve at the end of the line.
            context (set[int]): A set of object IDs currently being processed, used to detect circular references.
            level (int): The current nesting level for depth control.

        Returns:
            None: This method writes directly to the stream and returns nothing.

        Notes:
            - Empty sets and frozensets are written using their standard repr() representation
            - For regular sets, the output format is {item1, item2, ...}
            - For frozenset objects, the output format is frozenset({item1, item2, ...})
            - Set elements are sorted using _safe_key to handle unorderable types
            - The method respects the printer's width and indentation settings for formatting
            - This method is registered in the _dispatch dictionary for set.__repr__ and frozenset.__repr__
        """
        <your code>

    def _pprint_str(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Pretty print a string object with intelligent line wrapping and formatting.

        This method handles the pretty printing of string objects by breaking them into
        chunks that fit within the configured width constraints. It intelligently splits
        strings at word boundaries when possible and handles multi-line strings by
        processing each line separately.

        Parameters:
            object (Any): The string object to be pretty printed. Expected to be a string
                but typed as Any for consistency with the dispatch system.
            stream (IO[str]): The output stream where the formatted string will be written.
            indent (int): The current indentation level in spaces from the left margin.
            allowance (int): The number of characters to reserve at the end of the line
                for closing delimiters or punctuation.
            context (set[int]): A set of object IDs currently being processed, used to
                detect and handle circular references.
            level (int): The current nesting depth level in the object hierarchy.

        Returns:
            None: This method writes directly to the provided stream and returns nothing.

        Notes:
            - For empty strings, writes the repr() directly without special formatting
            - Splits multi-line strings by processing each line individually using splitlines(True)
            - Attempts to break long lines at word boundaries using regex pattern r"\S*\s*"
            - For top-level strings (level == 1), wraps the output in parentheses and adjusts
              indentation and allowance accordingly
            - If the string fits on a single line after processing, writes it without
              line breaks regardless of the chunking analysis
            - Preserves the exact representation of string content including escape sequences
              by using repr() on each chunk
        """
        <your code>

    def _pprint_bytes(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Pretty print bytes objects with intelligent line wrapping and formatting.

        This method handles the pretty printing of bytes objects by breaking them into
        manageable chunks that fit within the specified width constraints. For small
        bytes objects (4 bytes or less), it uses the standard repr. For larger objects,
        it wraps the content across multiple lines with proper indentation.

        Parameters:
            object (Any): The bytes object to be pretty printed. Expected to be of type bytes.
            stream (IO[str]): The output stream where the formatted representation will be written.
            indent (int): The current indentation level in spaces for the formatted output.
            allowance (int): The number of characters to reserve at the end of the line for
                            closing delimiters or other formatting elements.
            context (set[int]): A set of object IDs currently being processed, used to detect
                               and handle circular references during pretty printing.
            level (int): The current nesting depth level in the object hierarchy.

        Return:
            None: This method writes directly to the provided stream and returns nothing.

        Notes:
            - For bytes objects with 4 or fewer bytes, the standard repr() is used without wrapping
            - For larger bytes objects, the content is wrapped using the _wrap_bytes_repr helper function
            - When level equals 1, parentheses are added around the output and indentation is adjusted
            - The method respects the PrettyPrinter's width settings when determining line breaks
            - Each wrapped line after the first is indented according to the current indent level
            - This method is part of the dispatch table and is called automatically for bytes objects
        """
        <your code>

    def _pprint_bytearray(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Pretty-print a bytearray object to the output stream.

        This method formats a bytearray object by wrapping it in the bytearray() constructor
        and delegating the formatting of its contents to the bytes pretty-printing method.
        The output format will be "bytearray(b'...')" with proper line wrapping and indentation
        for large byte sequences.

        Parameters:
            object (Any): The bytearray object to be pretty-printed. Expected to be of type bytearray.
            stream (IO[str]): The output stream where the formatted representation will be written.
            indent (int): The current indentation level in spaces from the left margin.
            allowance (int): The number of characters reserved at the end of the line for closing
                            delimiters (like parentheses, brackets, etc.).
            context (set[int]): A set containing the object IDs of objects currently being processed
                               to detect and handle circular references.
            level (int): The current nesting depth level for recursion tracking and depth limiting.

        Returns:
            None: This method writes directly to the stream and returns nothing.

        Notes:
            - The method writes "bytearray(" at the beginning and ")" at the end
            - The actual byte content formatting is delegated to _pprint_bytes method
            - Indentation is increased by 10 spaces (length of "bytearray(") for the inner content
            - Allowance is increased by 1 to account for the closing parenthesis
            - The nesting level is incremented by 1 when calling the bytes formatter
            - This method is registered in the _dispatch dictionary for bytearray.__repr__
        """
        <your code>

    def _pprint_mappingproxy(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Pretty-print a MappingProxyType object with proper formatting and indentation.

        This method handles the pretty-printing of types.MappingProxyType objects by wrapping
        the underlying mapping data in a "mappingproxy(...)" representation. It creates a copy
        of the mapping proxy's data to avoid potential side effects and formats it using the
        standard formatting rules.

        Parameters:
            object (Any): The MappingProxyType object to be pretty-printed. Expected to be an
                instance of types.MappingProxyType with a copy() method.
            stream (IO[str]): The output stream where the formatted representation will be written.
                Must support write() operations for string data.
            indent (int): The current indentation level in spaces. Used to maintain proper
                alignment of nested structures.
            allowance (int): The number of characters reserved for closing brackets/parentheses
                at the current level. Used for width calculations.
            context (set[int]): A set containing object IDs currently being processed, used for
                detecting and handling circular references during formatting.
            level (int): The current nesting depth level. Used for recursion control and
                depth-limited formatting.

        Returns:
            None: This method writes directly to the provided stream and does not return a value.

        Notes:
            - The method writes "mappingproxy(" before formatting the underlying data and ")"
              after it to maintain the proper MappingProxyType representation
            - Uses object.copy() to create a snapshot of the mapping data, which prevents
              potential issues with concurrent modifications during formatting
            - The copied data is formatted using the standard _format method, inheriting all
              formatting rules for the underlying mapping type
            - This method is registered in the _dispatch dictionary and called automatically
              when a MappingProxyType object is encountered during pretty-printing
        """
        <your code>

    def _pprint_simplenamespace(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Pretty-print a types.SimpleNamespace object or its subclass to a stream.

        This method formats SimpleNamespace objects by displaying their attributes in a
        namespace-like representation. For the base SimpleNamespace type, it uses
        "namespace" as the class name to match the standard repr behavior. For subclasses,
        it uses the actual class name.

        Parameters:
            object (Any): The SimpleNamespace object or subclass instance to format.
                Must be a types.SimpleNamespace or its subclass.
            stream (IO[str]): The output stream where the formatted representation
                will be written.
            indent (int): The current indentation level in spaces from the left margin.
            allowance (int): The number of characters reserved for closing delimiters
                on the current line.
            context (set[int]): A set of object IDs currently being processed, used
                to detect and handle circular references.
            level (int): The current nesting depth level for the object being formatted.

        Return:
            None: This method writes directly to the stream and returns nothing.

        Notes:
            - For types.SimpleNamespace objects, the output uses "namespace" as the
              class name to match Python's standard representation
            - For subclasses of SimpleNamespace, the actual class name is used
            - Attributes are formatted as key=value pairs separated by commas
            - The method handles circular references by checking the context set
            - Output format: "namespace(attr1=value1, attr2=value2, ...)" or
              "SubclassName(attr1=value1, attr2=value2, ...)"
        """
        <your code>

    def _format_dict_items(
        self,
        items: list[tuple[Any, Any]],
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Format dictionary items for pretty printing with proper indentation and line breaks.

        This method handles the formatting of dictionary key-value pairs within a dictionary's
        pretty-printed representation. It writes each key-value pair on a separate line with
        appropriate indentation, formatting keys and values recursively while maintaining
        proper comma separation and alignment.

        Parameters:
            items (list[tuple[Any, Any]]): A list of key-value pairs from a dictionary,
                typically sorted for consistent output. Each tuple contains (key, value).
            stream (IO[str]): The output stream where the formatted dictionary items
                will be written.
            indent (int): The current base indentation level in spaces from the left margin.
            allowance (int): The number of characters reserved at the end of the line
                for closing brackets, parentheses, or other structural elements.
            context (set[int]): A set of object IDs currently being processed, used to
                detect and handle circular references during recursive formatting.
            level (int): The current nesting depth level, used for recursion control
                and depth limiting.

        Returns:
            None: This method writes directly to the stream and returns nothing.

        Notes:
            - If the items list is empty, the method returns immediately without writing anything
            - Each key-value pair is written on its own line with increased indentation
            - Keys are formatted using _repr() while values are formatted recursively using _format()
            - All items are followed by commas, and a final newline with base indentation is added
            - The method assumes the opening brace '{' has already been written to the stream
            - The closing brace '}' should be written by the caller after this method returns
        """
        <your code>

    def _format_namespace_items(
        self,
        items: list[tuple[Any, Any]],
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Format namespace items (key=value pairs) for pretty printing.

        This method handles the formatting of namespace-style items where each item
        consists of a key-value pair that should be displayed in the format "key=value".
        It's used for objects like SimpleNamespace and dataclass instances where
        attributes are displayed with their names and values.

        Parameters:
            items (list[tuple[Any, Any]]): A list of tuples where each tuple contains
                a key-value pair to be formatted. The first element is typically a
                string representing the attribute/field name, and the second element
                is the corresponding value.
            stream (IO[str]): The output stream where the formatted representation
                will be written.
            indent (int): The current indentation level in spaces.
            allowance (int): The number of characters to reserve at the end of the
                line for closing brackets/parentheses.
            context (set[int]): A set of object IDs currently being processed, used
                to detect and handle circular references.
            level (int): The current nesting level for depth control and recursion
                detection.

        Returns:
            None: This method writes directly to the provided stream and does not
            return a value.

        Notes:
            - If the items list is empty, the method returns immediately without
              writing anything to the stream.
            - Each item is formatted on a new line with proper indentation.
            - Circular references are detected using the context set and displayed
              as "..." to match standard recursive dataclass representation.
            - The method automatically adds commas after each item and proper
              spacing for alignment.
            - The key is written as-is (assumed to be a string), while the value
              is formatted recursively using the _format method.
        """
        <your code>

    def _format_items(
        self,
        items: list[Any],
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Format a list of items for pretty printing with proper indentation and line breaks.

        This method handles the formatting of sequence-like objects (lists, tuples, sets, etc.)
        by writing each item on a separate line with appropriate indentation. Each item is
        formatted recursively and followed by a comma. The method writes directly to the
        provided stream without returning any value.

        Parameters:
            items (list[Any]): The list of items to be formatted. Can contain any type of
                objects that will be recursively formatted.
            stream (IO[str]): The output stream where the formatted representation will be
                written. Typically a StringIO or file-like object.
            indent (int): The current indentation level in spaces from the left margin.
            allowance (int): The number of characters to reserve at the end of the line
                for closing brackets, parentheses, or other delimiters.
            context (set[int]): A set of object IDs currently being processed, used to
                detect and handle circular references to prevent infinite recursion.
            level (int): The current nesting depth level, used for depth limiting and
                recursion detection.

        Returns:
            None: This method writes directly to the stream and does not return a value.

        Notes:
            - If the items list is empty, the method returns immediately without writing anything
            - Each item is indented by indent + self._indent_per_level spaces
            - Items are separated by newlines and each item is followed by a comma
            - The method adds a final newline with the original indent level after all items
            - Circular references are handled through the context parameter to prevent infinite loops
            - This method is used internally by other pretty-printing methods for lists, tuples, sets, and similar collections
        """
        <your code>

    def _repr(
        self,
        object: Any,
        context: set[int],
        level: int
    ) -> str:
        """
        Generate a string representation of an object for pretty printing purposes.

        This is an internal method that creates a safe string representation of an object
        while respecting depth limits and handling circular references. It serves as a
        fallback when no specialized pretty printing method is available for the object's type.

        Parameters:
            object (Any): The object to generate a string representation for. Can be any
                Python object including built-in types, custom classes, or complex nested
                data structures.
            context (set[int]): A set containing the object IDs of objects currently being
                processed in the formatting chain. Used to detect and handle circular
                references by preventing infinite recursion.
            level (int): The current nesting level in the object hierarchy. Used in
                conjunction with the depth limit to determine when to truncate the
                representation of deeply nested structures.

        Returns:
            str: A string representation of the object. For simple scalar types, this
                returns the standard repr() output. For complex objects, it may return
                a truncated or simplified representation based on depth limits and
                circular reference detection.

        Notes:
            - This method is called internally by the _format method when no specialized
              pretty printer is available for an object's type
            - The method delegates to _safe_repr which handles depth limiting and
              circular reference detection
            - A copy of the context set is passed to _safe_repr to avoid modifying
              the original context during recursive calls
            - The method respects the PrettyPrinter's configured depth limit through
              the self._depth attribute
        """
        <your code>

    def _pprint_default_dict(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Pretty print a collections.defaultdict object to the output stream.

        This method formats a defaultdict by displaying its class name, default factory function,
        and dictionary contents in a readable multi-line format. The output follows the pattern:
        defaultdict(<default_factory>, <dict_contents>).

        Parameters:
            object (Any): The defaultdict object to be pretty printed. Expected to be an instance
                         of collections.defaultdict with a default_factory attribute.
            stream (IO[str]): The output stream where the formatted representation will be written.
                             Typically a file-like object that supports write operations.
            indent (int): The current indentation level in spaces. Used to align nested structures
                         properly within the overall formatting context.
            allowance (int): The number of characters reserved for closing brackets/parentheses
                            at the current nesting level. Used for width calculations.
            context (set[int]): A set containing object IDs currently being processed, used to
                               detect and handle circular references during formatting.
            level (int): The current nesting depth level. Used for recursion control and
                        depth-limited formatting.

        Returns:
            None: This method writes directly to the provided stream and returns nothing.

        Notes:
            - The method writes the default factory representation using _repr() method
            - The dictionary contents are formatted using the existing _pprint_dict() method
            - This method is registered in the _dispatch dictionary for collections.defaultdict.__repr__
            - Circular references in the default factory are handled through the context parameter
            - The formatting respects the PrettyPrinter's width and indentation settings
        """
        <your code>

    def _pprint_counter(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Internal method for pretty-printing collections.Counter objects with proper formatting and indentation.

        This method handles the specialized formatting of Counter objects by displaying them in the format Counter({key: count, ...}) with items ordered by their frequency (most common first). The method writes the formatted output directly to the provided stream.

        Parameters:
            object (Any): The Counter object to be pretty-printed. Expected to be an instance of collections.Counter.
            stream (IO[str]): The output stream where the formatted representation will be written.
            indent (int): The current indentation level in spaces for proper alignment of nested structures.
            allowance (int): The number of characters to reserve at the end of the line for closing brackets/parentheses.
            context (set[int]): A set of object IDs currently being processed, used to detect and handle circular references.
            level (int): The current nesting depth level for tracking recursion limits.

        Returns:
            None: This method writes directly to the stream and does not return a value.

        Notes:
            - Empty Counter objects are displayed as Counter() without the dictionary part
            - Non-empty Counter objects show items ordered by frequency using most_common() method
            - This method is part of the internal dispatch mechanism and should not be called directly
            - The method assumes the object is a valid Counter instance and does not perform type checking
            - Circular references are handled by the context parameter tracking object IDs
        """
        <your code>

    def _pprint_chain_map(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Internal method for pretty-printing collections.ChainMap objects with proper formatting and indentation.

        This method handles the pretty-printing of ChainMap objects by writing the class name followed by a formatted representation of the underlying maps. For empty ChainMaps or ChainMaps with a single empty mapping, it falls back to the standard repr() representation. Otherwise, it formats each mapping in the ChainMap's maps attribute as a list of items with proper indentation.

        Parameters:
            object (Any): The ChainMap object to be pretty-printed. Expected to be an instance of collections.ChainMap with a 'maps' attribute containing the underlying mappings.
            stream (IO[str]): The output stream where the formatted representation will be written. Must support write() method for string output.
            indent (int): The current indentation level in spaces. Used to determine how much to indent nested structures.
            allowance (int): The number of characters to reserve at the end of the line for closing brackets, parentheses, etc.
            context (set[int]): A set of object IDs currently being processed, used to detect and handle circular references during pretty-printing.
            level (int): The current nesting depth level. Used in conjunction with the printer's depth limit to control how deep the formatting goes.

        Returns:
            None: This method writes directly to the provided stream and does not return a value.

        Notes:
            - This is an internal method (indicated by the leading underscore) and is part of the PrettyPrinter's dispatch mechanism
            - The method is registered in the _dispatch dictionary to handle collections.ChainMap.__repr__
            - For empty ChainMaps or those with only one empty mapping, the standard repr() is used for brevity
            - The method delegates the actual formatting of the maps list to _format_items() which handles the indentation and comma placement
            - The output format follows the pattern: ClassName(map1, map2, ...) with proper line breaks and indentation for readability
        """
        <your code>

    def _pprint_deque(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Pretty-print a collections.deque object to the specified stream.

        This method formats a deque object with proper indentation and line breaks,
        displaying it in the format: deque([item1, item2, ...], maxlen=N) where
        maxlen is only shown if it's not None.

        Parameters:
            object (Any): The deque object to be pretty-printed. Expected to be an
                instance of collections.deque with standard deque attributes.
            stream (IO[str]): The output stream where the formatted representation
                will be written. Must support write() method for string output.
            indent (int): The current indentation level in spaces. Used to maintain
                proper alignment in nested structures.
            allowance (int): The number of characters to reserve at the end of the
                line for closing brackets/parentheses and other trailing syntax.
            context (set[int]): A set containing object IDs currently being processed,
                used to detect and handle circular references during formatting.
            level (int): The current nesting depth level, used for recursion control
                and depth limiting in the pretty-printing process.

        Returns:
            None: This method writes directly to the provided stream and returns nothing.

        Notes:
            - If the deque has a maxlen attribute that is not None, it will be displayed
              as a parameter in the output format
            - The method delegates to _format_items() for formatting the deque contents
            - Allowance is increased by 1 when formatting items to account for the
              closing "])" characters
            - This method is part of the internal dispatch system and should not be
              called directly by users
        """
        <your code>

    def _pprint_user_dict(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Pretty print a UserDict object by formatting its underlying data attribute.

        This method handles the pretty printing of collections.UserDict objects by delegating
        the formatting to the underlying data dictionary. UserDict is a dictionary-like
        container class that wraps a regular dictionary in its 'data' attribute.

        Parameters:
            object (Any): The UserDict object to be pretty printed. Expected to have a 'data' 
                         attribute containing the actual dictionary data.
            stream (IO[str]): The output stream where the formatted representation will be written.
            indent (int): The current indentation level in spaces from the left margin.
            allowance (int): The number of characters reserved on the current line for closing
                            delimiters (like brackets, braces, or parentheses).
            context (set[int]): A set of object IDs currently being processed, used to detect
                               and handle circular references during formatting.
            level (int): The current nesting depth level in the object hierarchy, used for
                        recursion control and depth limiting.

        Returns:
            None: This method writes directly to the provided stream and returns nothing.

        Notes:
            - The method decrements the level parameter (level - 1) when calling _format,
              which is consistent with other UserXxx type handlers in this class
            - This method assumes the UserDict object has a 'data' attribute containing
              the actual dictionary to be formatted
            - The formatting behavior will follow the same rules as regular dictionary
              pretty printing, including sorting of keys and proper indentation
            - This method is registered in the _dispatch dictionary to handle UserDict.__repr__
        """
        <your code>

    def _pprint_user_list(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Pretty print a UserList object by formatting its underlying data attribute.

        This method handles the pretty printing of collections.UserList objects by delegating
        the formatting to the underlying data attribute. UserList is a wrapper around a regular
        list, so this method extracts and formats the actual list data.

        Parameters:
            object (Any): The UserList object to be pretty printed. Expected to have a 'data' 
                         attribute containing the actual list data.
            stream (IO[str]): The output stream where the formatted representation will be written.
            indent (int): The current indentation level in spaces for the output formatting.
            allowance (int): The number of characters to reserve at the end of the line for 
                            closing brackets, parentheses, or other delimiters.
            context (set[int]): A set of object IDs currently being processed, used to detect 
                               and handle circular references during formatting.
            level (int): The current nesting depth level in the object hierarchy.

        Returns:
            None: This method writes directly to the provided stream and does not return a value.

        Notes:
            - This method decrements the level by 1 when calling _format to maintain proper 
              nesting level tracking for UserList objects
            - The method assumes the object has a 'data' attribute containing the actual list
            - This is part of the dispatch mechanism for pretty printing different object types
            - Circular references are handled through the context parameter tracking
        """
        <your code>

    def _pprint_user_string(
        self,
        object: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: set[int],
        level: int
    ) -> None:
        """
        Internal method for pretty-printing collections.UserString objects.

        This method handles the pretty-printing of UserString instances by delegating
        the formatting to the underlying string data stored in the object's 'data' attribute.
        It follows the same pattern as other UserXxx collection pretty-printers in this class.

        Parameters:
            object (Any): The UserString object to be pretty-printed. Expected to be an
                         instance of collections.UserString with a 'data' attribute containing
                         the actual string data.
            stream (IO[str]): The output stream where the formatted representation will be written.
            indent (int): The current indentation level in spaces from the left margin.
            allowance (int): The number of characters reserved at the end of the line for
                            closing brackets, parentheses, or other structural elements.
            context (set[int]): A set containing object IDs currently being processed, used
                               to detect and handle circular references during pretty-printing.
            level (int): The current nesting depth level in the object hierarchy, used for
                        depth limiting and recursion control.

        Returns:
            None: This method writes directly to the provided stream and does not return a value.

        Notes:
            - This method decrements the level parameter by 1 when calling _format, which is
              consistent with other UserXxx collection handlers in this class
            - The method relies on the UserString object having a 'data' attribute that
              contains the actual string content
            - This is an internal method (indicated by the leading underscore) and is typically
              called through the dispatch mechanism rather than directly
            - The method is registered in the _dispatch dictionary to handle UserString.__repr__
        """
        <your code>

    def _safe_repr(
        self,
        object: Any,
        context: set[int],
        maxlevels: int | None,
        level: int
    ) -> str:
        """
        Generate a safe string representation of an object with recursion detection and depth limiting.

        This method creates a string representation of any Python object while safely handling
        circular references and respecting maximum depth constraints. It provides special
        handling for built-in collection types (dict, list, tuple) to maintain readable
        formatting even when standard pretty-printing is not used.

        Parameters:
            object (Any): The object to generate a string representation for. Can be any
                Python object including nested collections, custom classes, or built-in types.
            context (set[int]): A set containing the object IDs of objects currently being
                processed in the representation chain. Used to detect circular references
                and prevent infinite recursion.
            maxlevels (int | None): The maximum depth to traverse when representing nested
                objects. If None, no depth limit is enforced. If an integer, representation
                will be truncated with "..." when this depth is exceeded.
            level (int): The current nesting level in the object hierarchy. Used in
                conjunction with maxlevels to determine when to truncate representation.

        Returns:
            str: A string representation of the object. For built-in scalars (str, int, 
                float, etc.), returns the standard repr(). For collections, returns a
                formatted representation with proper handling of nested structures.
                Circular references are represented as "<Recursion on {type} with id={id}>".
                Objects exceeding maxlevels are truncated with "..." notation.

        Notes:
            - Built-in scalar types (str, bytes, bytearray, float, complex, bool, None, int)
              are handled directly with repr() for efficiency
            - Dictionary representations are sorted by key for consistent output
            - The context set is modified during execution to track recursion but is
              properly cleaned up before returning
            - This method is primarily used internally by the PrettyPrinter class when
              standard pretty-printing formatters are not available or appropriate
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.