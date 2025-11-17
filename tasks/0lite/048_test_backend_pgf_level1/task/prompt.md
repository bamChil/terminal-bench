## Task
**Task: Multi-page PDF Generation and Color Mapping System**

**Core Functionalities:**
- Create and manage multi-page PDF documents from matplotlib figures
- Provide color mapping capabilities for data visualization (univariate and bivariate)
- Handle container-based layout and positioning of visual elements

**Main Features & Requirements:**
- Context manager support for automatic resource management and file finalization
- Lazy initialization and deferred file creation until content is added
- Support for PDF metadata, page counting, and annotations
- Flexible colormap operations including resampling, reversing, and RGBA conversion
- Proper cleanup and compilation processes (including LaTeX backend support)

**Key Challenges:**
- Ensure proper file handling and resource cleanup in various exit scenarios
- Handle different PDF backend implementations with consistent interfaces
- Manage color space transformations and clipping for complex data mappings
- Coordinate figure rendering, page layout, and final document assembly
- Balance memory efficiency with performance through lazy loading patterns

**NOTE**: 
- This test comes from the `matplotlib` library, and we have given you the content of this code repository under `/testbed/`, and you need to complete based on this code repository and supplement the files we specify. Remember, all your changes must be in this codebase, and changes that are not in this codebase will not be discovered and tested by us.
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
from matplotlib.backends.backend_pgf import PdfPages
```
This means that we will test one function/class: PdfPages.
And the defination and implementation of class PdfPages should be in `/testbed/lib/matplotlib/backends/backend_pgf.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
class PdfPages:
    """
    
        A multi-page PDF file.
    
        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> # Initialize:
        >>> with PdfPages('foo.pdf') as pdf:
        ...     # As many times as you like, create a figure fig and save it:
        ...     fig = plt.figure()
        ...     pdf.savefig(fig)
        ...     # When no figure is specified the current figure is saved
        ...     pdf.savefig()
    
        Notes
        -----
        In reality `PdfPages` is a thin wrapper around `PdfFile`, in order to avoid
        confusion when using `~.pyplot.savefig` and forgetting the format argument.
        
    """

    @_api.delete_parameter('3.10', 'keep_empty', addendum='This parameter does nothing.')
    def __init__(
        self,
        filename,
        keep_empty = None,
        metadata = None
    ):
        """

                Create a new PdfPages object.

                Parameters
                ----------
                filename : str or path-like or file-like
                    Plots using `PdfPages.savefig` will be written to a file at this location.
                    The file is opened when a figure is saved for the first time (overwriting
                    any older file with the same name).

                metadata : dict, optional
                    Information dictionary object (see PDF reference section 10.2.1
                    'Document Information Dictionary'), e.g.:
                    ``{'Creator': 'My software', 'Author': 'Me', 'Title': 'Awesome'}``.

                    The standard keys are 'Title', 'Author', 'Subject', 'Keywords',
                    'Creator', 'Producer', 'CreationDate', 'ModDate', and
                    'Trapped'. Values have been predefined for 'Creator', 'Producer'
                    and 'CreationDate'. They can be removed by setting them to `None`.

        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from matplotlib.backends.backend_pgf import PdfPages
```

### Interface Description 1
Below is **Interface Description 1** for file: lib-matplotlib-backends-backend_pgf.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
class PdfPages:
    """
    
        A multi-page PDF file using the pgf backend
    
        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> # Initialize:
        >>> with PdfPages('foo.pdf') as pdf:
        ...     # As many times as you like, create a figure fig and save it:
        ...     fig = plt.figure()
        ...     pdf.savefig(fig)
        ...     # When no figure is specified the current figure is saved
        ...     pdf.savefig()
        
    """

    def __init__(self, filename):
        """

                Create a new PdfPages object.

                Parameters
                ----------
                filename : str or path-like
                    Plots using `PdfPages.savefig` will be written to a file at this
                    location. Any older file with the same name is overwritten.

                metadata : dict, optional
                    Information dictionary object (see PDF reference section 10.2.1
                    'Document Information Dictionary'), e.g.:
                    ``{'Creator': 'My software', 'Author': 'Me', 'Title': 'Awesome'}``.

                    The standard keys are 'Title', 'Author', 'Subject', 'Keywords',
                    'Creator', 'Producer', 'CreationDate', 'ModDate', and
                    'Trapped'. Values have been predefined for 'Creator', 'Producer'
                    and 'CreationDate'. They can be removed by setting them to `None`.

                    Note that some versions of LaTeX engines may ignore the 'Producer'
                    key and set it to themselves.

        """
        <your code>

    def _write_header(self, width_inches, height_inches):
        """
        Write the LaTeX document header for a PGF-based PDF page.

        This method generates and writes the initial LaTeX document structure required
        for creating a PDF page using the PGF backend. It sets up the document class,
        page geometry, required packages, and begins the document environment.

        Parameters
        ----------
        width_inches : float
            The width of the page in inches. This value is used to configure the
            page geometry and ensure proper sizing of the PDF output.
        height_inches : float
            The height of the page in inches. This value is used to configure the
            page geometry and ensure proper sizing of the PDF output.

        Important Notes
        ---------------
        - This method writes directly to the internal file buffer (self._file)
        - The header includes PDF metadata information from self._info_dict
        - Sets up hyperref package with PDF metadata for document properties
        - Configures geometry package with zero margins and custom paper size
        - Includes the PGF package and custom preamble for LaTeX rendering
        - Sets paragraph indentation to zero and begins the document environment
        - Should only be called once per PDF document, typically for the first page
        - The output is encoded as UTF-8 bytes before writing to the file buffer
        """
        <your code>

    def __enter__(self):
        """
        Context manager entry method for the PdfPages class.

        This method is called when entering a `with` statement context and enables the PdfPages object to be used as a context manager for creating multi-page PDF files.

        Returns
        -------
        PdfPages
            Returns self to allow the PdfPages instance to be used within the context manager block.

        Notes
        -----
        This method is part of the context manager protocol. When used with a `with` statement, it allows for automatic resource management where the PDF file will be properly finalized and closed when exiting the context, even if an exception occurs.

        The typical usage pattern is:
            with PdfPages('filename.pdf') as pdf:
                # Create and save figures to the PDF
                pdf.savefig(figure)

        This ensures that `close()` will be called automatically when exiting the `with` block, which finalizes the PDF by running LaTeX compilation and moving the final PDF file to the specified location.
        """
        <your code>

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb
    ):
        """
        Exit the runtime context related to this object.

        This method is called when exiting a context manager (when used with the `with` statement). It handles cleanup operations for the PdfPages object, specifically finalizing the PDF document and running LaTeX compilation.

        Parameters
        ----------
        exc_type : type or None
            The exception type that caused the context to be exited, or None if no exception occurred.
        exc_val : Exception or None
            The exception instance that caused the context to be exited, or None if no exception occurred.
        exc_tb : traceback or None
            The traceback object associated with the exception, or None if no exception occurred.

        Returns
        -------
        None
            This method does not return a value. Returning None (or not returning anything) indicates that any exception should be propagated normally.

        Notes
        -----
        This method is automatically called when exiting a `with` block and delegates the actual cleanup work to the `close()` method. The exception parameters are provided by Python's context manager protocol but are not used in this implementation - exceptions are allowed to propagate normally.

        The method enables usage patterns like:
            with PdfPages('output.pdf') as pdf:
                # Create and save figures
                pdf.savefig(fig)
            # PDF is automatically finalized when exiting the with block
        """
        <your code>

    def close(self):
        """

                Finalize this object, running LaTeX in a temporary directory
                and moving the final pdf file to *filename*.

        """
        <your code>

    def _run_latex(self):
        """
        Execute LaTeX compilation process to generate PDF output from accumulated LaTeX content.

        This private method handles the compilation of LaTeX source code that has been written to the internal file buffer. It creates a temporary directory, writes the LaTeX source to a file, runs the configured LaTeX engine to compile it, and moves the resulting PDF to the final output location.

        Parameters
        ----------
        self : PdfPages
            The PdfPages instance containing the LaTeX content to be compiled.

        Returns
        -------
        None
            This method does not return a value but produces a PDF file as a side effect.

        Raises
        ------
        subprocess.CalledProcessError
            If the LaTeX compilation process fails or returns a non-zero exit code.
        FileNotFoundError
            If the specified LaTeX engine (from rcParams['pgf.texsystem']) is not found.
        OSError
            If there are issues with file operations during the compilation process.

        Notes
        -----
        - This method should only be called when there is at least one figure to compile (self._n_figures > 0)
        - The method uses the LaTeX engine specified in matplotlib's rcParams['pgf.texsystem']
        - LaTeX is run with '-interaction=nonstopmode' and '-halt-on-error' flags for automated processing
        - All intermediate files are created in a temporary directory that is automatically cleaned up
        - The final PDF file is moved to the location specified during PdfPages initialization
        """
        <your code>

    def savefig(self, figure = None, **kwargs):
        """

                Save a `.Figure` to this file as a new page.

                Any other keyword arguments are passed to `~.Figure.savefig`.

                Parameters
                ----------
                figure : `.Figure` or int, default: the active figure
                    The figure, or index of the figure, that is saved to the file.

        """
        <your code>

    def get_pagecount(self):
        """
        Return the current number of pages in the multipage pdf file.
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.
