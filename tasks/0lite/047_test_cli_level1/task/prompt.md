## Task
**Task: Flask Application Discovery and Environment Setup**

**Core Functionality:**
Automatically locate, import, and initialize Flask web applications from Python modules, with support for environment configuration loading.

**Main Features & Requirements:**
- Dynamically discover Flask apps using common naming conventions or specific identifiers
- Handle both direct app instances and factory function patterns
- Support flexible module import paths and app naming schemes
- Load environment variables from dotenv files (.env, .flaskenv)
- Provide CLI integration with version information display
- Offer configurable error handling (strict vs. graceful failure modes)

**Key Challenges:**
- Robust module import handling across different Python path configurations
- Distinguishing between various Flask app instantiation patterns
- Managing import errors vs. application discovery failures
- Maintaining backward compatibility while supporting modern Flask factory patterns
- Graceful degradation when optional dependencies are unavailable

**NOTE**: 
- This test comes from the `flask` library, and we have given you the content of this code repository under `/testbed/`, and you need to complete based on this code repository and supplement the files we specify. Remember, all your changes must be in this codebase, and changes that are not in this codebase will not be discovered and tested by us.
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
from flask.cli import find_best_app
```
This means that we will test one function/class: find_best_app.
And the defination and implementation of class find_best_app should be in `/testbed/src/flask/cli.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
def find_best_app(module: ModuleType) -> Flask:
    """
    Given a module instance this tries to find the best possible
        application in the module or raises an exception.
        
    """
    <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from flask.cli import find_best_app
from flask.cli import get_version
from flask.cli import load_dotenv
from flask.cli import locate_app
from flask.cli import prepare_import
```

### Interface Description 1
Below is **Interface Description 1** for file: src-flask-cli.py

This file contains 7 top-level interface(s) that need to be implemented.

```python
def find_best_app(module: ModuleType) -> Flask:
    """
    Given a module instance this tries to find the best possible
        application in the module or raises an exception.
        
    """
    <your code>

def prepare_import(path: str) -> str:
    """
    Given a filename this will try to calculate the python path, add it
        to the search path and return the actual module name that is expected.
        
    """
    <your code>

@t.overload
def locate_app(
    module_name: str,
    app_name: str | None,
    raise_if_not_found: t.Literal[True] = True
) -> Flask:
    """
    Locate and return a Flask application instance from a specified module.
    
    This function attempts to import the given module and locate a Flask application
    instance within it. When `raise_if_not_found` is True (the default), this overload
    guarantees that a Flask application will be returned or an exception will be raised.
    
    Args:
        module_name (str): The name of the module to import and search for a Flask 
            application. This can be a dotted import path like 'myapp.wsgi' or a 
            simple module name like 'app'.
        app_name (str | None): The specific name of the Flask application or factory 
            function within the module. If None, the function will automatically 
            search for common application names ('app', 'application') or factory 
            functions ('create_app', 'make_app'). If provided, it can be either a 
            variable name or a function call expression like 'create_app()' or 
            'create_app(config="development")'.
        raise_if_not_found (Literal[True]): When True, ensures that an exception 
            is raised if the module cannot be imported or no Flask application is 
            found. This parameter uses a literal type to enable proper overload 
            resolution.
    
    Returns:
        Flask: A Flask application instance. This overload guarantees a non-None 
            return value when `raise_if_not_found` is True.
    
    Raises:
        NoAppException: Raised when the module cannot be imported, when no Flask 
            application or factory function is found in the module, when the 
            specified app_name doesn't exist in the module, when a factory function 
            cannot be called with the provided arguments, or when the located object 
            is not a valid Flask application instance.
        ImportError: May be raised if there are import errors within the target 
            module itself (as opposed to the module not being found).
    
    Note:
        This is an overloaded function. When `raise_if_not_found` is True, the 
        function will always return a Flask instance or raise an exception. For 
        cases where you want to handle missing applications gracefully, use the 
        overload with `raise_if_not_found=False` which can return None.
    """
    <your code>

@t.overload
def locate_app(
    module_name: str,
    app_name: str | None,
    raise_if_not_found: t.Literal[False] = ...
) -> Flask | None:
    """
    Locate and return a Flask application instance from a module.
    
    This is an overloaded function that attempts to import a specified module and locate
    a Flask application instance within it. When `raise_if_not_found` is False, the
    function returns None if the application cannot be found instead of raising an exception.
    
    Args:
        module_name (str): The name of the module to import and search for a Flask application.
            This should be a valid Python module name (e.g., 'myapp' or 'package.module').
        app_name (str | None): The specific name of the Flask application or factory function
            to look for within the module. If None, the function will automatically search
            for common application names ('app', 'application') or factory functions
            ('create_app', 'make_app'). Can also be a function call string like 'create_app()'
            or 'create_app(args)' to call factory functions with arguments.
        raise_if_not_found (Literal[False]): When False, the function returns None instead
            of raising a NoAppException if the module cannot be imported or no Flask
            application is found. Defaults to False in this overload.
    
    Returns:
        Flask | None: A Flask application instance if found, or None if not found and
            raise_if_not_found is False.
    
    Raises:
        NoAppException: Never raised in this overload since raise_if_not_found is False.
            However, other exceptions like ImportError from within the imported module
            may still be raised and wrapped in NoAppException.
    
    Note:
        This overload is used when you want to gracefully handle cases where a Flask
        application might not be available, such as when searching through multiple
        potential module locations. The function will attempt to import the module
        and locate a Flask application using the find_best_app() or find_app_by_string()
        helper functions depending on whether app_name is provided.
    """
    <your code>

def locate_app(
    module_name: str,
    app_name: str | None,
    raise_if_not_found: bool = True
) -> Flask | None:
    """
    Locate and return a Flask application instance from a specified module.
    
    This function attempts to import a module and locate a Flask application within it.
    If no specific app name is provided, it will search for the best available Flask
    application using common naming conventions. If an app name is specified, it will
    look for that specific attribute or callable.
    
    Parameters:
        module_name (str): The name of the module to import and search for a Flask app.
            This can be a dotted import path like 'myapp.wsgi' or a simple module name.
        app_name (str | None): The specific name of the Flask application or factory
            function to look for within the module. If None, the function will
            automatically search for common Flask app names and factory functions.
        raise_if_not_found (bool, optional): Whether to raise an exception if the
            module cannot be imported or no Flask app is found. Defaults to True.
            When False, returns None instead of raising an exception for import failures.
    
    Returns:
        Flask | None: The Flask application instance if found. Returns None only when
            raise_if_not_found is False and the module cannot be imported. Otherwise,
            always returns a Flask instance or raises an exception.
    
    Raises:
        NoAppException: Raised when raise_if_not_found is True and:
            - The specified module cannot be imported
            - An ImportError occurs within the imported module during processing
            - No Flask application or factory function can be found in the module
            - The specified app_name cannot be found or called
            - The found object is not a valid Flask application instance
    
    Notes:
        - When app_name is None, the function uses find_best_app() which searches for
          common names like 'app', 'application', 'create_app', and 'make_app'
        - When app_name is provided, the function uses find_app_by_string() which can
          handle both variable names and function calls with arguments
        - Import errors that occur within the target module (not just missing modules)
          are re-raised as NoAppException with additional context
        - The function distinguishes between import failures and errors within the module
          by examining the traceback depth
    """
    <your code>

def get_version(
    ctx: click.Context,
    param: click.Parameter,
    value: t.Any
) -> None:
    """
    Click callback function that displays version information and exits.
    
    This function is designed to be used as a callback for a Click option (typically --version).
    When triggered, it displays the current Python version along with the versions of Flask
    and Werkzeug, then exits the CLI application.
    
    Parameters:
        ctx (click.Context): The Click context object containing command execution state
            and configuration. Used to access color settings and exit the application.
        param (click.Parameter): The Click parameter object that triggered this callback.
            Not used in the function but required by Click's callback signature.
        value (t.Any): The value of the parameter that triggered this callback. If falsy
            or if ctx.resilient_parsing is True, the function returns early without action.
    
    Returns:
        None: This function does not return a value as it either exits early or
            terminates the application via ctx.exit().
    
    Notes:
        - The function uses ctx.resilient_parsing to avoid executing during Click's
          parsing phase when building help text or performing other non-execution tasks
        - Version information is retrieved using importlib.metadata for Flask and Werkzeug
        - Output respects the context's color setting for terminal formatting
        - The function calls ctx.exit() which terminates the CLI application with exit code 0
        - This is typically used with the is_eager=True option setting to ensure version
          checking happens before other command processing
    """
    <your code>

def load_dotenv(
    path: str | os.PathLike[str] | None = None,
    load_defaults: bool = True
) -> bool:
    """
    Load "dotenv" files to set environment variables. A given path takes
        precedence over ``.env``, which takes precedence over ``.flaskenv``. After
        loading and combining these files, values are only set if the key is not
        already set in ``os.environ``.
    
        This is a no-op if `python-dotenv`_ is not installed.
    
        .. _python-dotenv: https://github.com/theskumar/python-dotenv#readme
    
        :param path: Load the file at this location.
        :param load_defaults: Search for and load the default ``.flaskenv`` and
            ``.env`` files.
        :return: ``True`` if at least one env var was loaded.
    
        .. versionchanged:: 3.1
            Added the ``load_defaults`` parameter. A given path takes precedence
            over default files.
    
        .. versionchanged:: 2.0
            The current directory is not changed to the location of the
            loaded file.
    
        .. versionchanged:: 2.0
            When loading the env files, set the default encoding to UTF-8.
    
        .. versionchanged:: 1.1.0
            Returns ``False`` when python-dotenv is not installed, or when
            the given path isn't a file.
    
        .. versionadded:: 1.0
        
    """
    <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.