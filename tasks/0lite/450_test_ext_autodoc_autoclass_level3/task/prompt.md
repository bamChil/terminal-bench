## Task
**Task: Programmatic Documentation Generation**

**Core Functionality:**
Generate structured documentation for Python code objects using Sphinx's autodoc system programmatically, without requiring manual directive writing.

**Main Features & Requirements:**
- Accept Python object specifications (type, name, options) as input parameters
- Integrate with existing Sphinx application environments and configurations
- Support multiple object types (modules, classes, functions, methods, attributes)
- Process customizable documentation options and formatting preferences
- Output reStructuredText-compatible documentation content

**Key Challenges:**
- Handle dynamic object discovery and import resolution
- Manage Sphinx application state and documenter registry integration
- Ensure compatibility with various autodoc configuration options
- Provide robust error handling for missing or invalid objects
- Maintain consistency with standard autodoc directive output formatting

**NOTE**: 
- This test is derived from the `sphinx` library, but you are NOT allowed to view this codebase or call any of its interfaces. It is **VERY IMPORTANT** to note that if we detect any viewing or calling of this codebase, you will receive a ZERO for this review.
- What's more, you need to install `pytest, pytest-timeout, pytest-json-report` in your environment, otherwise our tests won't run and you'll get **ZERO POINTS**!
- **CRITICAL**: This task is derived from `sphinx`, but you **MUST** implement the task description independently. It is **ABSOLUTELY FORBIDDEN** to use `pip install sphinx` or some similar commands to access the original implementation—doing so will be considered cheating and will result in an immediate score of ZERO! You must keep this firmly in mind throughout your implementation.
- You are now in `/testbed/`, and originally there was a specific implementation of `sphinx` under `/testbed/` that had been installed via `pip install -e .`. However, to prevent you from cheating, we've removed the code under `/testbed/`. While you can see traces of the installation via the pip show, it's an artifact, and `sphinx` doesn't exist. So you can't and don't need to use `pip install sphinx`, just focus on writing your `agent_code` and accomplishing our task.
- Also, don't try to `pip uninstall sphinx` even if the actual `sphinx` has already been deleted by us, as this will affect our evaluation of you, and uninstalling the residual `sphinx` will result in you getting a ZERO because our tests won't run.

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
from agent_code.tests.test_extensions.autodoc_util import do_autodoc
```
This means that we will test one function/class: do_autodoc.
And the defination and implementation of class do_autodoc should be in `/testbed/agent_code/tests/test_extensions/autodoc_util.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
def do_autodoc(
    app: Sphinx,
    objtype: str,
    name: str,
    options: dict[str, Any] | None = None
) -> StringList:
    """
    Generate autodoc documentation for a Python object using Sphinx's autodoc extension.
    
    This function programmatically creates documentation for a specified Python object
    using Sphinx's autodoc functionality. It sets up the necessary documenter bridge,
    processes options, and generates the documentation content that would normally be
    produced by autodoc directives in reStructuredText files.
    
    Parameters:
        app (Sphinx): The Sphinx application instance containing the environment,
            configuration, and registry of documenters.
        objtype (str): The type of object to document (e.g., 'module', 'class', 
            'function', 'method', 'attribute'). Must be a valid documenter type
            registered in the Sphinx application.
        name (str): The fully qualified name of the Python object to document
            (e.g., 'mymodule.MyClass' or 'mymodule.my_function').
        options (dict[str, Any] | None, optional): Dictionary of autodoc options
            to control documentation generation behavior. These correspond to the
            options available in autodoc directives (e.g., 'members', 'undoc-members',
            'show-inheritance'). Defaults to None, which results in an empty options dict.
    
    Returns:
        StringList: A docutils StringList containing the generated reStructuredText
            documentation lines for the specified object. This can be further processed
            or inserted into documentation.
    
    Raises:
        KeyError: If the specified objtype is not found in the application's
            documenter registry.
        ImportError: If the specified object name cannot be imported or found.
        
    Notes:
        - If the current document has no docname set, it will be automatically set to 'index'
        - The function creates a mock state object with a default tab width of 8 spaces
        - Options are merged with default autodoc options from the Sphinx configuration
        - The generated content follows the same format as standard autodoc directives
    """
    <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from agent_code.tests.test_extensions.autodoc_util import do_autodoc
```

### Interface Description 1
Below is **Interface Description 1** for file: tests-test_extensions-autodoc_util.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
def do_autodoc(
    app: Sphinx,
    objtype: str,
    name: str,
    options: dict[str, Any] | None = None
) -> StringList:
    """
    Generate autodoc documentation for a Python object using Sphinx's autodoc extension.
    
    This function programmatically creates documentation for a specified Python object
    using Sphinx's autodoc functionality. It sets up the necessary documenter bridge,
    processes options, and generates the documentation content that would normally be
    produced by autodoc directives in reStructuredText files.
    
    Parameters:
        app (Sphinx): The Sphinx application instance containing the environment,
            configuration, and registry of documenters.
        objtype (str): The type of object to document (e.g., 'module', 'class', 
            'function', 'method', 'attribute'). Must be a valid documenter type
            registered in the Sphinx application.
        name (str): The fully qualified name of the Python object to document
            (e.g., 'mymodule.MyClass' or 'mymodule.my_function').
        options (dict[str, Any] | None, optional): Dictionary of autodoc options
            to control documentation generation behavior. These correspond to the
            options available in autodoc directives (e.g., 'members', 'undoc-members',
            'show-inheritance'). Defaults to None, which results in an empty options dict.
    
    Returns:
        StringList: A docutils StringList containing the generated reStructuredText
            documentation lines for the specified object. This can be further processed
            or inserted into documentation.
    
    Raises:
        KeyError: If the specified objtype is not found in the application's
            documenter registry.
        ImportError: If the specified object name cannot be imported or found.
        
    Notes:
        - If the current document has no docname set, it will be automatically set to 'index'
        - The function creates a mock state object with a default tab width of 8 spaces
        - Options are merged with default autodoc options from the Sphinx configuration
        - The generated content follows the same format as standard autodoc directives
    """
    <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.