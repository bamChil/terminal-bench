## Task
**Task: Python Type Annotation Processing and Documentation Utilities**

**Core Functionalities:**
- Convert Python type annotations and class objects into human-readable string representations
- Provide flexible formatting modes for different documentation needs
- Enable safe module mocking for testing and documentation generation

**Main Features & Requirements:**
- Support multiple stringification modes (fully-qualified, smart, typing-aware)
- Handle complex type annotations with configurable verbosity levels
- Provide context-managed mock module injection
- Generate reStructuredText-compatible references for documentation
- Maintain compatibility with modern Python typing features

**Key Challenges:**
- Balancing readability vs. precision in type representation
- Handling edge cases in Python's dynamic type system
- Managing module dependencies during mocking
- Ensuring consistent formatting across different annotation styles
- Supporting both legacy and modern typing syntax

**NOTE**: 
- This test comes from the `sphinx` library, and we have given you the content of this code repository under `/testbed/`, and you need to complete based on this code repository and supplement the files we specify. Remember, all your changes must be in this codebase, and changes that are not in this codebase will not be discovered and tested by us.
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
from sphinx.ext.autodoc.mock import mock
```
This means that we will test one function/class: mock.
And the defination and implementation of class mock should be in `/testbed/sphinx/ext/autodoc/mock.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
def mock(modnames: list[str]) -> Iterator[None]:
    """
    Insert mock modules during context::
    
        with mock(['target.module.name']):
            # mock modules are enabled here
            ...
        
    """
    <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from sphinx.ext.autodoc.mock import mock
from sphinx.util.typing import restify
from sphinx.util.typing import stringify_annotation
```

### Interface Description 1
Below is **Interface Description 1** for file: sphinx-ext-autodoc-mock.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
@contextlib.contextmanager
def mock(modnames: list[str]) -> Iterator[None]:
    """
    Insert mock modules during context::
    
        with mock(['target.module.name']):
            # mock modules are enabled here
            ...
        
    """
    <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 2
Below is **Interface Description 2** for file: sphinx-util-typing.py

This file contains 2 top-level interface(s) that need to be implemented.

```python
def restify(
    cls: Any,
    mode: _RestifyMode = 'fully-qualified-except-typing'
) -> str:
    """
    Convert a type-like object to a reST reference.
    
        :param mode: Specify a method how annotations will be stringified.
    
                     'fully-qualified-except-typing'
                         Show the module name and qualified name of the annotation except
                         the "typing" module.
                     'smart'
                         Show the name of the annotation.
        
    """
    <your code>

def stringify_annotation(
    mode: _StringifyMode = 'fully-qualified-except-typing'
) -> str:
    """
    Stringify type annotation object.
    
        :param annotation: The annotation to stringified.
        :param mode: Specify a method how annotations will be stringified.
    
                     'fully-qualified-except-typing'
                         Show the module name and qualified name of the annotation except
                         the "typing" module.
                     'smart'
                         Show the name of the annotation.
                     'fully-qualified'
                         Show the module name and qualified name of the annotation.
    
        :param short_literals: Render :py:class:`Literals` in PEP 604 style (``|``).
        
    """
    <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.