## Task
**Task: Test Path Management and Resolution System**

**Core Functionality:**
Build a system that manages and resolves file paths for test execution, handling both absolute and partial path inputs.

**Main Features & Requirements:**
- Locate and return the root project directory
- Process various path input formats (empty, full paths, partial paths)
- Convert partial paths to absolute paths for reliable execution
- Validate and prepare path arguments for test framework integration

**Key Challenges:**
- Handle multiple path input scenarios robustly
- Ensure path resolution works across different directory structures
- Maintain compatibility with testing framework requirements
- Avoid errors from invalid or nonexistent paths

**NOTE**: 
- This test comes from the `sympy` library, and we have given you the content of this code repository under `/testbed/`, and you need to complete based on this code repository and supplement the files we specify. Remember, all your changes must be in this codebase, and changes that are not in this codebase will not be discovered and tested by us.
- What's more, you need to install `pytest, pytest-timeout, pytest-json-report` in your environment, otherwise our tests won't run and you'll get **ZERO POINTS**!

Your available resources are listed below:
- `/testbed/ace_bench/task/black_links.txt`: Prohibited URLs (all other web resources are allowed)


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
from sympy.testing.runtests_pytest import make_absolute_path
```
This means that we will test one function/class: make_absolute_path.
And the defination and implementation of class make_absolute_path should be in `/testbed/sympy/testing/runtests_pytest.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
def make_absolute_path(partial_path: str) -> str:
    """
    Convert a partial path to an absolute path.
    
        A path such a `sympy/core` might be needed. However, absolute paths should
        be used in the arguments to pytest in all cases as it avoids errors that
        arise from nonexistent paths.
    
        This function assumes that partial_paths will be passed in such that they
        begin with the explicit `sympy` directory, i.e. `sympy/...`.
    
        
    """
    <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from sympy.testing.runtests_pytest import make_absolute_path
from sympy.testing.runtests_pytest import sympy_dir
from sympy.testing.runtests_pytest import update_args_with_paths
```

### Interface Description 1
Below is **Interface Description 1** for file: sympy-testing-runtests_pytest.py

This file contains 3 top-level interface(s) that need to be implemented.

```python
@functools.lru_cache
def sympy_dir() -> pathlib.Path:
    """
    Returns the root SymPy directory.
    """
    <your code>

def update_args_with_paths(
    paths: List[str],
    keywords: Optional[Tuple[str]],
    args: List[str]
) -> List[str]:
    """
    Appends valid paths and flags to the args `list` passed to `pytest.main`.
    
        The are three different types of "path" that a user may pass to the `paths`
        positional arguments, all of which need to be handled slightly differently:
    
        1. Nothing is passed
            The paths to the `testpaths` defined in `pytest.ini` need to be appended
            to the arguments list.
        2. Full, valid paths are passed
            These paths need to be validated but can then be directly appended to
            the arguments list.
        3. Partial paths are passed.
            The `testpaths` defined in `pytest.ini` need to be recursed and any
            matches be appended to the arguments list.
    
        
    """
    <your code>

def make_absolute_path(partial_path: str) -> str:
    """
    Convert a partial path to an absolute path.
    
        A path such a `sympy/core` might be needed. However, absolute paths should
        be used in the arguments to pytest in all cases as it avoids errors that
        arise from nonexistent paths.
    
        This function assumes that partial_paths will be passed in such that they
        begin with the explicit `sympy` directory, i.e. `sympy/...`.
    
        
    """
    <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.