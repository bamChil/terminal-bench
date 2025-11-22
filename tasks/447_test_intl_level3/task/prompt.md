## Task
**Task: Internationalization Message Catalog Management**

**Core Functionality:**
Build a system to collect, organize, and process translatable messages from documentation sources for internationalization (i18n) support.

**Main Features & Requirements:**
- Collect translatable text strings with source location metadata (file paths, line numbers, unique IDs)
- Store and consolidate duplicate messages while tracking all their locations
- Provide iteration and access methods for message retrieval
- Transform document trees to preserve original messages before translation
- Generate data structures compatible with gettext-style translation workflows

**Key Challenges:**
- Handle missing or incomplete metadata gracefully (missing UIDs, line numbers)
- Consolidate duplicate messages while preserving all location references
- Ensure proper transform execution order in the processing pipeline
- Maintain data integrity during in-place document tree modifications

**NOTE**: 
- This test is derived from the `sphinx` library, but you are NOT allowed to view this codebase or call any of its interfaces. It is **VERY IMPORTANT** to note that if we detect any viewing or calling of this codebase, you will receive a ZERO for this review.
- What's more, you need to install `pytest, pytest-timeout, pytest-json-report` in your environment, otherwise our tests won't run and you'll get **ZERO POINTS**!
- **CRITICAL**: This task is derived from `sphinx`, but you **MUST** implement the task description independently. It is **ABSOLUTELY FORBIDDEN** to use `pip install sphinx` or some similar commands to access the original implementation—doing so will be considered cheating and will result in an immediate score of ZERO! You must keep this firmly in mind throughout your implementation.
- You are now in `/testbed/`, and originally there was a specific implementation of `sphinx` under `/testbed/` that had been installed via `pip install -e .`. However, to prevent you from cheating, we've removed the code under `/testbed/`. While you can see traces of the installation via the `pip show`, it's an artifact, and `sphinx` doesn't exist. So you can't and don't need to use `pip install sphinx`, just focus on writing your `agent_code` and accomplishing our task.
- Also, don't try to `pip uninstall sphinx` even if the actual `sphinx` has already been deleted by us, as this will affect our evaluation of you, and uninstalling the residual `sphinx` will result in you getting a ZERO because our tests won't run.
- You are now in `/testbed/`, and originally there was a specific implementation of `sphinx` under `/testbed/` that had been installed via `pip install -e .`. However, to prevent you from cheating, we've removed the code under `/testbed/`. While you can see traces of the installation via the pip show, it's an artifact, and `sphinx` doesn't exist. So you can't and don't need to use `pip install sphinx`, just focus on writing your `agent_code` and accomplishing our task.

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

Your final deliverable should be code in the `/testbed/agent_code` directory.
The final structure is like below, note that all dirs and files under agent_code/ are just examples, your codebase's structure should match import structure in **Test Description**, which I will tell you later.
```
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
from agent_code.babel.messages.catalog import Catalog
```
This means that we will test one function/class: Catalog.
And the defination and implementation of class Catalog should be in `/testbed/agent_code/babel/messages/catalog.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
class Catalog:
    """
    Catalog of translatable messages.
    """

    __slots__ = ('metadata',)

    def __init__(self) -> None:
        """
        Initialize a new Catalog instance for storing translatable messages.

        This constructor creates an empty catalog with initialized metadata storage
        for managing translatable message entries and their associated location
        information.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This is a constructor method that does not return a value.

        Notes
        -----
        - Initializes the metadata attribute as an empty dictionary that will map
          message strings to lists of tuples containing file path, line number,
          and unique identifier information
        - The metadata structure follows the format: 
          {msgid: [(file_path, line_number, uid), ...]}
        - This catalog is used to collect and organize translatable strings during
          the documentation building process for internationalization (i18n) support
        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from agent_code.babel.messages.catalog import Catalog
```

### Interface Description 1
Below is **Interface Description 1** for file: sphinx-builders-gettext.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
class Catalog:
    """
    Catalog of translatable messages.
    """

    __slots__ = ('metadata',)

    def __init__(self) -> None:
        """
        Initialize a new Catalog instance for storing translatable messages.

        This constructor creates an empty catalog with initialized metadata storage
        for managing translatable message entries and their associated location
        information.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This is a constructor method that does not return a value.

        Notes
        -----
        - Initializes the metadata attribute as an empty dictionary that will map
          message strings to lists of tuples containing file path, line number,
          and unique identifier information
        - The metadata structure follows the format: 
          {msgid: [(file_path, line_number, uid), ...]}
        - This catalog is used to collect and organize translatable strings during
          the documentation building process for internationalization (i18n) support
        """
        <your code>

    def add(
        self,
        msg: str,
        origin: Element | MsgOrigin
    ) -> None:
        """
        Add a translatable message to the catalog.

        This method registers a translatable message string along with its source location
        and metadata for inclusion in gettext-style message catalogs (.pot files). The
        message will be associated with location information derived from the origin
        parameter.

        Parameters:
            msg (str): The translatable message text to be added to the catalog. This
                is typically extracted from documentation content, templates, or other
                translatable elements.
            origin (Element | MsgOrigin): The source origin of the message, which can be
                either a docutils Element node or a MsgOrigin object. This provides
                location metadata including source file path, line number, and unique
                identifier for the message.

        Returns:
            None: This method does not return a value. It modifies the catalog's
                internal metadata dictionary in-place.

        Notes:
            - If the origin object does not have a 'uid' attribute, the message will
              be silently ignored and not added to the catalog. This typically occurs
              with replicated nodes like todo items where translation is unnecessary.
            - The method extracts source file path and line number from the origin.
              If no line number is available, it defaults to -1.
            - Messages with the same text content will have their metadata consolidated
              under a single entry, allowing multiple locations to be tracked for the
              same translatable string.
            - The source file path may be empty if not available from the origin object.
        """
        <your code>

    def __iter__(self) -> Iterator[Message]:
        """
        Return an iterator over all Message objects in the catalog.

        This method implements the iterator protocol for the Catalog class, allowing
        it to be used in for loops and other iteration contexts. It processes the
        internal metadata dictionary to create Message objects with consolidated
        location and UUID information.

        Returns:
            Iterator[Message]: An iterator that yields Message objects. Each Message
                contains:
                - text: The translatable message string
                - locations: A sorted list of unique (source_file, line_number) tuples
                  where the message appears
                - uuids: A list of unique identifiers associated with the message

        Notes:
            - Duplicate locations are automatically removed and the remaining locations
              are sorted for consistent output
            - The locations are extracted from the first two elements of each metadata
              tuple (source, line), while UUIDs are extracted from the third element
            - This method processes all messages stored in the catalog's metadata
              dictionary at the time of iteration
        """
        <your code>

    @property
    def messages(self) -> list[str]:
        """
        Get all translatable message strings from the catalog.

        This property provides access to all message identifiers (msgids) that have been
        collected in the catalog's metadata dictionary. These represent the original
        translatable text strings that were extracted from documentation sources.

        Returns:
            list[str]: A list of all message strings (keys) stored in the catalog's
                metadata dictionary. Each string represents a unique translatable
                message that was found during the documentation parsing process.

        Note:
            This property returns only the message text strings, not the associated
            metadata such as source locations or UUIDs. For complete message information
            including metadata, iterate over the catalog instance directly using the
            __iter__ method which yields Message objects.
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

