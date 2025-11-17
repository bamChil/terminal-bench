## Task
**Task: Python Documentation Domain Parser and Cross-Reference System**

**Core Functionalities:**
Build a Python-specific documentation domain system that parses Python code signatures, manages object registries, and resolves cross-references between Python modules, classes, and functions.

**Main Features & Requirements:**
- Parse Python type annotations and function argument lists with optional parameter support
- Maintain registries of Python objects (modules, classes, functions) with metadata (location, type, deprecation status)
- Generate hierarchical module indices with automatic grouping and collapsing
- Resolve cross-references using fuzzy matching and fallback strategies
- Support parallel build environments with data merging capabilities
- Handle aliased objects and provide search prioritization

**Key Challenges:**
- Accurate parsing of complex Python signatures while handling edge cases (e.g., commas in string literals)
- Efficient object lookup across module/class hierarchies with multiple search modes
- Managing object lifecycle during document rebuilds and parallel processing
- Balancing search relevance between canonical and aliased object references
- Maintaining data consistency across distributed build processes

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
from agent_code.sphinx.domains.python import PythonDomain
```
This means that we will test one function/class: PythonDomain.
And the defination and implementation of class PythonDomain should be in `/testbed/agent_code/sphinx/domains/python.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
class PythonDomain(Domain):
    """
    Python language domain.
    """

    name = "py"
    label = "Python"
    object_types = "{'function': ObjType(_('function'), 'func', 'obj'), 'data': ObjType(_('data'), 'data', 'obj'), 'class': ObjType(_('class'), 'class', 'exc', 'obj'), 'exception': ObjType(_('exception'), 'exc', 'class', 'obj'), 'method': ObjType(_('method'), 'meth', 'obj'), 'classmethod': ObjType(_('class method'), 'meth', 'obj'), 'staticmethod': ObjType(_('static method'), 'meth', 'obj'), 'attribute': ObjType(_('attribute'), 'attr', 'obj'), 'property': ObjType(_('property'), 'attr', '_prop', 'obj'), 'type': ObjType(_('type alias'), 'type', 'obj'), 'module': ObjType(_('module'), 'mod', 'obj')}"
    directives = "{'function': PyFunction, 'data': PyVariable, 'class': PyClasslike, 'exception': PyClasslike, 'method': PyMethod, 'classmethod': PyClassMethod, 'staticmethod': PyStaticMethod, 'attribute': PyAttribute, 'property': PyProperty, 'type': PyTypeAlias, 'module': PyModule, 'currentmodule': PyCurrentModule, 'decorator': PyDecoratorFunction, 'decoratormethod': PyDecoratorMethod}"
    roles = "{'data': PyXRefRole(), 'exc': PyXRefRole(), 'func': PyXRefRole(fix_parens=True), 'deco': _PyDecoXRefRole(), 'class': PyXRefRole(), 'const': PyXRefRole(), 'attr': PyXRefRole(), 'type': PyXRefRole(), 'meth': PyXRefRole(fix_parens=True), 'mod': PyXRefRole(), 'obj': PyXRefRole()}"
    initial_data = {'objects': {}, 'modules': {}}
    indices = "[PythonModuleIndex]"

    @property
    def objects(self) -> dict[str, ObjectEntry]:
        """
        Property that provides access to the Python domain's object registry.

        This property returns a dictionary mapping fully qualified object names to their
        corresponding ObjectEntry instances, which contain metadata about documented
        Python objects in the domain.

        Returns:
            dict[str, ObjectEntry]: A dictionary where keys are fully qualified object
                names (e.g., 'module.Class.method') and values are ObjectEntry named
                tuples containing:
                - docname: The document name where the object is defined
                - node_id: The unique identifier for the object's node
                - objtype: The type of the object (e.g., 'function', 'class', 'method')
                - aliased: Boolean indicating if this is an alias to another object

        Notes:
            - The returned dictionary is retrieved from the domain's data storage and
              is automatically created if it doesn't exist
            - This registry is used internally by Sphinx for cross-reference resolution
              and object lookup operations
            - The dictionary is shared across the entire documentation build process
              and persists object information for generating indices and resolving
              references
        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from agent_code.sphinx.domains.python import PythonDomain
from agent_code.sphinx.domains.python import PythonModuleIndex
from agent_code.sphinx.domains.python._annotations import _parse_annotation
from agent_code.sphinx.domains.python._annotations import _pseudo_parse_arglist
```

### Interface Description 1
Below is **Interface Description 1** for file: sphinx-domains-python-_annotations.py

This file contains 2 top-level interface(s) that need to be implemented.

```python
def _parse_annotation(
    annotation: str,
    env: BuildEnvironment
) -> list[Node]:
    """
    Parse type annotation.
    """
    <your code>

def _pseudo_parse_arglist(
    signode: desc_signature,
    arglist: str,
    *,
    multi_line_parameter_list: bool = False,
    trailing_comma: bool = True,
    env: BuildEnvironment,
) -> None:
    """
    'Parse' a list of arguments separated by commas.
    
        Arguments can have "optional" annotations given by enclosing them in
        brackets.  Currently, this will split at any comma, even if it's inside a
        string literal (e.g. default argument value).
        
    """
    <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 2
Below is **Interface Description 2** for file: sphinx-domains-python-__init__.py

This file contains 2 top-level interface(s) that need to be implemented.

```python
class PythonModuleIndex(Index):
    """
    Index subclass to provide the Python module index.
    """

    name = "modindex"
    localname = "_('Python Module Index')"
    shortname = "_('modules')"
    domain: PythonDomain

    def generate(
        self,
        docnames: Iterable[str] | None = None
    ) -> tuple[list[tuple[str, list[IndexEntry]]], bool]:
        """
        Generate index entries for Python modules.

        This method creates a hierarchical index of Python modules, organizing them alphabetically
        and applying collapsible grouping for submodules. The index includes module metadata
        such as platform information, deprecation status, and synopsis.

        Parameters
        ----------
        docnames : Iterable[str] | None, optional
            An iterable of document names to filter the modules by. If provided, only
            modules whose docname is in this collection will be included in the index.
            If None (default), all modules in the domain will be included.

        Returns
        -------
        tuple[list[tuple[str, list[IndexEntry]]], bool]
            A tuple containing:
            - A list of tuples where each tuple contains:
                - A string representing the first letter (lowercase) of module names
                - A list of IndexEntry objects for modules starting with that letter
            - A boolean indicating whether the index should be collapsed by default
              (True if the number of submodules exceeds the number of top-level modules)

        Notes
        -----
        - Module names are processed to remove common prefixes as specified in the
          'modindex_common_prefix' configuration option
        - Submodules are automatically grouped under their parent modules with
          appropriate indentation levels (subtype 0 for top-level, 1 for parent
          groups, 2 for submodules)
        - Dummy entries are created for submodules that don't have their parent
          module explicitly listed
        - The collapse heuristic helps improve user experience by automatically
          collapsing large module hierarchies
        - IndexEntry objects include module metadata like platform, deprecation
          status, and synopsis information
        - The returned sorted_content is ordered alphabetically by the first letter
          (lowercase) of the module names, ensuring consistent and predictable
          index organization
        """
        <your code>

class PythonDomain(Domain):
    """
    Python language domain.
    """

    name = "py"
    label = "Python"
    object_types = "{'function': ObjType(_('function'), 'func', 'obj'), 'data': ObjType(_('data'), 'data', 'obj'), 'class': ObjType(_('class'), 'class', 'exc', 'obj'), 'exception': ObjType(_('exception'), 'exc', 'class', 'obj'), 'method': ObjType(_('method'), 'meth', 'obj'), 'classmethod': ObjType(_('class method'), 'meth', 'obj'), 'staticmethod': ObjType(_('static method'), 'meth', 'obj'), 'attribute': ObjType(_('attribute'), 'attr', 'obj'), 'property': ObjType(_('property'), 'attr', '_prop', 'obj'), 'type': ObjType(_('type alias'), 'type', 'obj'), 'module': ObjType(_('module'), 'mod', 'obj')}"
    directives = "{'function': PyFunction, 'data': PyVariable, 'class': PyClasslike, 'exception': PyClasslike, 'method': PyMethod, 'classmethod': PyClassMethod, 'staticmethod': PyStaticMethod, 'attribute': PyAttribute, 'property': PyProperty, 'type': PyTypeAlias, 'module': PyModule, 'currentmodule': PyCurrentModule, 'decorator': PyDecoratorFunction, 'decoratormethod': PyDecoratorMethod}"
    roles = "{'data': PyXRefRole(), 'exc': PyXRefRole(), 'func': PyXRefRole(fix_parens=True), 'deco': _PyDecoXRefRole(), 'class': PyXRefRole(), 'const': PyXRefRole(), 'attr': PyXRefRole(), 'type': PyXRefRole(), 'meth': PyXRefRole(fix_parens=True), 'mod': PyXRefRole(), 'obj': PyXRefRole()}"
    initial_data = {'objects': {}, 'modules': {}}
    indices = "[PythonModuleIndex]"

    @property
    def objects(self) -> dict[str, ObjectEntry]:
        """
        Property that provides access to the Python domain's object registry.

        This property returns a dictionary mapping fully qualified object names to their
        corresponding ObjectEntry instances, which contain metadata about documented
        Python objects in the domain.

        Returns:
            dict[str, ObjectEntry]: A dictionary where keys are fully qualified object
                names (e.g., 'module.Class.method') and values are ObjectEntry named
                tuples containing:
                - docname: The document name where the object is defined
                - node_id: The unique identifier for the object's node
                - objtype: The type of the object (e.g., 'function', 'class', 'method')
                - aliased: Boolean indicating if this is an alias to another object

        Notes:
            - The returned dictionary is retrieved from the domain's data storage and
              is automatically created if it doesn't exist
            - This registry is used internally by Sphinx for cross-reference resolution
              and object lookup operations
            - The dictionary is shared across the entire documentation build process
              and persists object information for generating indices and resolving
              references
        """
        <your code>

    def note_object(
        self,
        name: str,
        objtype: str,
        node_id: str,
        aliased: bool = False,
        location: Any = None
    ) -> None:
        """
        Note a python object for cross reference.

                .. versionadded:: 2.1

        """
        <your code>

    @property
    def modules(self) -> dict[str, ModuleEntry]:
        """
        Property that provides access to the Python domain's module registry.

        This property returns a dictionary containing all Python modules that have been
        documented and registered within the current Sphinx documentation build. Each
        module entry contains metadata about the module including its location,
        synopsis, platform information, and deprecation status.

        Returns:
            dict[str, ModuleEntry]: A dictionary mapping module names to ModuleEntry
                objects. Each ModuleEntry contains:
                - docname: The document name where the module is defined
                - node_id: The unique identifier for the module's target node
                - synopsis: A brief description of the module's purpose
                - platform: Platform-specific information for the module
                - deprecated: Boolean indicating if the module is deprecated

        Notes:
            This property provides read-only access to the modules data structure.
            The underlying data is automatically managed by the domain when modules
            are processed during documentation building. Module entries are created
            via the note_module() method when py:module directives are encountered.

            The returned dictionary is used internally by Sphinx for cross-reference
            resolution, index generation, and module listing functionality.
        """
        <your code>

    def note_module(
        self,
        name: str,
        node_id: str,
        synopsis: str,
        platform: str,
        deprecated: bool
    ) -> None:
        """
        Note a python module for cross reference.

                .. versionadded:: 2.1

        """
        <your code>

    def clear_doc(self, docname: str) -> None:
        """
        Clear all Python objects and modules associated with a specific document.

        This method removes all Python domain entries (objects and modules) that belong
        to the specified document from the domain's internal data structures. It is
        typically called when a document is being rebuilt or removed to ensure that
        stale references are cleaned up.

        Parameters
        ----------
        docname : str
            The name of the document whose Python objects and modules should be
            removed from the domain. This should match the docname used when the
            objects were originally registered.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method performs a two-phase cleanup:

        1. **Objects cleanup**: Iterates through all registered Python objects 
           (functions, classes, methods, attributes, etc.) and removes those that
           belong to the specified document.

        2. **Modules cleanup**: Iterates through all registered Python modules
           and removes those that belong to the specified document.

        The method uses list comprehension to identify objects to remove before
        deletion to avoid modifying dictionaries during iteration, which could
        cause runtime errors.

        This is an internal method used by Sphinx's build system during document
        processing and should not typically be called directly by user code.
        """
        <your code>

    def merge_domaindata(
        self,
        docnames: Set[str],
        otherdata: dict[str, Any]
    ) -> None:
        """
        Merge domain data from another environment during parallel builds.

        This method is called during parallel documentation builds to merge Python domain
        data from worker processes back into the main environment. It integrates object
        definitions and module information that were collected in separate processes.

        Parameters:
            docnames (Set[str]): A set of document names that are being merged from the
                other environment. Only objects and modules belonging to documents in
                this set will be merged to avoid conflicts with data from other sources.
            otherdata (dict[str, Any]): The domain data dictionary from another environment
                containing 'objects' and 'modules' keys. The 'objects' key maps object
                names to ObjectEntry instances, while 'modules' key maps module names to
                ModuleEntry instances.

        Returns:
            None: This method modifies the current domain's data in-place and does not
            return any value.

        Important Notes:
            - Only merges data for documents specified in the docnames parameter to ensure
              data consistency across parallel builds
            - Does not perform duplicate checking (as noted by the XXX comment), so the
              calling code should ensure data integrity
            - Modifies the current domain's objects and modules dictionaries directly
            - This method is essential for the proper functioning of Sphinx's parallel
              build feature for Python documentation
        """
        <your code>

    def find_obj(
        self,
        env: BuildEnvironment,
        modname: str,
        classname: str,
        name: str,
        type: str | None,
        searchmode: int = 0
    ) -> list[tuple[str, ObjectEntry]]:
        """
        Find a Python object for "name", perhaps using the given module
                and/or classname.  Returns a list of (name, object entry) tuples.

        """
        <your code>

    def resolve_xref(
        self,
        env: BuildEnvironment,
        fromdocname: str,
        builder: Builder,
        type: str,
        target: str,
        node: pending_xref,
        contnode: Element
    ) -> nodes.reference | None:
        """
        Resolve cross-references for Python objects in the documentation.

        This method is responsible for resolving cross-references (xrefs) to Python objects
        such as classes, functions, methods, modules, etc. It searches for the target object
        in the domain's registry and creates appropriate reference nodes if found.

        Parameters:
            env (BuildEnvironment): The Sphinx build environment containing the documentation state
            fromdocname (str): The name of the document containing the cross-reference
            builder (Builder): The Sphinx builder being used to generate the documentation
            type (str): The type of reference being resolved (e.g., 'class', 'func', 'mod', 'attr')
            target (str): The target object name to resolve (may include module/class qualifiers)
            node (pending_xref): The pending cross-reference node that needs to be resolved
            contnode (Element): The content node that will be used as the link text

        Returns:
            nodes.reference | None: A reference node linking to the target object if found,
            or None if the target cannot be resolved. The reference includes the appropriate
            document name, anchor, and display text.

        Important notes:
            - Uses different search modes: exact match (searchmode=0) or fuzzy search (searchmode=1)
            - Implements fallback resolution for certain object types (e.g., 'class' falls back to 'data'/'attr' for type aliases)
            - Handles module context and class context from the cross-reference node
            - Prioritizes canonical (non-aliased) matches when multiple targets are found
            - Emits warnings when multiple non-aliased targets exist for the same reference
            - Special handling for module references using _make_module_refnode method
            - Supports conditional content resolution based on pending_xref_condition
        """
        <your code>

    def resolve_any_xref(
        self,
        env: BuildEnvironment,
        fromdocname: str,
        builder: Builder,
        target: str,
        node: pending_xref,
        contnode: Element
    ) -> list[tuple[str, nodes.reference]]:
        """
        Resolve cross-references for the :any: role in the Python domain.

        This method handles cross-reference resolution when using the :any: role, which can
        reference any type of Python object (modules, classes, functions, etc.) without
        specifying the exact object type.

        Parameters:
            env (BuildEnvironment): The Sphinx build environment containing the documentation state
            fromdocname (str): The name of the document containing the cross-reference
            builder (Builder): The Sphinx builder being used for output generation
            target (str): The target name to resolve (e.g., 'MyClass', 'my_function')
            node (pending_xref): The pending cross-reference node to be resolved
            contnode (Element): The content node that will be used as the link text

        Returns:
            list[tuple[str, nodes.reference]]: A list of tuples where each tuple contains:
                - str: The role name (e.g., 'py:class', 'py:func', 'py:mod')
                - nodes.reference: The resolved reference node that can be used as a link

        Important notes:
            - Always searches in "refspecific" mode (searchmode=1) when used with :any: role
            - Skips aliased objects when multiple matches are found to avoid duplicates
            - Handles modules specially by creating module reference nodes with additional metadata
            - For non-module objects, uses pending cross-reference conditions to determine content
            - Returns empty list if no matches are found
            - The role name in returned tuples is prefixed with 'py:' followed by the appropriate object type
        """
        <your code>

    def _make_module_refnode(
        self,
        builder: Builder,
        fromdocname: str,
        name: str,
        contnode: Node
    ) -> nodes.reference:
        """
        Create a reference node for a Python module cross-reference.

        This method generates a reference node specifically for Python modules, including
        additional metadata such as synopsis, deprecation status, and platform information
        in the reference title.

        Parameters:
            builder (Builder): The Sphinx builder instance used for generating the documentation.
                Contains configuration and methods needed for creating cross-references.
            fromdocname (str): The name of the document from which the reference originates.
                Used to calculate relative paths and handle cross-document linking.
            name (str): The fully qualified name of the module being referenced.
                Must exist as a key in the domain's modules dictionary.
            contnode (Node): The content node that will be wrapped by the reference.
                Typically contains the display text for the reference link.

        Returns:
            nodes.reference: A docutils reference node that links to the specified module.
                The node includes enhanced title information with module synopsis,
                deprecation warnings, and platform details when available.

        Important notes:
            - The module specified by 'name' must exist in self.modules, otherwise a KeyError
              will be raised when accessing module information.
            - The reference title is constructed by combining the module name with optional
              synopsis (prefixed with ': '), deprecation notice (' (deprecated)'), and
              platform information (formatted as ' (platform_name)').
            - This method is specifically designed for module references and should not be
              used for other Python object types like classes, functions, or methods.
        """
        <your code>

    def get_objects(self) -> Iterator[tuple[str, str, str, str, str, int]]:
        """
        Retrieve all documented Python objects from the domain for indexing and search purposes.

        This method iterates through all modules and objects registered in the Python domain
        and yields information about each one for use in Sphinx's object inventory system.
        The yielded data is used for cross-referencing, search functionality, and generating
        object inventories.

        Returns:
            Iterator[tuple[str, str, str, str, str, int]]: An iterator yielding tuples containing:
                - name (str): The display name of the object
                - dispname (str): The display name for presentation (same as name)
                - type (str): The object type ('module', 'function', 'class', etc.)
                - docname (str): The document name where the object is defined
                - anchor (str): The HTML anchor/node ID for linking to the object
                - priority (int): Search priority where:
                    * 0: Standard priority for modules
                    * 1: High priority for regular (non-aliased) objects
                    * -1: Low priority for aliased objects (not full-text searchable)

        Notes:
            - Modules are processed first and always have priority 0
            - Aliased objects receive priority -1 to exclude them from full-text search
            - Regular objects receive priority 1 for normal search ranking
            - Module objects are explicitly filtered out from the objects collection
              since they are already handled separately from the modules collection
        """
        <your code>

    def get_full_qualified_name(self, node: Element) -> str | None:
        """
        Retrieve the fully qualified name for a Python object reference node.

        This method constructs a complete dotted name path for a Python object by combining
        the module name, class name, and target reference from the given docutils Element node.

        Parameters
        ----------
        node : Element
            A docutils Element node representing a Python object reference. The node is
            expected to contain attributes for 'py:module', 'py:class', and 'reftarget'
            that are used to construct the qualified name.

        Returns
        -------
        str | None
            The fully qualified name as a dotted string (e.g., 'module.class.target'),
            or None if the node does not contain a 'reftarget' attribute. Empty or None
            components are automatically filtered out from the final name.

        Notes
        -----
        - The method extracts three key attributes from the node:
          - 'py:module': The module name containing the object
          - 'py:class': The class name containing the object (if applicable)  
          - 'reftarget': The target object name being referenced
        - Components are joined with dots only if they are not None or empty strings
        - If 'reftarget' is None, the method returns None immediately since no valid
          reference target exists
        - This is commonly used in Sphinx documentation processing to resolve cross-references
          and generate proper linking between Python objects
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.