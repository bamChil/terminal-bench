## Task
**Task: Implement Apache Iceberg Table Reader**

Create a function to read Apache Iceberg tables and convert them into pandas DataFrames. The core functionality should support:

1. **Data Access**: Connect to and read from Iceberg tables using table identifiers and optional catalog configurations
2. **Query Capabilities**: Enable data filtering, column selection, row limiting, and time-travel queries via snapshot IDs
3. **Flexibility**: Handle various catalog properties, case-sensitive column matching, and additional scan configurations

**Key Considerations**: 
- Maintain compatibility with different catalog systems and storage backends
- Optimize performance for large-scale data operations
- Handle experimental API status with potential future changes
- Ensure proper error handling for invalid table identifiers or connection issues

The implementation should balance ease of use with advanced querying capabilities while maintaining efficient data transfer from distributed Iceberg tables to local pandas DataFrames.

**NOTE**: 
- This test comes from the `pandas` library, and we have given you the content of this code repository under `/testbed/`, and you need to complete based on this code repository and supplement the files we specify. Remember, all your changes must be in this codebase, and changes that are not in this codebase will not be discovered and tested by us.
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
from pandas.io.iceberg import read_iceberg
```
This means that we will test one function/class: read_iceberg.
And the defination and implementation of class read_iceberg should be in `/testbed/pandas/io/iceberg.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
def read_iceberg(
    table_identifier: str,
    catalog_name: str | None = None
) -> DataFrame:
    """
    
        Read an Apache Iceberg table into a pandas DataFrame.
    
        .. versionadded:: 3.0.0
    
        .. warning::
    
           read_iceberg is experimental and may change without warning.
    
        Parameters
        ----------
        table_identifier : str
            Table identifier.
        catalog_name : str, optional
            The name of the catalog.
        catalog_properties : dict of {str: str}, optional
            The properties that are used next to the catalog configuration.
        row_filter : str, optional
            A string that describes the desired rows.
        selected_fields : tuple of str, optional
            A tuple of strings representing the column names to return in the output
            dataframe.
        case_sensitive : bool, default True
            If True column matching is case sensitive.
        snapshot_id : int, optional
            Snapshot ID to time travel to. By default the table will be scanned as of the
            current snapshot ID.
        limit : int, optional
            An integer representing the number of rows to return in the scan result.
            By default all matching rows will be fetched.
        scan_properties : dict of {str: obj}, optional
            Additional Table properties as a dictionary of string key value pairs to use
            for this scan.
    
        Returns
        -------
        DataFrame
            DataFrame based on the Iceberg table.
    
        See Also
        --------
        read_parquet : Read a Parquet file.
    
        Examples
        --------
        >>> df = pd.read_iceberg(
        ...     table_identifier="my_table",
        ...     catalog_name="my_catalog",
        ...     catalog_properties={"s3.secret-access-key": "my-secret"},
        ...     row_filter="trip_distance >= 10.0",
        ...     selected_fields=("VendorID", "tpep_pickup_datetime"),
        ... )  # doctest: +SKIP
        
    """
    <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from pandas.io.iceberg import read_iceberg
```

### Interface Description 1
Below is **Interface Description 1** for file: pandas-io-iceberg.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
def read_iceberg(
    table_identifier: str,
    catalog_name: str | None = None
) -> DataFrame:
    """
    
        Read an Apache Iceberg table into a pandas DataFrame.
    
        .. versionadded:: 3.0.0
    
        .. warning::
    
           read_iceberg is experimental and may change without warning.
    
        Parameters
        ----------
        table_identifier : str
            Table identifier.
        catalog_name : str, optional
            The name of the catalog.
        catalog_properties : dict of {str: str}, optional
            The properties that are used next to the catalog configuration.
        row_filter : str, optional
            A string that describes the desired rows.
        selected_fields : tuple of str, optional
            A tuple of strings representing the column names to return in the output
            dataframe.
        case_sensitive : bool, default True
            If True column matching is case sensitive.
        snapshot_id : int, optional
            Snapshot ID to time travel to. By default the table will be scanned as of the
            current snapshot ID.
        limit : int, optional
            An integer representing the number of rows to return in the scan result.
            By default all matching rows will be fetched.
        scan_properties : dict of {str: obj}, optional
            Additional Table properties as a dictionary of string key value pairs to use
            for this scan.
    
        Returns
        -------
        DataFrame
            DataFrame based on the Iceberg table.
    
        See Also
        --------
        read_parquet : Read a Parquet file.
    
        Examples
        --------
        >>> df = pd.read_iceberg(
        ...     table_identifier="my_table",
        ...     catalog_name="my_catalog",
        ...     catalog_properties={"s3.secret-access-key": "my-secret"},
        ...     row_filter="trip_distance >= 10.0",
        ...     selected_fields=("VendorID", "tpep_pickup_datetime"),
        ... )  # doctest: +SKIP
        
    """
    <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.