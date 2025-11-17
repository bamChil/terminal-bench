## Task
**Task: N-Dimensional Array Chunk Alignment System**

**Core Functionality:**
Develop a chunking system that optimally aligns multi-dimensional array data between variable storage preferences and backend storage requirements (e.g., Zarr, Dask).

**Main Features & Requirements:**
- Align n-dimensional chunk specifications while preserving original variable chunks when possible
- Generate grid-based chunks that respect backend storage boundaries and handle partial edge chunks
- Support regional chunking within specified slices of dimensions
- Maintain memory efficiency through intelligent chunk merging strategies

**Key Challenges:**
- Handle dimension mismatches and ensure uniform backend chunk sizes (except borders)
- Balance between variable chunking preferences and backend storage constraints
- Properly align chunks when regions don't start at natural boundaries
- Optimize memory usage while maintaining compatibility across different storage systems

**NOTE**: 
- This test comes from the `xarray` library, and we have given you the content of this code repository under `/testbed/`, and you need to complete based on this code repository and supplement the files we specify. Remember, all your changes must be in this codebase, and changes that are not in this codebase will not be discovered and tested by us.
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
from xarray.backends.chunks import align_nd_chunks
```
This means that we will test one function/class: align_nd_chunks.
And the defination and implementation of class align_nd_chunks should be in `/testbed/xarray/backends/chunks.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
def align_nd_chunks(
    nd_v_chunks: tuple[tuple[int, ...], ...],
    nd_backend_chunks: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    """
    Align n-dimensional chunks between variable chunks and backend storage chunks.
    
    This function takes two sets of n-dimensional chunk specifications and produces
    an aligned chunking scheme that optimizes memory usage while maintaining 
    compatibility with the backend storage format. The alignment algorithm attempts
    to preserve original variable chunks when possible, but will merge chunks when
    necessary to ensure proper alignment with backend chunk boundaries.
    
    Parameters
    ----------
    nd_v_chunks : tuple[tuple[int, ...], ...]
        N-dimensional tuple representing the chunk sizes for each dimension of the
        variable. Each inner tuple contains the chunk sizes along one dimension.
    nd_backend_chunks : tuple[tuple[int, ...], ...]
        N-dimensional tuple representing the chunk sizes for each dimension of the
        backend storage system (e.g., Zarr). Each inner tuple contains the chunk
        sizes along one dimension. Must have the same number of dimensions as
        nd_v_chunks.
    
    Returns
    -------
    tuple[tuple[int, ...], ...]
        N-dimensional tuple representing the aligned chunk sizes for each dimension.
        The returned chunking scheme is optimized to work efficiently with both
        the variable's preferred chunking and the backend storage requirements.
    
    Raises
    ------
    ValueError
        If the number of dimensions between nd_v_chunks and nd_backend_chunks
        do not match.
    ValueError
        If the total number of elements (sum of chunks) differs between
        corresponding dimensions of nd_v_chunks and nd_backend_chunks.
    ValueError
        If backend chunks are not of uniform size excluding border chunks.
        This indicates an unsupported backend chunking pattern.
    
    Notes
    -----
    The alignment algorithm makes several key assumptions:
    - Backend chunks should be uniform in size except for border chunks
    - The algorithm prioritizes preserving original variable chunks when possible
    - When chunks must be merged, it attempts to minimize memory overhead
    - Border chunks (first and last) are handled specially to account for
      partial chunks at array boundaries
    
    The function uses a sophisticated merging strategy that adds artificial
    padding to borders during processing, then removes it to produce the
    final aligned chunks. This ensures proper alignment while minimizing
    unnecessary chunk splits or memory usage.
    """
    <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from xarray.backends.chunks import align_nd_chunks
from xarray.backends.chunks import build_grid_chunks
```

### Interface Description 1
Below is **Interface Description 1** for file: xarray-backends-chunks.py

This file contains 2 top-level interface(s) that need to be implemented.

```python
def align_nd_chunks(
    nd_v_chunks: tuple[tuple[int, ...], ...],
    nd_backend_chunks: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    """
    Align n-dimensional chunks between variable chunks and backend storage chunks.
    
    This function takes two sets of n-dimensional chunk specifications and produces
    an aligned chunking scheme that optimizes memory usage while maintaining 
    compatibility with the backend storage format. The alignment algorithm attempts
    to preserve original variable chunks when possible, but will merge chunks when
    necessary to ensure proper alignment with backend chunk boundaries.
    
    Parameters
    ----------
    nd_v_chunks : tuple[tuple[int, ...], ...]
        N-dimensional tuple representing the chunk sizes for each dimension of the
        variable. Each inner tuple contains the chunk sizes along one dimension.
    nd_backend_chunks : tuple[tuple[int, ...], ...]
        N-dimensional tuple representing the chunk sizes for each dimension of the
        backend storage system (e.g., Zarr). Each inner tuple contains the chunk
        sizes along one dimension. Must have the same number of dimensions as
        nd_v_chunks.
    
    Returns
    -------
    tuple[tuple[int, ...], ...]
        N-dimensional tuple representing the aligned chunk sizes for each dimension.
        The returned chunking scheme is optimized to work efficiently with both
        the variable's preferred chunking and the backend storage requirements.
    
    Raises
    ------
    ValueError
        If the number of dimensions between nd_v_chunks and nd_backend_chunks
        do not match.
    ValueError
        If the total number of elements (sum of chunks) differs between
        corresponding dimensions of nd_v_chunks and nd_backend_chunks.
    ValueError
        If backend chunks are not of uniform size excluding border chunks.
        This indicates an unsupported backend chunking pattern.
    
    Notes
    -----
    The alignment algorithm makes several key assumptions:
    - Backend chunks should be uniform in size except for border chunks
    - The algorithm prioritizes preserving original variable chunks when possible
    - When chunks must be merged, it attempts to minimize memory overhead
    - Border chunks (first and last) are handled specially to account for
      partial chunks at array boundaries
    
    The function uses a sophisticated merging strategy that adds artificial
    padding to borders during processing, then removes it to produce the
    final aligned chunks. This ensures proper alignment while minimizing
    unnecessary chunk splits or memory usage.
    """
    <your code>

def build_grid_chunks(
    size: int,
    chunk_size: int,
    region: slice | None = None
) -> tuple[int, ...]:
    """
    Generate chunks for a grid-based storage backend (e.g., Zarr) within a specified region.
    
    This function creates a tuple of chunk sizes that align with the grid structure of the
    backend storage system. It handles partial chunks at boundaries and ensures proper
    alignment when a region slice is specified.
    
    Parameters
    ----------
    size : int
        The total size of the dimension to be chunked.
    chunk_size : int
        The desired size for each chunk in the grid. This typically corresponds to
        the chunk size of the backend storage format.
    region : slice or None, optional
        A slice object defining the region within the dimension to chunk. If None,
        the entire dimension from 0 to size is used. Default is None.
    
    Returns
    -------
    tuple[int, ...]
        A tuple of integers representing the size of each chunk. The first chunk
        may be smaller than chunk_size if the region doesn't start at a chunk
        boundary. The last chunk may also be smaller if the total size doesn't
        divide evenly by chunk_size.
    
    Notes
    -----
    - When the region start is not aligned with chunk boundaries, the first chunk
      will be sized to align with the grid structure of the backend storage.
    - If the calculated first chunk size is greater than or equal to the total
      size, a single chunk containing the entire dimension is returned.
    - This function is designed to work with chunked array libraries like Dask
      and storage backends like Zarr that have fixed grid structures.
    
    Examples
    --------
    For a dimension of size 13 with chunk_size 5 and no region:
    Returns (5, 5, 3) - two full chunks and one partial chunk
    
    For a dimension of size 10 with chunk_size 5 starting at position 2:
    Returns (3, 5, 2) - first chunk aligns with grid, then full chunk, then remainder
    """
    <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.