import numpy as np

from xarray.core.datatree import Variable


def align_nd_chunks(
        nd_v_chunks: tuple[tuple[int, ...], ...],
        nd_backend_chunks: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, ...], ...]:
    raise NotImplementedError('This function has been masked for testing')


def build_grid_chunks(
        size: int,
        chunk_size: int,
        region: slice | None = None
    ) -> tuple[int, ...]:
    raise NotImplementedError('This function has been masked for testing')


def grid_rechunk(
    v: Variable,
    enc_chunks: tuple[int, ...],
    region: tuple[slice, ...],
) -> Variable:
    nd_v_chunks = v.chunks
    if not nd_v_chunks:
        return v

    nd_grid_chunks = tuple(
        build_grid_chunks(
            v_size,
            region=interval,
            chunk_size=chunk_size,
        )
        for v_size, chunk_size, interval in zip(
            v.shape, enc_chunks, region, strict=True
        )
    )

    nd_aligned_chunks = align_nd_chunks(
        nd_v_chunks=nd_v_chunks,
        nd_backend_chunks=nd_grid_chunks,
    )
    v = v.chunk(dict(zip(v.dims, nd_aligned_chunks, strict=True)))
    return v


def validate_grid_chunks_alignment(
    nd_v_chunks: tuple[tuple[int, ...], ...] | None,
    enc_chunks: tuple[int, ...],
    backend_shape: tuple[int, ...],
    region: tuple[slice, ...],
    allow_partial_chunks: bool,
    name: str,
):
    if nd_v_chunks is None:
        return
    base_error = (
        "Specified Zarr chunks encoding['chunks']={enc_chunks!r} for "
        "variable named {name!r} would overlap multiple Dask chunks. "
        "Please check the Dask chunks at position {v_chunk_pos} and "
        "{v_chunk_pos_next}, on axis {axis}, they are overlapped "
        "on the same Zarr chunk in the region {region}. "
        "Writing this array in parallel with Dask could lead to corrupted data. "
        "To resolve this issue, consider one of the following options: "
        "- Rechunk the array using `chunk()`. "
        "- Modify or delete `encoding['chunks']`. "
        "- Set `safe_chunks=False`. "
        "- Enable automatic chunks alignment with `align_chunks=True`."
    )

    for axis, chunk_size, v_chunks, interval, size in zip(
        range(len(enc_chunks)),
        enc_chunks,
        nd_v_chunks,
        region,
        backend_shape,
        strict=True,
    ):
        for i, chunk in enumerate(v_chunks[1:-1]):
            if chunk % chunk_size:
                raise ValueError(
                    base_error.format(
                        v_chunk_pos=i + 1,
                        v_chunk_pos_next=i + 2,
                        v_chunk_size=chunk,
                        axis=axis,
                        name=name,
                        chunk_size=chunk_size,
                        region=interval,
                        enc_chunks=enc_chunks,
                    )
                )

        interval_start = interval.start or 0

        if len(v_chunks) > 1:
            # The first border size is the amount of data that needs to be updated on the
            # first chunk taking into account the region slice.
            first_border_size = chunk_size
            if allow_partial_chunks:
                first_border_size = chunk_size - interval_start % chunk_size

            if (v_chunks[0] - first_border_size) % chunk_size:
                raise ValueError(
                    base_error.format(
                        v_chunk_pos=0,
                        v_chunk_pos_next=0,
                        v_chunk_size=v_chunks[0],
                        axis=axis,
                        name=name,
                        chunk_size=chunk_size,
                        region=interval,
                        enc_chunks=enc_chunks,
                    )
                )

        if not allow_partial_chunks:
            region_stop = interval.stop or size

            error_on_last_chunk = base_error.format(
                v_chunk_pos=len(v_chunks) - 1,
                v_chunk_pos_next=len(v_chunks) - 1,
                v_chunk_size=v_chunks[-1],
                axis=axis,
                name=name,
                chunk_size=chunk_size,
                region=interval,
                enc_chunks=enc_chunks,
            )
            if interval_start % chunk_size:
                # The last chunk which can also be the only one is a partial chunk
                # if it is not aligned at the beginning
                raise ValueError(error_on_last_chunk)

            if np.ceil(region_stop / chunk_size) == np.ceil(size / chunk_size):
                # If the region is covering the last chunk then check
                # if the reminder with the default chunk size
                # is equal to the size of the last chunk
                if v_chunks[-1] % chunk_size != size % chunk_size:
                    raise ValueError(error_on_last_chunk)
            elif v_chunks[-1] % chunk_size:
                raise ValueError(error_on_last_chunk)