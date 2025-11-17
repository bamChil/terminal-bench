from __future__ import annotations

import functools
import re
from contextlib import suppress
from inspect import isclass
from typing import TYPE_CHECKING, Any

from polars.datatypes import (
    Binary,
    Boolean,
    Date,
    Datetime,
    Decimal,
    Duration,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    List,
    Null,
    String,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)
from polars.datatypes._parse import parse_py_type_into_dtype
from polars.datatypes.group import (
    INTEGER_DTYPES,
    UNSIGNED_INTEGER_DTYPES,
)

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


def dtype_from_database_typename(value: str) -> PolarsDataType | None:
    """

        Attempt to infer Polars dtype from database cursor `type_code` string value.

        Examples
        --------
        >>> dtype_from_database_typename("INT2")
        Int16
        >>> dtype_from_database_typename("NVARCHAR")
        String
        >>> dtype_from_database_typename("NUMERIC(10,2)")
        Decimal(precision=10, scale=2)
        >>> dtype_from_database_typename("TIMESTAMP WITHOUT TZ")
        Datetime(time_unit='us', time_zone=None)

    """
    raise NotImplementedError('This function has been masked for testing')


def dtype_from_cursor_description(
    cursor: Any,
    description: tuple[Any, ...],
) -> PolarsDataType | None:
    """Attempt to infer Polars dtype from database cursor description `type_code`."""
    type_code, _disp_size, internal_size, precision, scale, *_ = description
    dtype: PolarsDataType | None = None

    if isclass(type_code):
        # python types, eg: int, float, str, etc
        with suppress(TypeError):
            dtype = parse_py_type_into_dtype(type_code)  # type: ignore[arg-type]

    elif isinstance(type_code, str):
        # database/sql type names, eg: "VARCHAR", "NUMERIC", "BLOB", etc
        dtype = dtype_from_database_typename(
            value=type_code,
            raise_unmatched=False,
        )

    # check additional cursor attrs to refine dtype specification
    if dtype is not None:
        if dtype == Float64 and internal_size == 4:
            dtype = Float32

        elif dtype in INTEGER_DTYPES and internal_size in (2, 4, 8):
            bits = internal_size * 8
            dtype = integer_dtype_from_nbits(
                bits,
                unsigned=(dtype in UNSIGNED_INTEGER_DTYPES),
                default=dtype,
            )
        elif (
            dtype == Decimal
            and isinstance(precision, int)
            and isinstance(scale, int)
            and precision <= 38
            and scale <= 38
        ):
            dtype = Decimal(precision, scale)

    return dtype



