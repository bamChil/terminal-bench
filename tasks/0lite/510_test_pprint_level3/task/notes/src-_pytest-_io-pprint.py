# mypy: allow-untyped-defs
# This module was imported from the cpython standard library
# (https://github.com/python/cpython/) at commit
# c5140945c723ae6c4b7ee81ff720ac8ea4b52cfd (python3.12).
#
#
#  Original Author:      Fred L. Drake, Jr.
#                        fdrake@acm.org
#
#  This is a simple little module I wrote to make life easier.  I didn't
#  see anything quite like it in the library, though I may have overlooked
#  something.  I wrote this when I was trying to read some heavily nested
#  tuples with fairly non-descriptive content.  This is modeled very much
#  after Lisp/Scheme - style pretty-printing of lists.  If you find it
#  useful, thank small children who sleep at night.
from __future__ import annotations

import collections as _collections
from collections.abc import Callable
from collections.abc import Iterator
import dataclasses as _dataclasses
from io import StringIO as _StringIO
import re
import types as _types
from typing import Any
from typing import IO






class PrettyPrinter:

    _dispatch = {}

    def __init__(
            self,
            indent: int = 4,
            width: int = 80,
            depth: int | None = None
        ) -> None:
        """
        Handle pretty printing operations onto a stream using a set of
                configured parameters.

                indent
                    Number of spaces to indent for each level of nesting.

                width
                    Attempted maximum number of columns in the output.

                depth
                    The maximum depth to print out nested structures.


        """
        raise NotImplementedError('This function has been masked for testing')

    def pformat(self, object: Any) -> str:
        raise NotImplementedError('This function has been masked for testing')

    def _format(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_dataclass(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_dict(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_ordered_dict(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_list(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_tuple(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_set(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_str(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_bytes(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_bytearray(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_mappingproxy(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_simplenamespace(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _format_dict_items(
            self,
            items: list[tuple[Any, Any]],
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _format_namespace_items(
            self,
            items: list[tuple[Any, Any]],
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _format_items(
            self,
            items: list[Any],
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _repr(
            self,
            object: Any,
            context: set[int],
            level: int
        ) -> str:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_default_dict(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_counter(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_chain_map(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_deque(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_user_dict(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_user_list(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _pprint_user_string(
            self,
            object: Any,
            stream: IO[str],
            indent: int,
            allowance: int,
            context: set[int],
            level: int
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _safe_repr(
            self,
            object: Any,
            context: set[int],
            maxlevels: int | None,
            level: int
        ) -> str:
        raise NotImplementedError('This function has been masked for testing')


_builtin_scalars = frozenset(
    {str, bytes, bytearray, float, complex, bool, type(None), int}
)



