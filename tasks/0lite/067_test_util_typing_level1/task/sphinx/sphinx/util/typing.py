"""The composite types for Sphinx."""

from __future__ import annotations

import dataclasses
import sys
import types
import typing
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from docutils import nodes
from docutils.parsers.rst.states import Inliner

from sphinx.util import logging

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Annotated, Any, Final, Literal, Protocol, TypeAlias

    from typing_extensions import TypeIs

    from sphinx.application import Sphinx
    from sphinx.util.inventory import _InventoryItem

    _RestifyMode: TypeAlias = Literal[
        'fully-qualified-except-typing',
        'smart',
    ]
    _StringifyMode: TypeAlias = Literal[
        'fully-qualified-except-typing',
        'fully-qualified',
        'smart',
    ]

logger = logging.getLogger(__name__)


# classes that have an incorrect .__module__ attribute
# Map of (__module__, __qualname__) to the correct fully-qualified name
_INVALID_BUILTIN_CLASSES: Final[Mapping[tuple[str, str], str]] = {
    # types from 'contextvars'
    ('_contextvars', 'Context'): 'contextvars.Context',
    ('_contextvars', 'ContextVar'): 'contextvars.ContextVar',
    ('_contextvars', 'Token'): 'contextvars.Token',
    # types from 'ctypes':
    ('_ctypes', 'Array'): 'ctypes.Array',
    ('_ctypes', 'Structure'): 'ctypes.Structure',
    ('_ctypes', 'Union'): 'ctypes.Union',
    # types from 'io':
    ('_io', 'BufferedRandom'): 'io.BufferedRandom',
    ('_io', 'BufferedReader'): 'io.BufferedReader',
    ('_io', 'BufferedRWPair'): 'io.BufferedRWPair',
    ('_io', 'BufferedWriter'): 'io.BufferedWriter',
    ('_io', 'BytesIO'): 'io.BytesIO',
    ('_io', 'FileIO'): 'io.FileIO',
    ('_io', 'StringIO'): 'io.StringIO',
    ('_io', 'TextIOWrapper'): 'io.TextIOWrapper',
    # types from 'json':
    ('json.decoder', 'JSONDecoder'): 'json.JSONDecoder',
    ('json.encoder', 'JSONEncoder'): 'json.JSONEncoder',
    # types from 'lzma':
    ('_lzma', 'LZMACompressor'): 'lzma.LZMACompressor',
    ('_lzma', 'LZMADecompressor'): 'lzma.LZMADecompressor',
    # types from 'multiprocessing':
    ('multiprocessing.context', 'Process'): 'multiprocessing.Process',
    # types from 'pathlib':
    ('pathlib._local', 'Path'): 'pathlib.Path',
    ('pathlib._local', 'PosixPath'): 'pathlib.PosixPath',
    ('pathlib._local', 'PurePath'): 'pathlib.PurePath',
    ('pathlib._local', 'PurePosixPath'): 'pathlib.PurePosixPath',
    ('pathlib._local', 'PureWindowsPath'): 'pathlib.PureWindowsPath',
    ('pathlib._local', 'WindowsPath'): 'pathlib.WindowsPath',
    # types from 'pickle':
    ('_pickle', 'Pickler'): 'pickle.Pickler',
    ('_pickle', 'Unpickler'): 'pickle.Unpickler',
    # types from 'struct':
    ('_struct', 'Struct'): 'struct.Struct',
    # types from 'types':
    ('builtins', 'async_generator'): 'types.AsyncGeneratorType',
    ('builtins', 'builtin_function_or_method'): 'types.BuiltinMethodType',
    ('builtins', 'cell'): 'types.CellType',
    ('builtins', 'classmethod_descriptor'): 'types.ClassMethodDescriptorType',
    ('builtins', 'code'): 'types.CodeType',
    ('builtins', 'coroutine'): 'types.CoroutineType',
    ('builtins', 'ellipsis'): 'types.EllipsisType',
    ('builtins', 'frame'): 'types.FrameType',
    ('builtins', 'function'): 'types.LambdaType',
    ('builtins', 'generator'): 'types.GeneratorType',
    ('builtins', 'getset_descriptor'): 'types.GetSetDescriptorType',
    ('builtins', 'mappingproxy'): 'types.MappingProxyType',
    ('builtins', 'member_descriptor'): 'types.MemberDescriptorType',
    ('builtins', 'method'): 'types.MethodType',
    ('builtins', 'method-wrapper'): 'types.MethodWrapperType',
    ('builtins', 'method_descriptor'): 'types.MethodDescriptorType',
    ('builtins', 'module'): 'types.ModuleType',
    ('builtins', 'NoneType'): 'types.NoneType',
    ('builtins', 'NotImplementedType'): 'types.NotImplementedType',
    ('builtins', 'traceback'): 'types.TracebackType',
    ('builtins', 'wrapper_descriptor'): 'types.WrapperDescriptorType',
    # types from 'weakref':
    ('_weakrefset', 'WeakSet'): 'weakref.WeakSet',
    # types from 'zipfile':
    ('zipfile._path', 'CompleteDirs'): 'zipfile.CompleteDirs',
    ('zipfile._path', 'Path'): 'zipfile.Path',
}




# Text like nodes which are initialized with text and rawsource
TextlikeNode: TypeAlias = nodes.Text | nodes.TextElement

# path matcher
PathMatcher: TypeAlias = Callable[[str], bool]

# common role functions
if TYPE_CHECKING:

    class RoleFunction(Protocol):
        def __call__(
            self,
            name: str,
            rawtext: str,
            text: str,
            lineno: int,
            inliner: Inliner,
            /,
            options: dict[str, Any] | None = None,
            content: Sequence[str] = (),
        ) -> tuple[list[nodes.Node], list[nodes.system_message]]: ...

else:
    RoleFunction: TypeAlias = Callable[
        [str, str, str, int, Inliner, dict[str, typing.Any], Sequence[str]],
        tuple[list[nodes.Node], list[nodes.system_message]],
    ]

# A option spec for directive
OptionSpec: TypeAlias = dict[str, Callable[[str], typing.Any]]

# title getter functions for enumerable nodes (see sphinx.domains.std)
TitleGetter: TypeAlias = Callable[[nodes.Node], str]

# inventory data on memory
Inventory: TypeAlias = dict[str, dict[str, '_InventoryItem']]


class ExtensionMetadata(typing.TypedDict, total=False):
    """The metadata returned by an extension's ``setup()`` function.

    See :ref:`ext-metadata`.
    """

    version: str
    """The extension version (default: ``'unknown version'``)."""
    env_version: int
    """An integer that identifies the version of env data added by the extension."""
    parallel_read_safe: bool
    """Indicate whether parallel reading of source files is supported
    by the extension.
    """
    parallel_write_safe: bool
    """Indicate whether parallel writing of output files is supported
    by the extension (default: ``True``).
    """


if TYPE_CHECKING:
    _ExtensionSetupFunc: TypeAlias = Callable[[Sphinx], ExtensionMetadata]  # NoQA: PYI047 (false positive)


def get_type_hints(
    obj: Any,
    globalns: dict[str, Any] | None = None,
    localns: dict[str, Any] | None = None,
    include_extras: bool = False,
) -> dict[str, Any]:
    """Return a dictionary containing type hints for a function, method, module or class
    object.

    This is a simple wrapper of `typing.get_type_hints()` that does not raise an error on
    runtime.
    """
    from sphinx.util.inspect import safe_getattr  # lazy loading

    try:
        return typing.get_type_hints(
            obj, globalns, localns, include_extras=include_extras
        )
    except NameError:
        # Failed to evaluate ForwardRef (maybe TYPE_CHECKING)
        return safe_getattr(obj, '__annotations__', {})
    except AttributeError:
        # Failed to evaluate ForwardRef (maybe not runtime checkable)
        return safe_getattr(obj, '__annotations__', {})
    except TypeError:
        # Invalid object is given. But try to get __annotations__ as a fallback.
        return safe_getattr(obj, '__annotations__', {})
    except KeyError:
        # a broken class found
        # See: https://github.com/sphinx-doc/sphinx/issues/8084
        return {}








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
    raise NotImplementedError('This function has been masked for testing')




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
    raise NotImplementedError('This function has been masked for testing')




# deprecated name -> (object to return, canonical path or empty string, removal version)
_DEPRECATED_OBJECTS: dict[str, tuple[Any, str, tuple[int, int]]] = {
}  # fmt: skip


def __getattr__(name: str) -> Any:
    if name not in _DEPRECATED_OBJECTS:
        msg = f'module {__name__!r} has no attribute {name!r}'
        raise AttributeError(msg)

    from sphinx.deprecation import _deprecation_warning

    deprecated_object, canonical_name, remove = _DEPRECATED_OBJECTS[name]
    _deprecation_warning(__name__, name, canonical_name, remove=remove)
    return deprecated_object