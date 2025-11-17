"""Importer utilities for autodoc"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import traceback
import typing
from importlib.abc import FileLoader
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import decode_source, find_spec, module_from_spec, spec_from_loader
from pathlib import Path
from typing import TYPE_CHECKING, NewType, TypeVar

from sphinx.errors import PycodeError
from sphinx.ext.autodoc._property_types import (
    _AssignStatementProperties,
    _ClassDefProperties,
    _FunctionDefProperties,
    _ItemProperties,
    _ModuleProperties,
)
from sphinx.ext.autodoc._sentinels import (
    RUNTIME_INSTANCE_ATTRIBUTE,
    SLOTS_ATTR,
    UNINITIALIZED_ATTR,
)
from sphinx.ext.autodoc.mock import ismock, mock, undecorate
from sphinx.locale import __
from sphinx.pycode import ModuleAnalyzer
from sphinx.util import inspect, logging
from sphinx.util.inspect import (
    isclass,
    safe_getattr,
)
from sphinx.util.typing import get_type_hints

if TYPE_CHECKING:
    from collections.abc import Sequence
    from importlib.machinery import ModuleSpec
    from types import ModuleType
    from typing import Any, Protocol

    from sphinx.environment import BuildEnvironment, _CurrentDocument
    from sphinx.ext.autodoc._property_types import _AutodocFuncProperty, _AutodocObjType

    class _AttrGetter(Protocol):
        def __call__(self, obj: Any, name: str, default: Any = ..., /) -> Any: ...


_NATIVE_SUFFIXES: frozenset[str] = frozenset({'.pyx', *EXTENSION_SUFFIXES})
logger = logging.getLogger(__name__)














def import_object(
    modname: str,
    objpath: list[str],
    objtype: str = '',
    attrgetter: _AttrGetter = safe_getattr,
) -> Any:
    ret = _import_from_module_and_path(
        module_name=modname, obj_path=objpath, get_attr=attrgetter
    )
    if isinstance(ret, _ImportedObject):
        return [ret.module, ret.parent, ret.object_name, ret.obj]
    return None























