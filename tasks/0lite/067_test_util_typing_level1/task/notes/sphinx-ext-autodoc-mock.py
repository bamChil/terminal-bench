"""mock for autodoc"""

from __future__ import annotations

import contextlib
import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import MethodType, ModuleType
from typing import TYPE_CHECKING

from sphinx.util import logging
from sphinx.util.inspect import isboundmethod, safe_getattr

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Any

    from typing_extensions import TypeIs

logger = logging.getLogger(__name__)












@contextlib.contextmanager
def mock(modnames: list[str]) -> Iterator[None]:
    """
    Insert mock modules during context::

        with mock(['target.module.name']):
            # mock modules are enabled here
            ...

    """
    raise NotImplementedError('This function has been masked for testing')






def undecorate(subject: _MockObject) -> Any:
    """Unwrap mock if *subject* is decorated by mocked object.

    If not decorated, returns given *subject* itself.
    """
    if ismock(subject) and subject.__sphinx_decorator_args__:
        return subject.__sphinx_decorator_args__[0]
    else:
        return subject