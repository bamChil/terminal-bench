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








class MockLoader(Loader):
    """A loader for mocking."""

    def __init__(self, finder: MockFinder) -> None:
        super().__init__()
        self.finder = finder

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        logger.debug('[autodoc] adding a mock module as %s!', spec.name)
        self.finder.mocked_modules.append(spec.name)
        return _MockModule(spec.name)

    def exec_module(self, module: ModuleType) -> None:
        pass  # nothing to do


class MockFinder(MetaPathFinder):
    """A finder for mocking."""

    def __init__(self, modnames: list[str]) -> None:
        super().__init__()
        self.modnames = modnames
        self.loader = MockLoader(self)
        self.mocked_modules: list[str] = []

    def find_spec(
        self,
        fullname: str,
        path: Sequence[bytes | str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        for modname in self.modnames:
            # check if fullname is (or is a descendant of) one of our targets
            if modname == fullname or fullname.startswith(modname + '.'):
                return ModuleSpec(fullname, self.loader)

        return None

    def invalidate_caches(self) -> None:
        """Invalidate mocked modules on sys.modules."""
        for modname in self.mocked_modules:
            sys.modules.pop(modname, None)


@contextlib.contextmanager
def mock(modnames: list[str]) -> Iterator[None]:
    """Insert mock modules during context::

    with mock(['target.module.name']):
        # mock modules are enabled here
        ...
    """
    finder = MockFinder(modnames)
    try:
        sys.meta_path.insert(0, finder)
        yield
    finally:
        sys.meta_path.remove(finder)
        finder.invalidate_caches()


def ismockmodule(subject: Any) -> TypeIs[_MockModule]:
    """Check if the object is a mocked module."""
    return isinstance(subject, _MockModule)




def undecorate(subject: _MockObject) -> Any:
    """Unwrap mock if *subject* is decorated by mocked object.

    If not decorated, returns given *subject* itself.
    """
    if ismock(subject) and subject.__sphinx_decorator_args__:
        return subject.__sphinx_decorator_args__[0]
    else:
        return subject