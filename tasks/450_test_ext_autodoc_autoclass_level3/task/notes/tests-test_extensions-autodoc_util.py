from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

from sphinx.ext.autodoc._directive_options import (
    _AutoDocumenterOptions,
    _process_documenter_options,
)

# NEVER import those objects from sphinx.ext.autodoc directly
from sphinx.ext.autodoc.directive import DocumenterBridge
from sphinx.util.docutils import LoggingReporter

if TYPE_CHECKING:
    from typing import Any

    from docutils.statemachine import StringList

    from sphinx.application import Sphinx


def do_autodoc(
        app: Sphinx,
        objtype: str,
        name: str,
        options: dict[str, Any] | None = None
    ) -> StringList:
    raise NotImplementedError('This function has been masked for testing')