from __future__ import annotations

from typing import TYPE_CHECKING

from sphinx.domains.cpp._ast import (
    ASTDeclaration,
    ASTNestedName,
    ASTNestedNameElement,
)
from sphinx.locale import __
from sphinx.util import logging

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from typing import Any, NoReturn

    from sphinx.domains.cpp._ast import (
        ASTIdentifier,
        ASTOperator,
        ASTTemplateArgs,
        ASTTemplateDeclarationPrefix,
        ASTTemplateIntroduction,
        ASTTemplateParams,
    )
    from sphinx.environment import BuildEnvironment

logger = logging.getLogger(__name__)

















