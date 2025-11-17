"""The Python domain."""

from __future__ import annotations

import builtins
import inspect
import typing
from types import NoneType
from typing import TYPE_CHECKING, NamedTuple, cast

from docutils import nodes
from docutils.parsers.rst import directives

from sphinx import addnodes
from sphinx.domains import Domain, Index, IndexEntry, ObjType
from sphinx.domains.python._annotations import _parse_annotation
from sphinx.domains.python._object import PyObject
from sphinx.locale import _, __
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import (
    find_pending_xref_condition,
    make_id,
    make_refnode,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence, Set
    from typing import Any, ClassVar

    from docutils.nodes import Element, Node

    from sphinx.addnodes import desc_signature, pending_xref
    from sphinx.application import Sphinx
    from sphinx.builders import Builder
    from sphinx.environment import BuildEnvironment
    from sphinx.util.typing import ExtensionMetadata, OptionSpec

# re-export objects for backwards compatibility
# See: https://github.com/sphinx-doc/sphinx/issues/12295

from sphinx.domains.python._annotations import (  # NoQA: F401
    _parse_arglist,  # for sphinx-immaterial
    type_to_xref,
)
from sphinx.domains.python._object import (  # NoQA: F401
    PyField,
    PyGroupedField,
    PyTypedField,
    PyXrefMixin,
    py_sig_re,
)

_TYPING_ALL = frozenset(typing.__all__)

logger = logging.getLogger(__name__)

pairindextypes = {
    'module': 'module',
    'keyword': 'keyword',
    'operator': 'operator',
    'object': 'object',
    'exception': 'exception',
    'statement': 'statement',
    'builtin': 'built-in function',
}




































def filter_meta_fields(
    app: Sphinx, domain: str, objtype: str, content: Element
) -> None:
    """Filter ``:meta:`` field from its docstring."""
    if domain != 'py':
        return

    for node in content:
        if isinstance(node, nodes.field_list):
            fields = cast('list[nodes.field]', node)
            # removing list items while iterating the list needs reversed()
            for field in reversed(fields):
                field_name = cast('nodes.field_body', field[0]).astext().strip()
                if field_name == 'meta' or field_name.startswith('meta '):
                    node.remove(field)


class PythonModuleIndex(Index):
    """
    Index subclass to provide the Python module index.
    """

    name = "modindex"
    localname = "_('Python Module Index')"
    shortname = "_('modules')"
    domain = "# Type: PythonDomain"

    def generate(
            self,
            docnames: Iterable[str] | None = None
        ) -> tuple[list[tuple[str, list[IndexEntry]]], bool]:
        raise NotImplementedError('This function has been masked for testing')


class PythonDomain(Domain):
    """
    Python language domain.
    """

    name = "py"
    label = "Python"
    object_types = "{'function': ObjType(_('function'), 'func', 'obj'), 'data': ObjType(_('data'), 'data', 'obj'), 'class': ObjType(_('class'), 'class', 'exc', 'obj'), 'exception': ObjType(_('exception'), 'exc', 'class', 'obj'), 'method': ObjType(_('method'), 'meth', 'obj'), 'classmethod': ObjType(_('class method'), 'meth', 'obj'), 'staticmethod': ObjType(_('static method'), 'meth', 'obj'), 'attribute': ObjType(_('attribute'), 'attr', 'obj'), 'property': ObjType(_('property'), 'attr', '_prop', 'obj'), 'type': ObjType(_('type alias'), 'type', 'obj'), 'module': ObjType(_('module'), 'mod', 'obj')}"
    directives = "{'function': PyFunction, 'data': PyVariable, 'class': PyClasslike, 'exception': PyClasslike, 'method': PyMethod, 'classmethod': PyClassMethod, 'staticmethod': PyStaticMethod, 'attribute': PyAttribute, 'property': PyProperty, 'type': PyTypeAlias, 'module': PyModule, 'currentmodule': PyCurrentModule, 'decorator': PyDecoratorFunction, 'decoratormethod': PyDecoratorMethod}"
    roles = "{'data': PyXRefRole(), 'exc': PyXRefRole(), 'func': PyXRefRole(fix_parens=True), 'deco': _PyDecoXRefRole(), 'class': PyXRefRole(), 'const': PyXRefRole(), 'attr': PyXRefRole(), 'type': PyXRefRole(), 'meth': PyXRefRole(fix_parens=True), 'mod': PyXRefRole(), 'obj': PyXRefRole()}"
    initial_data = {'objects': {}, 'modules': {}}
    indices = "[PythonModuleIndex]"

    @property
    def objects(self) -> dict[str, ObjectEntry]:
        raise NotImplementedError('This function has been masked for testing')

    def note_object(
            self,
            name: str,
            objtype: str,
            node_id: str,
            aliased: bool = False,
            location: Any = None
        ) -> None:
        """
        Note a python object for cross reference.

                .. versionadded:: 2.1

        """
        raise NotImplementedError('This function has been masked for testing')

    @property
    def modules(self) -> dict[str, ModuleEntry]:
        raise NotImplementedError('This function has been masked for testing')

    def note_module(
            self,
            name: str,
            node_id: str,
            synopsis: str,
            platform: str,
            deprecated: bool
        ) -> None:
        """
        Note a python module for cross reference.

                .. versionadded:: 2.1

        """
        raise NotImplementedError('This function has been masked for testing')

    def clear_doc(self, docname: str) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def merge_domaindata(
            self,
            docnames: Set[str],
            otherdata: dict[str, Any]
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def find_obj(
            self,
            env: BuildEnvironment,
            modname: str,
            classname: str,
            name: str,
            type: str | None,
            searchmode: int = 0
        ) -> list[tuple[str, ObjectEntry]]:
        """
        Find a Python object for "name", perhaps using the given module
                and/or classname.  Returns a list of (name, object entry) tuples.

        """
        raise NotImplementedError('This function has been masked for testing')

    def resolve_xref(
            self,
            env: BuildEnvironment,
            fromdocname: str,
            builder: Builder,
            type: str,
            target: str,
            node: pending_xref,
            contnode: Element
        ) -> nodes.reference | None:
        raise NotImplementedError('This function has been masked for testing')

    def resolve_any_xref(
            self,
            env: BuildEnvironment,
            fromdocname: str,
            builder: Builder,
            target: str,
            node: pending_xref,
            contnode: Element
        ) -> list[tuple[str, nodes.reference]]:
        raise NotImplementedError('This function has been masked for testing')

    def _make_module_refnode(
            self,
            builder: Builder,
            fromdocname: str,
            name: str,
            contnode: Node
        ) -> nodes.reference:
        raise NotImplementedError('This function has been masked for testing')

    def get_objects(self) -> Iterator[tuple[str, str, str, str, str, int]]:
        raise NotImplementedError('This function has been masked for testing')

    def get_full_qualified_name(self, node: Element) -> str | None:
        raise NotImplementedError('This function has been masked for testing')


def builtin_resolver(
    app: Sphinx, env: BuildEnvironment, node: pending_xref, contnode: Element
) -> Element | None:
    """Do not emit nitpicky warnings for built-in types."""
    if node.get('refdomain') != 'py':
        return None
    elif node.get('reftype') in {'class', 'obj'} and node.get('reftarget') == 'None':
        return contnode
    elif node.get('reftype') in {'class', 'obj', 'exc'}:
        reftarget = node.get('reftarget')
        if inspect.isclass(getattr(builtins, reftarget, None)):
            # built-in class
            return contnode
        if _is_typing(reftarget):
            # typing class
            return contnode

    return None


def _is_typing(s: str, /) -> bool:
    return s.removeprefix('typing.') in _TYPING_ALL


def setup(app: Sphinx) -> ExtensionMetadata:
    app.setup_extension('sphinx.directives')

    app.add_domain(PythonDomain)
    app.add_config_value(
        'python_use_unqualified_type_names', False, 'env', types=frozenset({bool})
    )
    app.add_config_value(
        'python_maximum_signature_line_length',
        None,
        'env',
        types=frozenset({int, NoneType}),
    )
    app.add_config_value(
        'python_trailing_comma_in_multi_line_signatures',
        True,
        'env',
        types=frozenset({bool}),
    )
    app.add_config_value(
        'python_display_short_literal_types', False, 'env', types=frozenset({bool})
    )
    app.connect('object-description-transform', filter_meta_fields)
    app.connect('missing-reference', builtin_resolver, priority=900)

    return {
        'version': 'builtin',
        'env_version': 4,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }