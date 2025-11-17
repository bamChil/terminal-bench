from __future__ import annotations

import functools
import operator
import re
import sys
from inspect import Parameter, Signature
from typing import TYPE_CHECKING, NewType, TypeVar

from docutils.statemachine import StringList

from sphinx.errors import PycodeError
from sphinx.ext.autodoc._directive_options import (
    annotation_option,
    bool_option,
    class_doc_from_option,
    exclude_members_option,
    identity,
    inherited_members_option,
    member_order_option,
    members_option,
)
from sphinx.ext.autodoc._member_finder import _filter_members, _get_members_to_document
from sphinx.ext.autodoc._sentinels import (
    ALL,
    RUNTIME_INSTANCE_ATTRIBUTE,
    SLOTS_ATTR,
    SUPPRESS,
    UNINITIALIZED_ATTR,
)
from sphinx.ext.autodoc.importer import (
    _get_attribute_comment,
    _is_runtime_instance_attribute_not_commented,
    _load_object_by_name,
    _resolve_name,
)
from sphinx.ext.autodoc.mock import ismock
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer
from sphinx.util import inspect, logging
from sphinx.util.docstrings import prepare_docstring, separate_metadata
from sphinx.util.inspect import (
    evaluate_signature,
    getdoc,
    object_description,
    safe_getattr,
    stringify_signature,
)
from sphinx.util.typing import get_type_hints, restify, stringify_annotation

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from types import ModuleType
    from typing import Any, ClassVar, Literal

    from sphinx.config import Config
    from sphinx.environment import BuildEnvironment, _CurrentDocument
    from sphinx.events import EventManager
    from sphinx.ext.autodoc._directive_options import _AutoDocumenterOptions
    from sphinx.ext.autodoc._property_types import (
        _AssignStatementProperties,
        _ClassDefProperties,
        _FunctionDefProperties,
        _ItemProperties,
        _ModuleProperties,
    )
    from sphinx.ext.autodoc.directive import DocumenterBridge
    from sphinx.registry import SphinxComponentRegistry
    from sphinx.util.typing import OptionSpec, _RestifyMode

logger = logging.getLogger('sphinx.ext.autodoc')

#: extended signature RE: with explicit module name separated by ::
py_ext_sig_re = re.compile(
    r"""^ ([\w.]+::)?            # explicit module name
          ([\w.]+\.)?            # module and/or class name(s)
          (\w+)  \s*             # thing name
          (?: \[\s*(.*?)\s*])?   # optional: type parameters list
          (?: \((.*)\)           # optional: arguments
           (?:\s* -> \s* (.*))?  #           return annotation
          )? $                   # and nothing more
    """,
    re.VERBOSE,
)










class DecoratorDocumenter(FunctionDocumenter):
    """Specialized Documenter subclass for decorator functions."""

    props: _FunctionDefProperties

    objtype = 'decorator'

    # must be lower than FunctionDocumenter
    priority = FunctionDocumenter.priority - 1


# Types which have confusing metaclass signatures it would be best not to show.
# These are listed by name, rather than storing the objects themselves, to avoid
# needing to import the modules.
_METACLASS_CALL_BLACKLIST = frozenset({
    'enum.EnumType.__call__',
})


# Types whose __new__ signature is a pass-through.
_CLASS_NEW_BLACKLIST = frozenset({
    'typing.Generic.__new__',
})




class ExceptionDocumenter(ClassDocumenter):
    """Specialized ClassDocumenter subclass for exceptions."""

    props: _ClassDefProperties

    objtype = 'exception'
    member_order = 10

    # needs a higher priority than ClassDocumenter
    priority = ClassDocumenter.priority + 5

    @classmethod
    def can_document_member(
        cls: type[Documenter], member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        try:
            return isinstance(member, type) and issubclass(member, BaseException)
        except TypeError as exc:
            # It's possible for a member to be considered a type, but fail
            # issubclass checks due to not being a class. For example:
            # https://github.com/sphinx-doc/sphinx/issues/11654#issuecomment-1696790436
            msg = (
                f'{cls.__name__} failed to discern if member {member} with'
                f' membername {membername} is a BaseException subclass.'
            )
            raise ValueError(msg) from exc


class DataDocumenter(Documenter):
    """Specialized Documenter subclass for data items."""

    props: _AssignStatementProperties

    __uninitialized_global_variable__ = True

    objtype = 'data'
    member_order = 40
    priority = -10
    option_spec: ClassVar[OptionSpec] = dict(Documenter.option_spec)
    option_spec['annotation'] = annotation_option
    option_spec['no-value'] = bool_option

    @classmethod
    def can_document_member(
        cls: type[Documenter], member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return isinstance(parent, ModuleDocumenter) and isattr

    def update_annotations(self, parent: Any) -> None:
        """Update __annotations__ to support type_comment and so on."""
        annotations = dict(inspect.getannotations(parent))
        parent.__annotations__ = annotations

        try:
            analyzer = ModuleAnalyzer.for_module(self.props.module_name)
            analyzer.analyze()
            for (classname, attrname), annotation in analyzer.annotations.items():
                if not classname and attrname not in annotations:
                    annotations[attrname] = annotation
        except PycodeError:
            pass

    def should_suppress_value_header(self) -> bool:
        if self.props._obj is UNINITIALIZED_ATTR:
            return True
        else:
            doc = self.get_doc() or []
            _docstring, metadata = separate_metadata(
                '\n'.join(functools.reduce(operator.iadd, doc, []))
            )
            if 'hide-value' in metadata:
                return True

        return False

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)
        sourcename = self.get_sourcename()
        if self.options.annotation is SUPPRESS or inspect.isgenericalias(
            self.props._obj
        ):
            pass
        elif self.options.annotation:
            self.add_line('   :annotation: %s' % self.options.annotation, sourcename)
        else:
            if self.config.autodoc_typehints != 'none':
                # obtain annotation for this data
                annotations = get_type_hints(
                    self.parent,
                    None,
                    self.config.autodoc_type_aliases,
                    include_extras=True,
                )
                if self.props.name in annotations:
                    mode = _get_render_mode(self.config.autodoc_typehints_format)
                    short_literals = self.config.python_display_short_literal_types
                    objrepr = stringify_annotation(
                        annotations.get(self.props.name),
                        mode,
                        short_literals=short_literals,
                    )
                    self.add_line('   :type: ' + objrepr, sourcename)

            try:
                if (
                    self.options.no_value
                    or self.should_suppress_value_header()
                    or ismock(self.props._obj)
                ):
                    pass
                else:
                    objrepr = object_description(self.props._obj)
                    self.add_line('   :value: ' + objrepr, sourcename)
            except ValueError:
                pass

    def get_module_comment(self, attrname: str) -> list[str] | None:
        try:
            analyzer = ModuleAnalyzer.for_module(self.props.module_name)
            analyzer.analyze()
            key = ('', attrname)
            if key in analyzer.attr_docs:
                return list(analyzer.attr_docs[key])
        except PycodeError:
            pass

        return None

    def get_doc(self) -> list[list[str]] | None:
        # Check the variable has a docstring-comment
        comment = self.get_module_comment(self.props.name)
        if comment:
            return [comment]
        else:
            return super().get_doc()

    def add_content(self, more_content: StringList | None) -> None:
        # Disable analyzing variable comment on Documenter.add_content() to control it on
        # DataDocumenter.add_content()
        self.analyzer = None

        if not more_content:
            more_content = StringList()

        _add_content_generic_alias_(
            more_content,
            self.props._obj,
            autodoc_typehints_format=self.config.autodoc_typehints_format,
        )
        super().add_content(more_content)


class MethodDocumenter(Documenter):
    """Specialized Documenter subclass for methods (normal, static and class)."""

    props: _FunctionDefProperties

    __docstring_signature__ = True

    objtype = 'method'
    directivetype = 'method'
    member_order = 50
    priority = 1  # must be more than FunctionDocumenter

    @classmethod
    def can_document_member(
        cls: type[Documenter], member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return inspect.isroutine(member) and not isinstance(parent, ModuleDocumenter)

    def format_args(self, **kwargs: Any) -> str:
        if self.config.autodoc_typehints in {'none', 'description'}:
            kwargs.setdefault('show_annotation', False)
        if self.config.autodoc_typehints_format == 'short':
            kwargs.setdefault('unqualified_typehints', True)
        if self.config.python_display_short_literal_types:
            kwargs.setdefault('short_literals', True)

        try:
            if self.props._obj == object.__init__ and self.parent != object:  # NoQA: E721
                # Classes not having own __init__() method are shown as no arguments.
                #
                # Note: The signature of object.__init__() is (self, /, *args, **kwargs).
                #       But it makes users confused.
                args = '()'
            else:
                if inspect.isstaticmethod(
                    self.props._obj, cls=self.parent, name=self.props.object_name
                ):
                    self._events.emit(
                        'autodoc-before-process-signature', self.props._obj, False
                    )
                    sig = inspect.signature(
                        self.props._obj,
                        bound_method=False,
                        type_aliases=self.config.autodoc_type_aliases,
                    )
                else:
                    self._events.emit(
                        'autodoc-before-process-signature', self.props._obj, True
                    )
                    sig = inspect.signature(
                        self.props._obj,
                        bound_method=True,
                        type_aliases=self.config.autodoc_type_aliases,
                    )
                args = stringify_signature(sig, **kwargs)
        except TypeError as exc:
            msg = __('Failed to get a method signature for %s: %s')
            logger.warning(msg, self.props.full_name, exc)
            return ''
        except ValueError:
            args = ''

        if self.config.strip_signature_backslash:
            # escape backslashes for reST
            args = args.replace('\\', '\\\\')
        return args

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)

        sourcename = self.get_sourcename()
        obj = self.parent.__dict__.get(self.props.object_name, self.props._obj)
        if inspect.isabstractmethod(obj):
            self.add_line('   :abstractmethod:', sourcename)
        if inspect.iscoroutinefunction(obj) or inspect.isasyncgenfunction(obj):
            self.add_line('   :async:', sourcename)
        if (
            inspect.is_classmethod_like(obj)
            or inspect.is_singledispatch_method(obj)
            and inspect.is_classmethod_like(obj.func)
        ):
            self.add_line('   :classmethod:', sourcename)
        if inspect.isstaticmethod(obj, cls=self.parent, name=self.props.object_name):
            self.add_line('   :staticmethod:', sourcename)
        if self.analyzer and self.props.dotted_parts in self.analyzer.finals:
            self.add_line('   :final:', sourcename)

    def format_signature(self, **kwargs: Any) -> str:
        if self.config.autodoc_typehints_format == 'short':
            kwargs.setdefault('unqualified_typehints', True)
        if self.config.python_display_short_literal_types:
            kwargs.setdefault('short_literals', True)

        sigs = []
        if (
            self.analyzer
            and self.props.dotted_parts in self.analyzer.overloads
            and self.config.autodoc_typehints != 'none'
        ):
            # Use signatures for overloaded methods instead of the implementation method.
            overloaded = True
        else:
            overloaded = False
            sig = super().format_signature(**kwargs)
            sigs.append(sig)

        meth = self.parent.__dict__.get(self.props.name)
        if inspect.is_singledispatch_method(meth):
            from sphinx.ext.autodoc._property_types import _FunctionDefProperties

            # append signature of singledispatch'ed functions
            for typ, func in meth.dispatcher.registry.items():
                if typ is object:
                    pass  # default implementation. skipped.
                else:
                    if inspect.isclassmethod(func):
                        func = func.__func__
                    dispatchmeth = self.annotate_to_first_argument(func, typ)
                    if dispatchmeth:
                        documenter = MethodDocumenter(self.directive, '')
                        documenter.props = _FunctionDefProperties(
                            obj_type='method',
                            module_name='',
                            parts=('',),
                            docstring_lines=(),
                            _obj=dispatchmeth,
                            _obj___module__=None,
                            properties=frozenset(),
                        )
                        documenter.parent = self.parent
                        sigs.append(documenter.format_signature())
        if overloaded and self.analyzer is not None:
            if inspect.isstaticmethod(
                self.props._obj, cls=self.parent, name=self.props.object_name
            ):
                actual = inspect.signature(
                    self.props._obj,
                    bound_method=False,
                    type_aliases=self.config.autodoc_type_aliases,
                )
            else:
                actual = inspect.signature(
                    self.props._obj,
                    bound_method=True,
                    type_aliases=self.config.autodoc_type_aliases,
                )

            __globals__ = safe_getattr(self.props._obj, '__globals__', {})
            for overload in self.analyzer.overloads[self.props.dotted_parts]:
                overload = self.merge_default_value(actual, overload)
                overload = evaluate_signature(
                    overload, __globals__, self.config.autodoc_type_aliases
                )

                if not inspect.isstaticmethod(
                    self.props._obj, cls=self.parent, name=self.props.object_name
                ):
                    parameters = list(overload.parameters.values())
                    overload = overload.replace(parameters=parameters[1:])
                sig = stringify_signature(overload, **kwargs)
                sigs.append(sig)

        return '\n'.join(sigs)

    def merge_default_value(self, actual: Signature, overload: Signature) -> Signature:
        """Merge default values of actual implementation to the overload variants."""
        parameters = list(overload.parameters.values())
        for i, param in enumerate(parameters):
            actual_param = actual.parameters.get(param.name)
            if actual_param and param.default == '...':
                parameters[i] = param.replace(default=actual_param.default)

        return overload.replace(parameters=parameters)

    def annotate_to_first_argument(
        self, func: Callable[..., Any], typ: type
    ) -> Callable[..., Any] | None:
        """Annotate type hint to the first argument of function if needed."""
        try:
            sig = inspect.signature(func, type_aliases=self.config.autodoc_type_aliases)
        except TypeError as exc:
            msg = __('Failed to get a method signature for %s: %s')
            logger.warning(msg, self.props.full_name, exc)
            return None
        except ValueError:
            return None

        if len(sig.parameters) == 1:
            return None

        def dummy():  # type: ignore[no-untyped-def]  # NoQA: ANN202
            pass

        params = list(sig.parameters.values())
        if params[1].annotation is Parameter.empty:
            params[1] = params[1].replace(annotation=typ)
            try:
                dummy.__signature__ = sig.replace(  # type: ignore[attr-defined]
                    parameters=params
                )
                return dummy
            except (AttributeError, TypeError):
                # failed to update signature (ex. built-in or extension types)
                return None

        return func

    def get_doc(self) -> list[list[str]] | None:
        if self._new_docstrings is not None:
            # docstring already returned previously, then modified due to
            # ``__docstring_signature__ = True``. Just return the
            # previously-computed result, so that we don't loose the processing.
            return self._new_docstrings
        if self.props.name == '__init__':
            docstring = getdoc(
                self.props._obj,
                self.get_attr,
                self.config.autodoc_inherit_docstrings,
                self.parent,
                self.props.object_name,
            )
            if docstring is not None and (
                docstring == object.__init__.__doc__  # for pypy
                or docstring.strip() == object.__init__.__doc__  # for !pypy
            ):
                docstring = None
            if docstring:
                tab_width = self.directive.state.document.settings.tab_width
                return [prepare_docstring(docstring, tabsize=tab_width)]
            else:
                return []
        elif self.props.name == '__new__':
            docstring = getdoc(
                self.props._obj,
                self.get_attr,
                self.config.autodoc_inherit_docstrings,
                self.parent,
                self.props.object_name,
            )
            if docstring is not None and (
                docstring == object.__new__.__doc__  # for pypy
                or docstring.strip() == object.__new__.__doc__  # for !pypy
            ):
                docstring = None
            if docstring:
                tab_width = self.directive.state.document.settings.tab_width
                return [prepare_docstring(docstring, tabsize=tab_width)]
            else:
                return []
        else:
            return super().get_doc()






class DocstringSignatureMixin:
    """Retained for compatibility."""

    __docstring_signature__ = True


class ModuleLevelDocumenter(Documenter):
    """Retained for compatibility."""


class ClassLevelDocumenter(Documenter):
    """Retained for compatibility."""





