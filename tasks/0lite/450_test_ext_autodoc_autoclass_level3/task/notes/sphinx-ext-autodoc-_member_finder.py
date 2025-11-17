from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING, Literal

from sphinx.errors import PycodeError
from sphinx.events import EventManager
from sphinx.ext.autodoc._directive_options import _AutoDocumenterOptions
from sphinx.ext.autodoc._property_types import _ClassDefProperties, _ModuleProperties
from sphinx.ext.autodoc._sentinels import ALL, INSTANCE_ATTR, SLOTS_ATTR
from sphinx.ext.autodoc.mock import ismock, undecorate
from sphinx.locale import __
from sphinx.pycode import ModuleAnalyzer
from sphinx.util import inspect, logging
from sphinx.util.docstrings import separate_metadata
from sphinx.util.inspect import (
    getannotations,
    getdoc,
    getmro,
    getslots,
    isclass,
    isenumclass,
    safe_getattr,
    unwrap_all,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence, Set
    from typing import Any, Literal

    from sphinx.events import EventManager
    from sphinx.ext.autodoc._directive_options import _AutoDocumenterOptions
    from sphinx.ext.autodoc._property_types import (
        _ClassDefProperties,
        _ModuleProperties,
    )
    from sphinx.ext.autodoc._sentinels import (
        ALL_T,
        EMPTY_T,
        INSTANCE_ATTR_T,
        SLOTS_ATTR_T,
    )
    from sphinx.ext.autodoc.importer import _AttrGetter

logger = logging.getLogger('sphinx.ext.autodoc')
special_member_re = re.compile(r'^__\S+__$')






def _filter_members(
    obj_members_seq: Iterable[ObjectMember],
    *,
    want_all: bool,
    events: EventManager,
    get_attr: _AttrGetter,
    options: _AutoDocumenterOptions,
    orig_name: str,
    props: _ModuleProperties | _ClassDefProperties,
    inherit_docstrings: bool,
    inherited_members: Set[str],
    exclude_members: EMPTY_T | Set[str] | None,
    special_members: ALL_T | Sequence[str] | None,
    private_members: ALL_T | Sequence[str] | None,
    undoc_members: Literal[True] | None,
    attr_docs: dict[tuple[str, str], list[str]],
) -> Iterator[tuple[str, Any, bool]]:
    # search for members in source code too
    namespace = props.dotted_parts  # will be empty for modules

    # process members and determine which to skip
    for obj in obj_members_seq:
        member_name = obj.__name__
        member_obj = obj.object
        has_attr_doc = (namespace, member_name) in attr_docs
        try:
            keep = _should_keep_member(
                member_name=member_name,
                member_obj=member_obj,
                member_docstring=obj.docstring,
                member_cls=obj.class_,
                get_attr=get_attr,
                has_attr_doc=has_attr_doc,
                inherit_docstrings=inherit_docstrings,
                inherited_members=inherited_members,
                parent=props._obj,
                want_all=want_all,
                exclude_members=exclude_members,
                special_members=special_members,
                private_members=private_members,
                undoc_members=undoc_members,
            )
        except Exception as exc:
            logger.warning(
                __(
                    'autodoc: failed to determine %s.%s (%r) to be documented, '
                    'the following exception was raised:\n%s'
                ),
                orig_name,
                member_name,
                member_obj,
                exc,
                type='autodoc',
            )
            keep = False

        # give the user a chance to decide whether this member
        # should be skipped
        if events is not None:
            # let extensions preprocess docstrings
            skip_user = events.emit_firstresult(
                'autodoc-skip-member',
                props.obj_type,
                member_name,
                member_obj,
                not keep,
                options,
            )
            if skip_user is not None:
                keep = not skip_user

        if keep:
            # if is_attr is True, the member is documented as an attribute
            is_attr = member_obj is INSTANCE_ATTR or has_attr_doc
            yield member_name, member_obj, is_attr








def _should_keep_member(
    *,
    member_name: str,
    member_obj: Any,
    member_docstring: Sequence[str] | None,
    member_cls: Any,
    get_attr: _AttrGetter,
    has_attr_doc: bool,
    inherit_docstrings: bool,
    inherited_members: Set[str],
    parent: Any,
    want_all: bool,
    exclude_members: EMPTY_T | Set[str] | None,
    special_members: ALL_T | Sequence[str] | None,
    private_members: ALL_T | Sequence[str] | None,
    undoc_members: Literal[True] | None,
) -> bool:
    if member_docstring:
        # hack for ClassDocumenter to inject docstring
        doclines: Sequence[str] | None = member_docstring
    else:
        doc = getdoc(
            member_obj,
            get_attr,
            inherit_docstrings,
            parent,
            member_name,
        )
        # Ignore non-string __doc__
        doclines = doc.splitlines() if isinstance(doc, str) else None

        # if the member __doc__ is the same as self's __doc__, it's just
        # inherited and therefore not the member's doc
        cls = get_attr(member_obj, '__class__', None)
        if cls:
            cls_doc = get_attr(cls, '__doc__', None)
            if cls_doc == doc:
                doclines = None

    if doclines is not None:
        doc, metadata = separate_metadata('\n'.join(doclines))
    else:
        doc = ''
        metadata = {}
    has_doc = bool(doc or undoc_members)

    if 'private' in metadata:
        # consider a member private if docstring has "private" metadata
        is_private = True
    elif 'public' in metadata:
        # consider a member public if docstring has "public" metadata
        is_private = False
    else:
        is_private = member_name.startswith('_')

    if ismock(member_obj) and not has_attr_doc:
        # mocked module or object
        return False

    if exclude_members and member_name in exclude_members:
        # remove members given by exclude-members
        return False

    if not want_all:
        # keep documented attributes
        return has_doc or has_attr_doc

    is_filtered_inherited_member = _is_filtered_inherited_member(
        member_name,
        member_cls=member_cls,
        parent=parent,
        inherited_members=inherited_members,
        get_attr=get_attr,
    )

    if special_member_re.match(member_name):
        # special __methods__
        if special_members and member_name in special_members:
            if member_name == '__doc__':  # NoQA: SIM114
                return False
            elif is_filtered_inherited_member:
                return False
            return has_doc
        return False

    if is_private:
        if has_attr_doc or has_doc:
            if private_members is None:  # NoQA: SIM114
                return False
            elif has_doc and is_filtered_inherited_member:
                return False
            return member_name in private_members
        return False

    if has_attr_doc:
        # keep documented attributes
        return True

    if is_filtered_inherited_member:
        return False

    # ignore undocumented members if :undoc-members: is not given
    return has_doc


def _is_filtered_inherited_member(
    member_name: str,
    *,
    member_cls: Any,
    parent: Any,
    inherited_members: Set[str],
    get_attr: _AttrGetter,
) -> bool:
    if not inspect.isclass(parent):
        return False

    seen = set()
    for cls in parent.__mro__:
        if member_name in cls.__dict__:
            seen.add(cls)
        if (
            cls.__name__ in inherited_members
            and cls != parent
            and any(issubclass(potential_child, cls) for potential_child in seen)
        ):
            # given member is a member of specified *super class*
            return True
        if member_cls is cls:
            return False
        if member_name in cls.__dict__:
            return False
        if member_name in get_attr(cls, '__annotations__', {}):
            return False
    return False