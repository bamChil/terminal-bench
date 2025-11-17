"""Docutils transforms used by Sphinx when reading documents."""

from __future__ import annotations

from re import DOTALL, match
from textwrap import indent
from typing import TYPE_CHECKING, Any, TypeVar

import docutils.utils
from docutils import nodes

from sphinx import addnodes
from sphinx.domains.std import make_glossary_term, split_term_classifiers
from sphinx.errors import ConfigError
from sphinx.locale import __
from sphinx.locale import init as init_locale
from sphinx.transforms import SphinxTransform
from sphinx.util import get_filetype, logging
from sphinx.util.docutils import LoggingReporter
from sphinx.util.i18n import docname_to_domain
from sphinx.util.index_entries import split_index_msg
from sphinx.util.nodes import (
    IMAGE_TYPE_NODES,
    LITERAL_TYPE_NODES,
    NodeMatcher,
    extract_messages,
    traverse_translatable_index,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from docutils.frontend import Values

    from sphinx.application import Sphinx
    from sphinx.config import Config
    from sphinx.environment import BuildEnvironment
    from sphinx.registry import SphinxComponentRegistry
    from sphinx.util.typing import ExtensionMetadata


logger = logging.getLogger(__name__)

# The attributes not copied to the translated node
#
# * refexplict: For allow to give (or not to give) an explicit title
#               to the pending_xref on translation
EXCLUDED_PENDING_XREF_ATTRIBUTES = ('refexplicit',)


N = TypeVar('N', bound=nodes.Node)






class PreserveTranslatableMessages(SphinxTransform):
    """Preserve original translatable messages before translation"""

    default_priority = 10  # this MUST be invoked before Locale transform

    def apply(self, **kwargs: Any) -> None:
        for node in self.document.findall(addnodes.translatable):
            node.preserve_original_messages()




class Locale(SphinxTransform):
    """
    Replace translatable nodes with their translated doctree.
    """

    default_priority = 20

    def apply(self, **kwargs: Any) -> None:
        raise NotImplementedError('This function has been masked for testing')


class TranslationProgressTotaliser(SphinxTransform):
    """Calculate the number of translated and untranslated nodes."""

    default_priority = 25  # MUST happen after Locale

    def apply(self, **kwargs: Any) -> None:
        from sphinx.builders.gettext import MessageCatalogBuilder

        if issubclass(self.env._builder_cls, MessageCatalogBuilder):
            return

        total = translated = 0
        for node in NodeMatcher(nodes.Element, translated=Any).findall(self.document):
            total += 1
            if node['translated']:
                translated += 1

        self.document['translation_progress'] = {
            'total': total,
            'translated': translated,
        }


class AddTranslationClasses(SphinxTransform):
    """Add ``translated`` or ``untranslated`` classes to indicate translation status."""

    default_priority = 950

    def apply(self, **kwargs: Any) -> None:
        from sphinx.builders.gettext import MessageCatalogBuilder

        if issubclass(self.env._builder_cls, MessageCatalogBuilder):
            return

        if not self.config.translation_progress_classes:
            return

        if self.config.translation_progress_classes is True:
            add_translated = add_untranslated = True
        elif self.config.translation_progress_classes == 'translated':
            add_translated = True
            add_untranslated = False
        elif self.config.translation_progress_classes == 'untranslated':
            add_translated = False
            add_untranslated = True
        else:
            msg = (
                'translation_progress_classes must be '
                'True, False, "translated" or "untranslated"'
            )
            raise ConfigError(msg)

        for node in NodeMatcher(nodes.Element, translated=Any).findall(self.document):
            if node['translated']:
                if add_translated:
                    node.setdefault('classes', []).append('translated')  # type: ignore[arg-type]
            else:
                if add_untranslated:
                    node.setdefault('classes', []).append('untranslated')  # type: ignore[arg-type]


class RemoveTranslatableInline(SphinxTransform):
    """Remove inline nodes used for translation as placeholders."""

    default_priority = 999

    def apply(self, **kwargs: Any) -> None:
        from sphinx.builders.gettext import MessageCatalogBuilder

        if issubclass(self.env._builder_cls, MessageCatalogBuilder):
            return

        matcher = NodeMatcher(nodes.inline, translatable=Any)
        for inline in matcher.findall(self.document):
            inline.parent.remove(inline)
            inline.parent += inline.children


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_transform(PreserveTranslatableMessages)
    app.add_transform(Locale)
    app.add_transform(TranslationProgressTotaliser)
    app.add_transform(AddTranslationClasses)
    app.add_transform(RemoveTranslatableInline)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }