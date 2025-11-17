from __future__ import annotations

import ast
import functools
import operator
import token
from collections import deque
from inspect import Parameter
from itertools import chain, islice
from typing import TYPE_CHECKING

from docutils import nodes

from sphinx import addnodes
from sphinx.addnodes import pending_xref, pending_xref_condition
from sphinx.locale import _
from sphinx.pycode.parser import Token, TokenProcessor
from sphinx.util.inspect import signature_from_str

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any

    from docutils.nodes import Element, Node

    from sphinx.addnodes import desc_signature
    from sphinx.environment import BuildEnvironment






def _parse_annotation(
        annotation: str,
        env: BuildEnvironment
    ) -> list[Node]:
    """
    Parse type annotation.
    """
    raise NotImplementedError('This function has been masked for testing')


class _TypeParameterListParser(TokenProcessor):
    def __init__(self, sig: str) -> None:
        signature = sig.replace('\n', '').strip()
        super().__init__([signature])
        # Each item is a tuple (name, kind, default, annotation) mimicking
        # ``inspect.Parameter`` to allow default values on VAR_POSITIONAL
        # or VAR_KEYWORD parameters.
        self.type_params: list[tuple[str, int, Any, Any]] = []

    def fetch_type_param_spec(self) -> list[Token]:
        tokens = []
        while current := self.fetch_token():
            tokens.append(current)
            for ldelim, rdelim in ('(', ')'), ('{', '}'), ('[', ']'):
                if current == [token.OP, ldelim]:
                    tokens += self.fetch_until([token.OP, rdelim])
                    break
            else:
                if current == token.INDENT:
                    tokens += self.fetch_until(token.DEDENT)
                elif current.match([token.OP, ':'], [token.OP, '='], [token.OP, ',']):
                    tokens.pop()
                    break
        return tokens

    def parse(self) -> None:
        while current := self.fetch_token():
            if current == token.NAME:
                tp_name = current.value.strip()
                if self.previous and self.previous.match(
                    [token.OP, '*'], [token.OP, '**']
                ):
                    if self.previous == [token.OP, '*']:
                        tp_kind = Parameter.VAR_POSITIONAL
                    else:
                        tp_kind = Parameter.VAR_KEYWORD  # type: ignore[assignment]
                else:
                    tp_kind = Parameter.POSITIONAL_OR_KEYWORD  # type: ignore[assignment]

                tp_ann: Any = Parameter.empty
                tp_default: Any = Parameter.empty

                current = self.fetch_token()
                if current and current.match([token.OP, ':'], [token.OP, '=']):
                    if current == [token.OP, ':']:
                        tokens = self.fetch_type_param_spec()
                        tp_ann = self._build_identifier(tokens)

                    if self.current and self.current == [token.OP, '=']:
                        tokens = self.fetch_type_param_spec()
                        tp_default = self._build_identifier(tokens)

                if (
                    tp_kind != Parameter.POSITIONAL_OR_KEYWORD
                    and tp_ann != Parameter.empty
                ):
                    msg = (
                        'type parameter bound or constraint is not allowed '
                        f'for {tp_kind.description} parameters'
                    )
                    raise SyntaxError(msg)

                type_param = (tp_name, tp_kind, tp_default, tp_ann)
                self.type_params.append(type_param)

    def _build_identifier(self, tokens: list[Token]) -> str:
        idents: list[str] = []
        tokens: Iterable[Token] = iter(tokens)  # type: ignore[no-redef]
        # do not format opening brackets
        for tok in tokens:
            if not tok.match([token.OP, '('], [token.OP, '['], [token.OP, '{']):
                # check if the first non-delimiter character is an unpack operator
                is_unpack_operator = tok.match([token.OP, '*'], [token.OP, ['**']])
                idents.append(self._pformat_token(tok, native=is_unpack_operator))
                break
            idents.append(tok.value)

        # check the remaining tokens
        stop = Token(token.ENDMARKER, '', (-1, -1), (-1, -1), '<sentinel>')
        is_unpack_operator = False
        for tok, op, after in _triplewise(chain(tokens, [stop, stop])):
            ident = self._pformat_token(tok, native=is_unpack_operator)
            idents.append(ident)
            # determine if the next token is an unpack operator depending
            # on the left and right hand side of the operator symbol
            is_unpack_operator = op.match([token.OP, '*'], [token.OP, '**']) and not (
                tok.match(
                    token.NAME,
                    token.NUMBER,
                    token.STRING,
                    [token.OP, ')'],
                    [token.OP, ']'],
                    [token.OP, '}'],
                )
                and after.match(
                    token.NAME,
                    token.NUMBER,
                    token.STRING,
                    [token.OP, '('],
                    [token.OP, '['],
                    [token.OP, '{'],
                )
            )

        return ''.join(idents).strip()

    def _pformat_token(self, tok: Token, native: bool = False) -> str:
        if native:
            return tok.value

        if tok.match(token.NEWLINE, token.ENDMARKER):
            return ''

        if tok.match([token.OP, ':'], [token.OP, ','], [token.OP, '#']):
            return f'{tok.value} '

        # Arithmetic operators are allowed because PEP 695 specifies the
        # default type parameter to be *any* expression (so "T1 << T2" is
        # allowed if it makes sense). The caller is responsible to ensure
        # that a multiplication operator ("*") is not to be confused with
        # an unpack operator (which will not be surrounded by spaces).
        #
        # The operators are ordered according to how likely they are to
        # be used and for (possible) future implementations (e.g., "&" for
        # an intersection type).
        if tok.match(
            # Most likely operators to appear
            [token.OP, '='], [token.OP, '|'],
            # Type composition (future compatibility)
            [token.OP, '&'], [token.OP, '^'], [token.OP, '<'], [token.OP, '>'],
            # Unlikely type composition
            [token.OP, '+'], [token.OP, '-'], [token.OP, '*'], [token.OP, '**'],
            # Unlikely operators but included for completeness
            [token.OP, '@'], [token.OP, '/'], [token.OP, '//'], [token.OP, '%'],
            [token.OP, '<<'], [token.OP, '>>'], [token.OP, '>>>'],
            [token.OP, '<='], [token.OP, '>='], [token.OP, '=='], [token.OP, '!='],
        ):  # fmt: skip
            return f' {tok.value} '

        return tok.value


def _parse_type_list(
    tp_list: str,
    env: BuildEnvironment,
    multi_line_parameter_list: bool = False,
    trailing_comma: bool = True,
) -> addnodes.desc_type_parameter_list:
    """Parse a list of type parameters according to PEP 695."""
    type_params = addnodes.desc_type_parameter_list(tp_list)
    type_params['multi_line_parameter_list'] = multi_line_parameter_list
    type_params['multi_line_trailing_comma'] = trailing_comma
    # formal parameter names are interpreted as type parameter names and
    # type annotations are interpreted as type parameter bound or constraints
    parser = _TypeParameterListParser(tp_list)
    parser.parse()
    for tp_name, tp_kind, tp_default, tp_ann in parser.type_params:
        # no positional-only or keyword-only allowed in a type parameters list
        if tp_kind in {Parameter.POSITIONAL_ONLY, Parameter.KEYWORD_ONLY}:
            msg = (
                'positional-only or keyword-only parameters '
                'are prohibited in type parameter lists'
            )
            raise SyntaxError(msg)

        node = addnodes.desc_type_parameter()
        if tp_kind == Parameter.VAR_POSITIONAL:
            node += addnodes.desc_sig_operator('', '*')
        elif tp_kind == Parameter.VAR_KEYWORD:
            node += addnodes.desc_sig_operator('', '**')
        node += addnodes.desc_sig_name('', tp_name)

        if tp_ann is not Parameter.empty:
            annotation = _parse_annotation(tp_ann, env)
            if not annotation:
                continue

            node += addnodes.desc_sig_punctuation('', ':')
            node += addnodes.desc_sig_space()

            type_ann_expr = addnodes.desc_sig_name('', '', *annotation)  # type: ignore[arg-type]
            # a type bound is ``T: U`` whereas type constraints
            # must be enclosed with parentheses. ``T: (U, V)``
            if tp_ann.startswith('(') and tp_ann.endswith(')'):
                type_ann_text = type_ann_expr.astext()
                if type_ann_text.startswith('(') and type_ann_text.endswith(')'):
                    node += type_ann_expr
                else:
                    # surrounding braces are lost when using _parse_annotation()
                    node += addnodes.desc_sig_punctuation('', '(')
                    node += type_ann_expr  # type constraint
                    node += addnodes.desc_sig_punctuation('', ')')
            else:
                node += type_ann_expr  # type bound

        if tp_default is not Parameter.empty:
            # Always surround '=' with spaces, even if there is no annotation
            node += addnodes.desc_sig_space()
            node += addnodes.desc_sig_operator('', '=')
            node += addnodes.desc_sig_space()
            node += nodes.inline(
                '', tp_default, classes=['default_value'], support_smartquotes=False
            )

        type_params += node
    return type_params


def _parse_arglist(
    arglist: str,
    env: BuildEnvironment,
    multi_line_parameter_list: bool = False,
    trailing_comma: bool = True,
) -> addnodes.desc_parameterlist:
    """Parse a list of arguments using AST parser"""
    params = addnodes.desc_parameterlist(arglist)
    params['multi_line_parameter_list'] = multi_line_parameter_list
    params['multi_line_trailing_comma'] = trailing_comma
    sig = signature_from_str('(%s)' % arglist)
    last_kind = None
    for param in sig.parameters.values():
        if param.kind != param.POSITIONAL_ONLY and last_kind == param.POSITIONAL_ONLY:
            params += _positional_only_separator()
        if param.kind == param.KEYWORD_ONLY and last_kind in {
            param.POSITIONAL_OR_KEYWORD,
            param.POSITIONAL_ONLY,
            None,
        }:
            params += _keyword_only_separator()

        node = addnodes.desc_parameter()
        if param.kind == param.VAR_POSITIONAL:
            node += addnodes.desc_sig_operator('', '*')
            node += addnodes.desc_sig_name('', param.name)
        elif param.kind == param.VAR_KEYWORD:
            node += addnodes.desc_sig_operator('', '**')
            node += addnodes.desc_sig_name('', param.name)
        else:
            node += addnodes.desc_sig_name('', param.name)

        if param.annotation is not param.empty:
            children = _parse_annotation(param.annotation, env)
            node += addnodes.desc_sig_punctuation('', ':')
            node += addnodes.desc_sig_space()
            node += addnodes.desc_sig_name('', '', *children)  # type: ignore[arg-type]
        if param.default is not param.empty:
            if param.annotation is not param.empty:
                node += addnodes.desc_sig_space()
                node += addnodes.desc_sig_operator('', '=')
                node += addnodes.desc_sig_space()
            else:
                node += addnodes.desc_sig_operator('', '=')
            node += nodes.inline(
                '', param.default, classes=['default_value'], support_smartquotes=False
            )

        params += node
        last_kind = param.kind

    if last_kind == Parameter.POSITIONAL_ONLY:
        params += _positional_only_separator()

    return params


def _positional_only_separator() -> addnodes.desc_parameter:
    # PEP 570: Separator for positional only parameters: /
    positional_only_abbr = nodes.abbreviation(
        '/', '/', explanation=_('Positional-only parameter separator (PEP 570)')
    )
    positional_only_op = addnodes.desc_sig_operator(
        '/', '', positional_only_abbr, classes=['positional-only-separator']
    )
    return addnodes.desc_parameter('/', '', positional_only_op)


def _keyword_only_separator() -> addnodes.desc_parameter:
    # PEP 3102: Separator for keyword only parameters: *
    keyword_only_abbr = nodes.abbreviation(
        '*', '*', explanation=_('Keyword-only parameters separator (PEP 3102)')
    )
    keyword_only_op = addnodes.desc_sig_operator(
        '*', '', keyword_only_abbr, classes=['keyword-only-separator']
    )
    return addnodes.desc_parameter('*', '', keyword_only_op)


def _pseudo_parse_arglist(
        signode: desc_signature,
        arglist: str
    ) -> None:
    """
    'Parse' a list of arguments separated by commas.

        Arguments can have "optional" annotations given by enclosing them in
        brackets.  Currently, this will split at any comma, even if it's inside a
        string literal (e.g. default argument value).

    """
    raise NotImplementedError('This function has been masked for testing')


def _triplewise(iterable: Iterable[Token]) -> Iterator[tuple[Token, ...]]:
    # sliding_window('ABCDEFG', 4) --> ABCD BCDE CDEF DEFG
    it = iter(iterable)
    window = deque(islice(it, 3), maxlen=3)
    if len(window) == 3:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)