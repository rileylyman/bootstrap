from __future__ import annotations

import dataclasses
import enum
import inspect
from typing import Any, List, NoReturn, Optional, Self, Tuple


@dataclasses.dataclass
class Span:
    line_start: int
    line_end: int
    col_start: int
    col_end: int

    def display(self, source: str) -> str:
        lines = source.splitlines()[self.line_start - 1 : self.line_end]
        if len(lines) == 1:
            s = lines[0] + "\n"
            for i in range(len(lines[0])):
                if i >= self.col_start and i < self.col_end:
                    s += "^"
                else:
                    s += " "
            return s
        else:
            assert False


class TokenType(enum.IntEnum):
    LET = enum.auto()
    ID = enum.auto()
    COLON = enum.auto()
    OPAREN = enum.auto()
    CPAREN = enum.auto()
    OBRACK = enum.auto()
    CBRACK = enum.auto()
    ARROW = enum.auto()
    DOT = enum.auto()
    STAR = enum.auto()
    AMP = enum.auto()
    COMMA = enum.auto()
    EQUALS = enum.auto()
    INDENT = enum.auto()
    DEDENT = enum.auto()
    INT_LITERAL = enum.auto()
    STRING_LITERAL = enum.auto()


class Token:
    def __init__(self, value: Any, ttype: TokenType, span: Span = Span(0, 0, 0, 0)):
        self.value = value
        self.ttype = ttype
        self.span = span

    def display(self, source: str):
        return f"{self.ttype.name}:\n{self.span.display(source)}"

    def __str__(self):
        value_str = "" if not self.value else f"({str(self.value)})"
        return f"{self.ttype.name}{value_str}"


class Lexer:
    keywords = {"let": TokenType.LET}

    def __init__(self, source: str):
        self._source = source
        self._bol = 0
        self._line = 1
        self._cur = 0
        self._curr_indent = 0

    def _col(self) -> int:
        return self._cur - self._bol

    def _is_more(self) -> bool:
        return self._cur < len(self._source)

    def _cur_char(self) -> str:
        assert self._is_more()
        return self._source[self._cur]

    def _eat_one(self):
        if self._cur_char() == "\n":
            self._line += 1
            self._bol = self._cur + 1

        self._cur += 1

    @staticmethod
    def _retrofill_span(fun):
        def wrapper(self, *args, **kwargs) -> Optional[Token | List[Token]]:
            first_line = self._line
            first_col = self._col()
            token = fun(self, *args, **kwargs)
            if not token:
                return None

            if isinstance(token, list):
                for t in token:
                    t.span = Span(first_line, self._line, first_col, self._col())
            else:
                token.span = Span(first_line, self._line, first_col, self._col())
            return token

        return wrapper

    @_retrofill_span
    def _lex_keyword_or_id(self) -> Optional[Token]:
        chars = []
        while self._is_more() and (
            self._cur_char().isalnum() or self._cur_char() == "_"
        ):
            if not chars and self._cur_char().isnumeric():
                return None
            chars.append(self._cur_char())
            self._eat_one()

        if not chars:
            return None

        s = "".join(chars)
        return Token(s, self.keywords.get(s, TokenType.ID))

    @_retrofill_span
    def _lex_int_literal(self) -> Optional[Token]:
        chars = []
        while self._is_more() and self._cur_char().isnumeric():
            chars.append(self._cur_char())
            self._eat_one()

        if not chars:
            return None

        return Token(int("".join(chars)), TokenType.INT_LITERAL)

    @_retrofill_span
    def _lex_string_literal(self) -> Optional[Token]:
        chars = []
        if self._cur_char() != '"':
            return None
        self._eat_one()
        while self._is_more() and self._cur_char() != '"':
            chars.append(self._cur_char())
            self._eat_one()

        assert self._is_more(), "unexpected end of input while parsing string literal"
        self._eat_one()

        return Token("".join(chars), TokenType.STRING_LITERAL)

    @_retrofill_span
    def _lex_single(self) -> Optional[Token]:
        current = self._cur_char()
        match current:
            case "(":
                self._eat_one()
                return Token("(", TokenType.OPAREN)
            case ")":
                self._eat_one()
                return Token(")", TokenType.CPAREN)
            case "[":
                self._eat_one()
                return Token("[", TokenType.OBRACK)
            case "]":
                self._eat_one()
                return Token("]", TokenType.CBRACK)
            case ":":
                self._eat_one()
                return Token(":", TokenType.COLON)
            case "=":
                self._eat_one()
                return Token("=", TokenType.EQUALS)
            case ".":
                self._eat_one()
                return Token(".", TokenType.DOT)
            case "*":
                self._eat_one()
                return Token("*", TokenType.DOT)
            case "&":
                self._eat_one()
                return Token("&", TokenType.AMP)
            case ",":
                self._eat_one()
                return Token(",", TokenType.COMMA)
            case "-":
                self._eat_one()
                if self._is_more() and self._cur_char() == ">":
                    self._eat_one()
                    return Token("->", TokenType.ARROW)

    @_retrofill_span
    def _lex_dent(self) -> Optional[List[Token]]:
        # TODO: assert no tabs in entire source
        if self._col() != 0:
            return None
        count = 0
        while self._cur_char() == " ":
            count += 1
            self._eat_one()

        assert count % 2 == 0, f"expected even count, got {count}"
        indents = count // 2
        if indents == self._curr_indent:
            return None

        ret = []
        ttype = TokenType.INDENT if indents > self._curr_indent else TokenType.DEDENT
        for _ in range(abs(indents - self._curr_indent)):
            ret.append(Token(None, ttype))

        self._curr_indent = indents

        return ret

    def _trim_left(self):
        stuff_behind = lambda: self._source[self._bol : self._cur].strip()
        stuff_ahead = lambda: self._source[
            self._cur : self._source.find("\n", self._cur)
        ].strip()
        while self._is_more() and self._cur_char().isspace():
            if stuff_ahead() and stuff_behind():
                self._eat_one()
            elif stuff_behind() or (not stuff_ahead() and not stuff_behind()):
                self._eat_one()
            else:
                break

    def tokens(self) -> List[Token]:
        ret = []
        while True:
            found = False
            self._trim_left()
            if not self._is_more():
                for _ in range(self._curr_indent):
                    ret.append(Token(None, TokenType.DEDENT, Span(1, 1, 0, 0)))
                break

            if tokens := self._lex_dent():
                ret.extend(tokens)

            for f in [
                self._lex_int_literal,
                self._lex_keyword_or_id,
                self._lex_string_literal,
                self._lex_single,
            ]:
                if token := f():
                    ret.append(token)
                    found = True
            assert found

        return ret


class Node:
    def show(self, source: str, indent: int = 0):
        print(indent * " " + self.__class__.__name__ + ":")
        members = inspect.getmembers(self, lambda a: not inspect.isroutine(a))
        for key, val in members:
            if key.startswith("__"):
                continue
            print((indent + 2) * " " + key + ":")

            def show(v: Any, source: str, indent: int):
                if dict(inspect.getmembers(v)).get("show"):
                    v.show(source, indent)
                else:
                    print(indent * " " + str(v))

            if isinstance(val, list):
                for v in val:
                    show(v, source, indent + 4)
            else:
                show(val, source, indent + 4)


class Statement(Node):
    @classmethod
    def parse(cls, tokens: List[Token], source: str) -> Tuple[Self, List[Token]]:
        match tokens[0].ttype:
            case TokenType.LET:
                return VarDeclStmt.parse(tokens, source)
            case _:
                return AssignStmt.parse(tokens, source)

    @staticmethod
    def is_stmt_next(tokens: List[Token]) -> bool:
        match tokens[0].ttype:
            case TokenType.LET:
                return True
            case TokenType.ID:
                if len(tokens) < 2:
                    return False
                return tokens[1].ttype == TokenType.EQUALS
        return False


class TypeAnnotation(Node):
    @classmethod
    def parse(cls, tokens: List[Token], source: str) -> Tuple[Self, List[Token]]:
        match tokens[0].ttype:
            case TokenType.OPAREN:
                return FnTypeAnnotation.parse(tokens, source)
            case _:
                return TypeNameAnnotation.parse(tokens, source)


@dataclasses.dataclass
class TypeNameAnnotation(TypeAnnotation):
    name: Token

    @classmethod
    def parse(cls, tokens: List[Token], source: str) -> Tuple[Self, List[Token]]:
        name, tokens = expect(tokens, [TokenType.ID], source)
        return cls(name), tokens


@dataclasses.dataclass
class ParamList(Node):
    params: List[Parameter]

    @classmethod
    def parse(cls, tokens: List[Token], source: str) -> Tuple[Self, List[Token]]:
        param_list = []
        while tokens[0].ttype != TokenType.CPAREN:
            param, tokens = Parameter.parse(tokens, source)
            param_list.append(param)
        return cls(param_list), tokens


@dataclasses.dataclass
class FnTypeAnnotation(TypeAnnotation):
    oparen: Token
    param_list: ParamList
    cparen: Token
    arrow: Token
    return_type: TypeAnnotation

    @classmethod
    def parse(cls, tokens: List[Token], source: str) -> Tuple[Self, List[Token]]:
        oparen, tokens = expect(tokens, [TokenType.OPAREN], source)
        param_list, tokens = ParamList.parse(tokens, source)
        cparen, tokens = expect(tokens, [TokenType.CPAREN], source)
        arrow, tokens = expect(tokens, [TokenType.ARROW], source)
        return_type, tokens = TypeAnnotation.parse(tokens, source)

        return cls(oparen, param_list, cparen, arrow, return_type), tokens


@dataclasses.dataclass
class Parameter(Node):
    name: Token
    colon: Token
    annotation: TypeAnnotation
    comma: Optional[Token]

    @classmethod
    def parse(cls, tokens: List[Token], source: str) -> Tuple[Self, List[Token]]:
        name, tokens = expect(tokens, [TokenType.ID], source)
        colon, tokens = expect(tokens, [TokenType.COLON], source)
        annotation, tokens = TypeAnnotation.parse(tokens, source)
        comma = None

        comma_or_cparen, _ = expect(tokens, [TokenType.COMMA, TokenType.CPAREN], source)
        if comma_or_cparen.ttype == TokenType.COMMA:
            comma = comma_or_cparen
            tokens = tokens[1:]

        return cls(name, colon, annotation, comma), tokens


@dataclasses.dataclass
class Block(Node):
    indent: Token
    entries: List[Expression | Statement]
    block_value: Optional[Expression]
    dedent: Token

    @classmethod
    def parse(cls, tokens: List[Token], source: str) -> Tuple[Self, List[Token]]:
        indent, tokens = expect(tokens, [TokenType.INDENT], source)

        entries = []
        while tokens[0].ttype != TokenType.DEDENT:
            if Statement.is_stmt_next(tokens):
                stmt, tokens = Statement.parse(tokens, source)
                entries.append(stmt)
            else:
                expr, tokens = Expression.parse(tokens, source)
                entries.append(expr)

        block_value = None
        if entries and isinstance(entries[-1], Expression):
            block_value = entries[-1]

        dedent, tokens = expect(tokens, [TokenType.DEDENT], source)

        return cls(indent, entries, block_value, dedent), tokens


@dataclasses.dataclass
class VarDeclStmt(Statement):
    let: Token
    name: Token
    colon: Token
    annotation: Optional[TypeAnnotation]
    equals_or_colon: Token
    expr_or_block: Expression | Block

    @classmethod
    def parse(cls, tokens: List[Token], source: str) -> Tuple[Self, List[Token]]:
        let, tokens = expect(tokens, [TokenType.LET], source)
        name, tokens = expect(tokens, [TokenType.ID], source)
        colon, tokens = expect(tokens, [TokenType.COLON], source)

        annotation = None
        if tokens[0].ttype != TokenType.EQUALS:
            annotation, tokens = TypeAnnotation.parse(tokens, source)

        equals_or_colon, tokens = expect(
            tokens, [TokenType.EQUALS, TokenType.COLON], source
        )

        if tokens[0].ttype == TokenType.INDENT:
            expr_or_block, tokens = Block.parse(tokens, source)
        else:
            expr_or_block, tokens = Expression.parse(tokens, source)

        return cls(let, name, colon, annotation, equals_or_colon, expr_or_block), tokens


@dataclasses.dataclass
class AssignStmt(Statement):
    name: Token
    equals: Token
    expr_or_block: Expression | Block

    @classmethod
    def parse(cls, tokens: List[Token], source: str) -> Tuple[Self, List[Token]]:
        name, tokens = expect(tokens, [TokenType.ID], source)
        equals, tokens = expect(tokens, [TokenType.EQUALS], source)
        match tokens[0].ttype:
            case TokenType.INDENT:
                expr_or_block, tokens = Block.parse(tokens, source)
            case _:
                expr_or_block, tokens = Expression.parse(tokens, source)
        return cls(name, equals, expr_or_block), tokens


class Expression(Node):
    @classmethod
    def parse(cls, tokens: List[Token], source: str) -> Tuple[Self, List[Token]]:
        match tokens[0].ttype:
            case TokenType.INT_LITERAL:
                return IntLitExpr.parse(tokens, source)
            case TokenType.STRING_LITERAL:
                return StringLitExpr.parse(tokens, source)
            case TokenType.ID:
                if len(tokens) == 1 or tokens[1].ttype != TokenType.OPAREN:
                    return NameExpr.parse(tokens, source)
                else:
                    return CallExpr.parse(tokens, source)
        error(f"{tokens[0].display(source)}\nunexpected token")


@dataclasses.dataclass
class NameExpr(Expression):
    name: Token

    @classmethod
    def parse(cls, tokens: List[Token], source: str) -> Tuple[Self, List[Token]]:
        name, tokens = expect(tokens, [TokenType.ID], source)
        return cls(name), tokens


@dataclasses.dataclass
class IntLitExpr(Expression):
    literal: Token

    @classmethod
    def parse(cls, tokens: List[Token], source: str) -> Tuple[Self, List[Token]]:
        literal, tokens = expect(tokens, [TokenType.INT_LITERAL], source)
        return cls(literal), tokens


@dataclasses.dataclass
class StringLitExpr(Expression):
    literal: Token

    @classmethod
    def parse(cls, tokens: List[Token], source: str) -> Tuple[Self, List[Token]]:
        literal, tokens = expect(tokens, [TokenType.STRING_LITERAL], source)
        return cls(literal), tokens


@dataclasses.dataclass
class Argument(Node):
    expr: Expression
    comma: Optional[Token]

    @classmethod
    def parse(cls, tokens: List[Token], source: str) -> Tuple[Self, List[Token]]:
        expr, tokens = Expression.parse(tokens, source)
        comma = None

        comma_or_cparen, _ = expect(tokens, [TokenType.COMMA, TokenType.CPAREN], source)
        if comma_or_cparen.ttype == TokenType.COMMA:
            comma = comma_or_cparen
            tokens = tokens[1:]

        return cls(expr, comma), tokens


@dataclasses.dataclass
class ArgList(Node):
    args: List[Argument]

    @classmethod
    def parse(cls, tokens: List[Token], source: str) -> Tuple[Self, List[Token]]:
        args = []
        while tokens[0].ttype != TokenType.CPAREN:
            arg, tokens = Argument.parse(tokens, source)
            args.append(arg)

        return cls(args), tokens


@dataclasses.dataclass
class CallExpr(Expression):
    name: Token
    oparen: Token
    arg_list: ArgList
    cparen: Token

    @classmethod
    def parse(cls, tokens: List[Token], source: str) -> Tuple[Self, List[Token]]:
        name, tokens = expect(tokens, [TokenType.ID], source)
        oparen, tokens = expect(tokens, [TokenType.OPAREN], source)
        arg_list, tokens = ArgList.parse(tokens, source)
        cparen, tokens = expect(tokens, [TokenType.CPAREN], source)

        return cls(name, oparen, arg_list, cparen), tokens


def error(s: str) -> NoReturn:
    print(s)
    exit(1)


def expect(
    tokens: List[Token],
    expected: List[TokenType],
    source: str,
    msg: Optional[str] = None,
) -> Tuple[Token, List[Token]]:
    if not tokens:
        error("unexpected end of input")

    first = tokens[0]
    for exp in expected:
        if first.ttype == exp:
            return first, tokens[1:]

    msg = msg or f"expected {expected}"
    error(f"{first.display(source)}\n{msg}")


class Parser:
    def __init__(self, source: str, tokens: List[Token]):
        self._source = source
        self._tokens = tokens

    def parse(self) -> List[Statement]:
        stmts = []
        while self._tokens:
            stmt, self._tokens = Statement.parse(self._tokens, self._source)
            stmts.append(stmt)

        return stmts


def main():
    with open("test.bs", "r") as f:
        source = f.read()

    lexer = Lexer(source)
    tokens = lexer.tokens()
    parser = Parser(source, tokens)
    stmts = parser.parse()

    for s in stmts:
        s.show(source)


if __name__ == "__main__":
    main()
