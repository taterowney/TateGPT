from typing import Literal
from parser.charsets import ALL_CHARS, ALPHAS, LETTERS, DIGITS, WHITESPACE, CAPITALS, LOWERCASE, PUNCTUATION

from parser.linked_string import linkedString

'''
WHAT IS THIS?

This is a simple parser meant for cleaning and removing specific text patterns from long strings. It was created to process large amounts of text from Wikipedia into a format suitable for training an LLM (by removing HTML tags, references, etc.)

HOW TO USE:

Expressions can be created by combining LiteralExpr and other more complex expressions using the "+" operator:
>>> expr = LiteralExpr("hello") + LiteralExpr("world")    # Matches "helloworld"
>>> expr2 = LiteralExpr("hello") + Anything() + LiteralExpr("world")    # Matches "helloworld", "hello world", "hello, world", etc. ("hello" followed by "world" with any amount of characters in between)

Adding strings to expressions is equivalent to adding LiteralExpr objects:
>>> expr2 = "hello" + Anything() + "world"    # Equivalent to the previous example

Expressions can be repeated a specific number of times using the "*" operator:
>>> expr3 = LiteralExpr("hello") * 3    # Matches "hellohellohello"
>>> expr4 = LiteralExpr("hello") * range(2, 4)    # Matches anywhere between 2 and 4 "hello"s (inclusive) in a row 
>>> expr4 = LiteralExpr("hello")[2:4]    # Equivalent to the previous example
>>> expr5 = LiteralExpr("hello") * ANY_AMOUNT    # Matches any number of "hello"s in a row, including 0
>>> expr6 = (LiteralExpr("hello") * 3 + "world ") * 2    # Matches "hellohellohelloworld hellohellohelloworld "

Many different characters can be matched using the OneOf expression:
>>> expr6 = OneOf(ALL_CHARS)    # Matches any single unicode character
>>> expr7 = "hello" + OneOf(PUNCTUATION)    # Matches "hello" followed by a single punctuation character (e.g. "hello!")
>>> expr8 = OneOf(ALPHAS) * 3    # Matches any 3-letter word
>>> expr9 = OneOf(["hello", "goodbye"])    # Matches either "hello" or "goodbye"

Expressions can be combined using other operators:
>>> expr10 = first_expr | second_expr    # Matches either first_expr or second_expr
>>> expr11 = first_expr & second_expr    # Text must match both first_expr and second_expr
>>> expr12 = ~first_expr    # Text must not match first_expr

Detecting matching delimiters is also possible:
>>> opening, closing = MatchingDelimiters("[[", "]]")
>>> expr13 = opening + Anything() + closing    # Matches anything between "[[" and "]]" inclusive; intelligently determines which closing delimiter matches which opening delimiter

Using expressions:
>>> expr.find("helloworld helloworld")    # Returns the starting and ending index of the first match: (0, 10)
>>> expr.find_all("helloworld helloworld")    # Returns all starting and ending indices of matches: [(0, 10), (11, 21)]
>>> expr.matches("helloworld")    # Determines if the entire string matches the pattern: True in this case

Modifying strings based on expressions:
>>> expr = "hello" + Anything() + "world"
>>> expr.remove_from("hello world!")    # Removes all instances of the pattern: returns "!"
>>> expr.replace_in("hello world!", "goodbye")    # Replaces all instances of the pattern: returns "goodbye!"
>>> expr.replace_in("hello world hello world", lambda x: x.upper())    # Replaces all instances of the pattern with the result of the lambda function: returns "HELLO WORLD HELLO WORLD"

Working with specific parts of the expression:
>>> expr = "hello" + ReturnValue(Anything()) + "world" + ReturnValue(OneOf(PUNCTUATION))
>>> expr.replace_in("hello test world!", lambda anything, punctuation: anything.upper() + punctuation)    # The lambda function is fed with whatever has matched the pattern inside of the ReturnValue functions: returns " TEST !"

TODO:
- Add "beginning of string" and "end of string" expressions
'''



INF = float('inf')
ANY_AMOUNT = slice(0, INF)

def _bad_call(*args, **kwargs):
    raise NotImplementedError("This function should be implemented by a subclass")

class _ReturnValueHandler:
    retvals = None

class Expression:
    def __init__(self, pattern_fxn=_bad_call):
        self.pattern_fxn = pattern_fxn
        self.should_return_values = False

    def find(self, text, start=0, end=-1):
        # Should not be overridden (override _find instead)
        # helps to handle passing information between inductively nested expressions
        opening, closing = self._find(text, start, end)
        if self.should_return_values and _ReturnValueHandler.retvals is not None:
            _ReturnValueHandler.retvals[self] = self.get_retvals(text[opening:closing])
        return opening, closing

    def _find(self, text, start=0, end=-1):
        # Finds the starting and ending index of the first match, such that this match is minimal
        if end == -1:
            end = len(text)
        # try all possible n-grams: very inefficient
        for ngram_size in range(1, end - start + 1):
            for i in range(start, end - ngram_size + 1):
                if self.pattern_fxn(text[i:i+ngram_size]):
                    return i, i + ngram_size
        return -1, -1

    def matches(self, text):
        # Returns true if the entire string matches the pattern (even if it's not minimal)
        # Should not be overridden (override _matches instead)
        res = self._matches(text)
        if self.should_return_values and _ReturnValueHandler.retvals is not None and res:
            _ReturnValueHandler.retvals[self] = text
        return res

    def _matches(self, text):
        # Returns true if the entire string matches the pattern (even if it's not minimal)
        return self.pattern_fxn(text)

    def find_all(self, text, start=0, end=-1, overlap=False):
        # Returns all matches in the text
        if end == -1:
            end = len(text)
        idx = start
        if not overlap:
            while idx < end:
                beginning_of_expr, idx = self.find(text, idx, end)
                if beginning_of_expr == -1:
                    break
                yield beginning_of_expr, idx
                # Handle if there are infinitely many matches (i.e. everything matches)
                if beginning_of_expr == idx:
                    idx += 1
        else:
            beginning_of_expr = -1
            while beginning_of_expr < end:
                beginning_of_expr, idx = self.find(text, beginning_of_expr+1, end)
                if beginning_of_expr != -1:
                    yield beginning_of_expr, idx
                else:
                    break

    def replace_in(self, string, replace_func):
        if type(replace_func) == str:
            replace_with = replace_func
            replace_func = lambda x: replace_with
        offset = 0
        _ReturnValueHandler.retvals = {}
        for start, end in self.find_all(string):
            if len(_ReturnValueHandler.retvals) > 0:
                new_string = replace_func(*(_ReturnValueHandler.retvals.values()))
                _ReturnValueHandler.retvals = {}
            else:
                new_string = replace_func(string[start:end])
            string = string[:start+offset] + new_string + string[end+offset:]
            offset += len(new_string) - (end - start)
        _ReturnValueHandler.retvals = None
        return string

    def get_retvals(self, text):
        # Returns the values of the pattern inside of ReturnValue objects; otherwise, returns the entire string
        return text

    def remove_from(self, string):
        return self.replace_in(string, "")

    def __add__(self, other):
        if isinstance(other, Expression):
            if isinstance(self, SequentialExpression) and isinstance(other, SequentialExpression):
                return SequentialExpression(self.expressions + other.expressions)
            if isinstance(self, SequentialExpression):
                return SequentialExpression(self.expressions + [other])
            if isinstance(other, SequentialExpression):
                return SequentialExpression([self] + other.expressions)
            return SequentialExpression([self, other])
        elif isinstance(other, str):
            return self + LiteralExpr(other)
        raise TypeError(f"can only concatenate Expression or string (not '{type(other).__name__}') to Expression")

    def __radd__(self, other):
        if isinstance(other, str):
            return LiteralExpr(other) + self
        raise TypeError(f"can only concatenate Expression or string (not '{type(other).__name__}') to Expression")

    def __mul__(self, other):
        if isinstance(other, int):
            return RepeatedExpression(self, other, other)
        if isinstance(other, slice) or isinstance(other, range):
            start, stop = other.start, other.stop
            if start is None:
                start = 0
            if stop is None:
                stop = INF
            if start < 0 or stop < start:
                raise ValueError("Cannot have negative occurrences of a pattern")
            return RepeatedExpression(self, start, stop)
        raise TypeError(f"can only multiply Expression by int, slice, or range (not '{type(other).__name__}')")

    def __rmul__(self, other):
        return self * other

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self * item
        raise TypeError(f"can only index Expression by slice (not '{type(item).__name__}')")

    def __or__(self, other):
        if isinstance(other, DisjunctiveExpression):
            return DisjunctiveExpression([self] + other.expressions)
        return DisjunctiveExpression([self, other])

    def __and__(self, other):
        if isinstance(other, ConjunctiveExpression):
            return ConjunctiveExpression([self] + other.expressions)
        return ConjunctiveExpression([self, other])

    def __invert__(self):
        return NegativeExpression(self)

class SequentialExpression(Expression):
    def __init__(self, expressions):
        self.expressions = expressions
        super().__init__()

    def _find(self, text, start=0, end=-1):
        idx = start
        if end == -1:
            end = len(text)
        while idx < end:
            beginning_of_expr, idx = self.expressions[0].find(text, idx, end)
            if beginning_of_expr == -1:
                return -1, -1
            previous_idx = beginning_of_expr
            for i in range(len(self.expressions[1:])):
                pattern = self.expressions[i+1]
                beginning_of_pattern, new_idx = pattern.find(text, idx, end)
                if beginning_of_pattern == -1:
                    break
                if beginning_of_pattern > idx:
                    # go back here and see if the longer string still fits the previous pattern
                    if not self.expressions[i].matches(text[previous_idx:beginning_of_pattern]):
                        break
                previous_idx = idx
                idx = new_idx
                if i == len(self.expressions) - 2:
                    return beginning_of_expr, idx
            if idx == beginning_of_expr:
                idx += 1
        return -1, -1

    def _matches(self, text):
        idx = 0
        beginning_of_pattern = 0
        previous_idx = 0
        for i in range(len(self.expressions)):
            pattern = self.expressions[i]
            beginning_of_pattern, idx = pattern.find(text, idx)
            if beginning_of_pattern == -1:
                return False
            if beginning_of_pattern > previous_idx:
                if not self.expressions[i-1].matches(text[previous_idx:beginning_of_pattern]):
                    return False
            previous_idx = idx
        if not self.expressions[-1].matches(text[beginning_of_pattern:]):
            return False
        return True

class RepeatedExpression(Expression):
    # Allows for the provided expression to repeat a variable number of times
    def __init__(self, expression, min_repeats=0, max_repeats=INF):
        self.expression = expression
        self.min_repeats = min_repeats
        self.max_repeats = max_repeats
        super().__init__()

    def _find(self, text, start=0, end=-1):
        idx = start
        if end == -1:
            end = len(text)
        if self.min_repeats == 0:
            return start, start
        if isinstance(self.expression, LiteralExpr):
            result = text.find(self.expression.literal*self.min_repeats, start, end)
            if result == -1:
                return -1, -1
            return result, result + len(self.expression.literal)*self.min_repeats
        previous_idx = start
        for _ in range(self.min_repeats):
            beginning_of_expr, idx = self.expression.find(text, idx, end)
            if beginning_of_expr == -1:
                return -1, -1
            if beginning_of_expr > previous_idx:
                if not self.expression.matches(text[previous_idx:beginning_of_expr]):
                    return -1, -1
            previous_idx = idx
        return start, idx

    def _matches(self, text):
        idx = 0
        previous_idx = 0
        num_repeats = 0
        while idx < len(text):
            beginning_of_pattern, idx = self.expression.find(text, idx)
            if beginning_of_pattern == -1:
                return False
            if beginning_of_pattern > previous_idx:
                if not self.expression.matches(text[previous_idx:beginning_of_pattern]):
                    return False
            previous_idx = idx
            num_repeats += 1
            if num_repeats > self.max_repeats:
                return False
        return num_repeats >= self.min_repeats

class DisjunctiveExpression(Expression):
    def __init__(self, expressions):
        self.expressions = expressions
        super().__init__()

    def _find(self, text, start=0, end=-1):
        if end == -1:
            end = len(text)
        best = (INF, INF)
        for expression in self.expressions:
            beginning_of_expr, idx = expression.find(text, start, end)
            if beginning_of_expr != -1:
                if beginning_of_expr < best[0]:
                    best = beginning_of_expr, idx
        if best != (INF, INF):
            return best
        return -1, -1

    def _matches(self, text):
        for expression in self.expressions:
            if expression.matches(text):
                return True
        return False

class ConjunctiveExpression(Expression):
    # Only works if selected text matches all expressions at the same time
    # Still need to debug
    def __init__(self, expressions):
        self.expressions = expressions
        super().__init__()

    def _find(self, text, start=0, end=-1):
        # Finds the starting and ending index of the first match, such that this match is minimal but still satisfies all of the expressions
        if end == -1:
            end = len(text)
        idx = start
        while idx < end:
            beginning_of_expr, idx = self.expressions[0].find(text, idx, end)
            if beginning_of_expr == -1:
                return -1, -1
            previous_idx = beginning_of_expr
            for i in range(len(self.expressions[1:])):
                pattern = self.expressions[i+1]
                beginning_of_pattern, new_idx = pattern.find(text, idx, end)
                if beginning_of_pattern == -1:
                    break
                if beginning_of_pattern > idx:
                    # go back here and see if the longer string still fits the previous pattern
                    if not self.expressions[i].matches(text[previous_idx:beginning_of_pattern]):
                        break
                previous_idx = idx
                idx = new_idx
                if i == len(self.expressions) - 2:
                    return beginning_of_expr, idx
            if idx == beginning_of_expr:
                idx += 1
        return -1, -1

    def _matches(self, text):
        idx = 0
        beginning_of_pattern = 0
        previous_idx = 0
        for i in range(len(self.expressions)):
            pattern = self.expressions[i]
            beginning_of_pattern, idx = pattern.find(text, idx)
            if beginning_of_pattern == -1:
                return False
            if beginning_of_pattern > previous_idx:
                if not self.expressions[i-1].matches(text[previous_idx:beginning_of_pattern]):
                    return False
            previous_idx = idx
        if not self.expressions[-1].matches(text[beginning_of_pattern:]):
            return False
        return True

class NegativeExpression(Expression):
    def __init__(self, expression):
        self.expression = expression
        super().__init__()

    def _find(self, text, start=0, end=-1):
        beginning_of_expr, idx = self.expression.find(text, start, end)
        if beginning_of_expr != start:
            return start, start
        return -1, -1

    def _matches(self, text):
        return not self.expression.matches(text)


class LiteralExpr(Expression):
    def __init__(self, literal):
        if type(literal) == str:
            self.literal = literal
        self.literal = literal
        super().__init__()

    def __add__(self, other):
        if isinstance(other, LiteralExpr):
            return LiteralExpr(self.literal + other.literal)
        return super().__add__(other)

    def _find(self, text, start=0, end=-1):
        if end == -1:
            end = len(text)
        idx = text.find(self.literal, start, end)
        if idx == -1:
            return -1, -1
        return idx, idx + len(self.literal)

    def _matches(self, text):
        return text == self.literal

class OneOf(Expression):
    # More efficient version of DisjunctiveExpression, but only works for a list of strings
    def __init__(self, literals):
        self.literals = literals
        if literals in (ALL_CHARS, ):
            self.is_charset = True
        else:
            self.is_charset = False
        super().__init__()

    def _find(self, text, start=0, end=-1):
        if end == -1:
            end = len(text)
        if not self.is_charset:
            for literal in self.literals:
                idx = text.find(literal, start, end)
                if idx != -1:
                    return idx, idx + len(literal)
            return -1, -1
        # If it's just a very large set of single characters, use a more efficient algorithm
        for idx in range(start, end):
            if text[idx] in self.literals:
                return idx, idx + 1
        return -1, -1

    def _matches(self, text):
        return text in self.literals

class _DelimiterTracker:
    # Used to keep track of where the opening delimiter is, so that the closing delimiter can match it
    search_index = None
    opening_delimiter = None
    closing_delimiter = None

class OpeningDelimiter(Expression):
    def __init__(self, delimiter, tracker):
        self.delimiter = delimiter
        self.tracker = tracker
        super().__init__()

    def _find(self, text, start=0, end=-1):
        if end == -1:
            end = len(text)
        idx = text.find(self.delimiter, start, end)
        if idx == -1:
            return -1, -1
        self.tracker.search_index = idx
        return idx, idx + len(self.delimiter)

    def _matches(self, text):
        return text == self.delimiter

class ClosingDelimiter(Expression):
    def __init__(self, delimiter, tracker):
        self.delimiter = delimiter
        self.tracker = tracker
        super().__init__()

    def _find(self, text, start=0, end=-1):
        if self.tracker.search_index == -1:
            return -1, -1
        if end == -1:
            end = len(text)
        start_of_delimiter, end_of_delimiter = self.find_matching_delimiters(text, self.tracker.opening_delimiter, self.delimiter, self.tracker.search_index, end)
        self.tracker.search_index = -1
        if start_of_delimiter == -1:
            return -1, -1
        if start_of_delimiter < start:
            return -1, -1
        return start_of_delimiter, end_of_delimiter

    def find_matching_delimiters(self, s, opening, closing, begin=None, end=None):
        # This stack will store the indices of the opening delimiters
        stack = []

        # Lengths of opening and closing delimiters
        open_len = len(opening)
        close_len = len(closing)

        # Traverse the string using index to access potential delimiters
        if begin is None:
            begin = 0
        if end is None:
            end = len(s)
        i = begin

        max_stack_length = 0

        while i < end:

            # Check for an opening delimiter
            if i + open_len <= len(s) and s[i:i + open_len] == opening:
                # Push the current index to the stack
                stack.append(i)
                max_stack_length = max(max_stack_length, len(stack))
                i += open_len  # Move index past the delimiter
            # Check for a closing delimiter
            elif i + close_len <= len(s) and s[i:i + close_len] == closing:
                if stack:
                    # Pop the index of the last unmatched opening delimiter
                    start = stack.pop()
                    if not stack:
                        # If the stack is empty, we've found the outermost pair
                        return i, i+close_len
                i += close_len  # Move index past the delimiter
            else:
                i += 1  # Move to the next character
        # if the outermost delimiter is not matched, find if there are more inside it that do match; otherwise just return -1, -1
        # if max_stack_length > 1 and len(stack) > 0:
        #     return find_matching_delimiters(s, opening, closing, stack[0] + open_len, end)
        return -1, -1

    def _matches(self, text):
        return text == self.delimiter


def MatchingDelimiters(opening_delimiter="(", closing_delimiter=")"):
    if opening_delimiter == closing_delimiter:
        raise ValueError("Opening and closing delimiters must be different")
    tracker = _DelimiterTracker()
    tracker.opening_delimiter = opening_delimiter
    tracker.closing_delimiter = closing_delimiter
    return OpeningDelimiter(opening_delimiter, tracker), ClosingDelimiter(closing_delimiter, tracker)

def Anything(literals=ALL_CHARS):
    return OneOf(literals) * ANY_AMOUNT

def Or(*expressions):
    return DisjunctiveExpression(expressions)

def And(*expressions):
    return ConjunctiveExpression(expressions)

def Not(expression):
    return NegativeExpression(expression)

def ReturnValue(expression):
    if type(expression) == str:
        expression = LiteralExpr(expression)
    expression.should_return_values = True
    return expression

def find_matching_delimiters(s, opening, closing, begin=None, end=None):
    # This stack will store the indices of the opening delimiters
    stack = []

    # Lengths of opening and closing delimiters
    open_len = len(opening)
    close_len = len(closing)

    # Traverse the string using index to access potential delimiters
    if begin is None:
        begin = 0
    if end is None:
        end = len(s)
    i = begin

    max_stack_length = 0

    while i < end:

        # Check for an opening delimiter
        if i + open_len <= len(s) and s[i:i + open_len] == opening:
            # Push the current index to the stack
            stack.append(i)
            max_stack_length = max(max_stack_length, len(stack))
            i += open_len  # Move index past the delimiter
        # Check for a closing delimiter
        elif i + close_len <= len(s) and s[i:i + close_len] == closing:
            if stack:
                # Pop the index of the last unmatched opening delimiter
                start = stack.pop()
                if not stack:
                    # If the stack is empty, we've found the outermost pair
                    return i, i+close_len
            i += close_len  # Move index past the delimiter
        else:
            i += 1  # Move to the next character
    # if the outermost delimiter is not matched, find if there are more inside it that do match; otherwise just return -1, -1
    # if max_stack_length > 1 and len(stack) > 0:
    #     return find_matching_delimiters(s, opening, closing, stack[0] + open_len, end)
    return -1, -1


if __name__ == '__main__':
    opening, closing = MatchingDelimiters("[[", "]]")
    combined = opening + Anything() + "|" + ReturnValue(Anything()) + closing
    # combined = opening + ReturnValue("hello world") + closing
    test = '[[hello world]] [[This is a link|This is the part we want to keep]] test'
    # print(find_matching_delimiters(test, "[[", "]]", begin=0))
    # TODO: fix this
    print(combined.find(test, start=1))
    print(list(combined.find_all(test, overlap=True)))
    # print(combined.replace_in(test, lambda x: x))
