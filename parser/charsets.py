class _AllChars:
    def __contains__(self, char):
        if len(char) == 1:
            return True
        return False

    def __iter__(self):
        for i in range(55295):
            yield chr(i)

    def __len__(self):
        return 55295

ALL_CHARS = _AllChars()

ALPHAS = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

LETTERS = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

DIGITS = list('0123456789')

WHITESPACE = list(' \t\n')

CAPITALS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

LOWERCASE = list('abcdefghijklmnopqrstuvwxyz')

PUNCTUATION = list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')