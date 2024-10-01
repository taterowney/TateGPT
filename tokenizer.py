from lxml.html.diff import token
from tiktoken_ext.openai_public import cl100k_base


def get_tokenizer():
    import tiktoken
    cl100k_base = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.Encoding(
        # If you're changing the set of special tokens, make sure to use a different name
        # It should be clear from the name what behaviour to expect.
        name="cl100k_im",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            "<|startoftext|>": 100261,
            "<|padding|>": 100256,
        }
    )
    return enc

# INDEX TOKENIZER
# tokenizer = get_tokenizer()
# unused = []
# for i in range(tokenizer.max_token_value):
#     try:
#         tokenizer.decode([i])
#     except:
#         unused.append(i)
# print(f"vocab_size = {tokenizer.max_token_value}")
# print(f"PADDING_TOKEN = {unused[0]}")
# print(f"UNUSED_TOKENS = {unused[1:]}")


PADDING_TOKEN = 100256
START_OF_TEXT = 100261
END_OF_TEXT = 100257
UNUSED_TOKENS = [100262, 100263, 100264, 100265, 100266, 100267, 100268, 100269, 100270, 100271, 100272, 100273, 100274, 100275, 100277]
# START_OF_TEXT = -1
# END_OF_TEXT = -2
# PADDING_TOKEN = -3
# UNUSED_TOKENS = [100256, 100257, 100261, 100262, 100263, 100264, 100265, 100266, 100267, 100268, 100269, 100270, 100271, 100272, 100273, 100274, 100275]

VOCAB_SIZE = 100278


seq_len = 50 # amount of context tokens used

TOKENIZER = get_tokenizer()

def decode(tokens):
    global TOKENIZER
    if not TOKENIZER:
        TOKENIZER = get_tokenizer()
    tokens = [i for i in tokens if i not in UNUSED_TOKENS+ [PADDING_TOKEN, START_OF_TEXT, END_OF_TEXT]]
    return TOKENIZER.decode(tokens)

def encode(text):
    global TOKENIZER
    if not TOKENIZER:
        TOKENIZER = get_tokenizer()
    return [START_OF_TEXT] + TOKENIZER.encode(text) + [END_OF_TEXT]


def pad_start(tokens, new_len):
    if len(tokens) > new_len:
        return tokens[-new_len:]
    return [PADDING_TOKEN] * (new_len - len(tokens)) + tokens

def pad_end(tokens, new_len):
    if type(tokens) == str:
        length = tokens.count(",") + 1
        if length > new_len:
            return ','.join(tokens.split(",")[:new_len])
        return tokens + ("," + str(PADDING_TOKEN)) * (new_len - length)

    if len(tokens) > new_len:
        return tokens[:new_len]
    return tokens + [PADDING_TOKEN] * (new_len - len(tokens))

if __name__ == '__main__':
    test = "1,2,3,4,5,6,7,8,9,10"
    print(pad_end(test, 15))
