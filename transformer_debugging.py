from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

PADDING_TOKEN = 49999
VOCAB_SIZE = 50000

FEATURES = 256
BATCH_SIZE = 4
MAX_TOKENS = 50


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, nhead: int, vocab_size=VOCAB_SIZE, dim_feedforward: int = 512, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)
        self.generator = nn.Linear(emb_size, vocab_size)
        self.src_tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor):

        # embed and encode the input tokens
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))

        # create masks to prevent the model from caring about padding tokens
        src_padding_mask = (src == PADDING_TOKEN).type(torch.float32)
        tgt_padding_mask = (trg == PADDING_TOKEN).type(torch.float32)

        # create masks to prevent the model peeking at future tokens
        tgt_mask = self.transformer.generate_square_subsequent_mask(trg.shape[1]).to(DEVICE)

        outs = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        return self.generator(outs)


torch.manual_seed(0)

transformer = Seq2SeqTransformer(3, 3, FEATURES, 4, VOCAB_SIZE, FEATURES)


for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PADDING_TOKEN)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


class DebuggingDataset(torch.utils.data.Dataset):
    def __init__(self, num_tokens=MAX_TOKENS):
        self.num_tokens = num_tokens
        self.padding_token = PADDING_TOKEN

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        title_tokens = [1]*self.num_tokens
        # body_tokens = [1, 0]*(self.num_tokens//2) + [1]
        body_tokens = [1, 0, 0, 1]*(self.num_tokens//4) + [1]
        return torch.tensor(title_tokens, dtype=torch.long), torch.tensor(body_tokens[:-1], dtype=torch.long), torch.tensor(body_tokens[1:], dtype=torch.long)



from torch.utils.data import DataLoader

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = DebuggingDataset()
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE)

    for src, tgt, _ in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:, :-1]

        logits = model(src, tgt_input)

        optimizer.zero_grad()

        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


for i in range(20):
    print(train_epoch(transformer, optimizer))


from torch.nn.utils.rnn import pad_sequence

def predict_tokens(model, prompt, context, max_length=10):
    model.eval()
    with torch.no_grad():
        prompt = pad_sequence(torch.tensor(prompt).unsqueeze(0), batch_first=True, padding_value=PADDING_TOKEN).to(DEVICE)
        context = torch.tensor(context).unsqueeze(0).to(DEVICE)

        for _ in range(max_length):
            out = model(prompt, context)
            out = torch.argmax(out, dim=-1)
            context = torch.cat((context, out[:, -1].unsqueeze(0)), dim=1)
            print(context)

# predict_tokens(transformer, [1]*50, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
predict_tokens(transformer, [1]*50, [1, 0, 0, 1, 1, 0])
