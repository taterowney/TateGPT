from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
import math, tqdm, time, os, glob

from tokenizer import PADDING_TOKEN, VOCAB_SIZE, pad_end, encode, decode

# Gcloud command:
# gcloud compute ssh instance-name

# Screen session
# screen -S python-session
# command; sudo shutdown -h now
# (crtl + a) then d to detach
# To reattach:
# screen -r python-session


# If this is True then the training loops will be cut short so it doesn't go through the entire process
EXPERIMENTING = False
CLOUD = 1
LOCAL = 0
PLATFORM = CLOUD

# MAXIMUM POWER!!!!!
# MAX_TOKENS = 1000
# FEATURES = 2048
# BATCH_SIZE = 16

if PLATFORM == CLOUD:
    MAX_TOKENS = 50
    FEATURES = 256
    BATCH_SIZE = 64
else:
    # Lame version
    MAX_TOKENS = 50
    FEATURES = 256
    BATCH_SIZE = 4


def get_device():
    # return torch.device('cpu')
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
        self.normalize_weights()

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

    def normalize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

def save_model(model, path=f"./models/transformer_model_{int(time.time())}.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path=None):
    if not path:
        path = max(glob.glob("./models/*.pth"), key=os.path.getctime)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


# DATA LOADING + HANDLING

class XMLDataset(torch.utils.data.Dataset):
    def __init__(self, directory, num_tokens=MAX_TOKENS):
        self.num_tokens = num_tokens
        self.padding_token = PADDING_TOKEN
        from lxml import etree
        self.xml_roots = []
        for file in os.listdir(directory):
            if file.endswith(".xml"):
                self.xml_roots.append(etree.parse( os.path.join(directory, file) ).getroot())
        # keeps track of which n-grams are where
        self.indices = []
        for file_num, root in enumerate(self.xml_roots):
            for tag_num, child in enumerate(root):
                # this includes both the context, and the target (which is the context shifted right by 1)
                length = len(child.find('value').text.split(',')) // (num_tokens + 1)
                for chunk in range(length+1):
                    self.indices.append((file_num, tag_num, chunk))
        self.length = len(self.indices)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file_num, tag_num, chunk_num = self.indices[idx]
        root = self.xml_roots[file_num]
        child = root[tag_num]
        body_tokens = list(map(int, child.find('value').text.split(',')))
        chunk = self.get_nth_chunk(body_tokens, chunk_num)
        title_tokens = list(map(int, child.find('key').text.split(',')))
        title_tokens = pad_end(title_tokens, self.num_tokens)
        return torch.tensor(title_tokens, dtype=torch.long), torch.tensor(chunk, dtype=torch.long)

    def get_nth_chunk(self, tokens, n):
        # returns the n-th set of num_tokens+1 tokens
        if len(tokens) >= (n+1)*(self.num_tokens+1):
            return tokens[n*(self.num_tokens+1):(n+1)*(self.num_tokens+1)]
        else:
            return tokens[n*(self.num_tokens+1):] + [self.padding_token] * ((self.num_tokens+1) - len(tokens[n*(self.num_tokens+1):]))

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
        return torch.tensor(title_tokens, dtype=torch.long), torch.tensor(body_tokens, dtype=torch.long)


def get_dataloaders(name="default", train=0.9, validation=0.05, test=0.05, batch_size=BATCH_SIZE, num_tokens=MAX_TOKENS):
    if name == "default":
        ds = XMLDataset('cleaned_data', num_tokens=num_tokens)
        train, validation, test = torch.utils.data.random_split(ds, [int(len(ds)*train), int(len(ds)*validation), len(ds) - int(len(ds)*train) - int(len(ds)*validation)])
        return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True), torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True), torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    elif name == "debug":
        ds = DebuggingDataset(num_tokens=num_tokens)
        train, validation, test = torch.utils.data.random_split(ds, [90, 5, 5])
        return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True), torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True), torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)


def train_epoch(model, optimizer, loss_fn, train_loader):
    model.train()
    losses = 0

    train = XMLDataset('cleaned_data', num_tokens=MAX_TOKENS)
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)

    i = 0
    for src, tgt in tqdm.tqdm(train_loader, desc="Training model"):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        logits = model(src, tgt[:, :-1])

        optimizer.zero_grad()

        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        i += 1
        if EXPERIMENTING and i > 50:
            break

    return losses / i

def evaluate(model, loss_fn, data_loader):
    model.eval()
    losses = 0

    i = 0
    for src, tgt in tqdm.tqdm(data_loader, desc="Evaluating model"):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        logits = model(src, tgt[:, :-1])

        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

        i += 1
        if EXPERIMENTING and i > 5:
            break

    return losses / i


def fit(model, num_epochs=10):
    begin_time = str(int(time.time()))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PADDING_TOKEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    train, validation, test = get_dataloaders("default")
    # train, validation, test = get_dataloaders("debug")
    best_val_loss = float('inf')
    val_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train_epoch(model, optimizer, loss_fn, train)
        val_loss = evaluate(model, loss_fn, validation)
        end_time = time.time()

        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./models/best_model_{begin_time}.pth')

        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Validation loss: {val_loss:.3f}, Epoch time = {end_time - start_time:.3f}s")

    print(val_losses)

def predict_tokens(model, prompt, context, max_length=10):
    model.eval()
    with torch.no_grad():
        prompt = pad_sequence(torch.tensor(prompt).unsqueeze(0), batch_first=True, padding_value=PADDING_TOKEN).to(DEVICE)
        context = torch.tensor(context).unsqueeze(0).to(DEVICE)

        for _ in range(max_length):
            out = model(prompt, context)
            out = torch.argmax(out, dim=-1)
            context = torch.cat((context, out[:, -1].unsqueeze(0)), dim=1)
            if out[:, -1] == PADDING_TOKEN:
                break
    return context

def get_num_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

def get_model():
    return Seq2SeqTransformer(num_encoder_layers=3, num_decoder_layers=3, emb_size=FEATURES, nhead=8, dim_feedforward=FEATURES).to(DEVICE)

def start_training(epochs=20):
    model = get_model()
    fit(model, num_epochs=epochs)
    return model

def keep_training(epochs=20):
    model = get_model()
    model = load_model(model)
    fit(model, num_epochs=epochs)
    return model

def memory_usage_test():
    global EXPERIMENTING
    EXPERIMENTING = True
    if DEVICE == torch.device('cuda'):
        print("\n\n")
        torch.cuda.memory._record_memory_history()
        start_training(1)
        print(torch.cuda.memory_summary(DEVICE))
        print(torch.cuda.memory._dump_snapshot("snapshot.pickle"))


def example_prediction():
    model = get_model()
    model = load_model(model)
    tokens = predict_tokens(model, [50257,2025,998,1042,50256], [50257,7061,6,2025,998,1042,7061,6,318,257,1964,8876,290,3356,326,318,1028,477])
    print(tokens)
    print(decode(tokens[0].tolist()))

def predict_text(prompt, context):
    model = get_model()
    model = load_model(model)
    prompt = encode(prompt)
    context = encode(context)[:-1]
    tokens = predict_tokens(model, prompt, context)
    return decode(tokens[0].tolist())

if __name__ == '__main__':
    start_training(epochs=8)
    # memory_usage_test()
