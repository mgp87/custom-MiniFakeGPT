import streamlit as st
import torch
from torch.nn import functional as F
import torch.nn as nn
import time
import numpy as np
import torch.optim as optim

st.title("FakeGPT")

uploaded_file = st.file_uploader("Choose a text file", type="txt")

def clean_text(text):
    cleaned_text = text.decode("utf-8").replace('"', '')
    return cleaned_text

if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type, "FileSize":uploaded_file.size}
    st.write(file_details)
    file_content = uploaded_file.getvalue()
    text = clean_text(file_content)
    st.text_area("File Content", text[:1000], height=200)

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    batch_size = 32
    block_size = 32
    max_n_iters = 30000
    evaluation_interval = 100
    evaluation_iterations = 200
    learning_rate = 5e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Parameters:
    n_embeddings = 64
    n_head = 4
    n_layers = 4
    dropout = 0.1

    char_to_int = {char:i for i, char in enumerate(chars)}
    int_to_char = {i:char for i, char in enumerate(chars)}

    encode = lambda s: [char_to_int[c] for c in s]
    decode = lambda l: ''.join([int_to_char[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long, device=device)
    train_split = int(0.9 * len(data)) # 90% of data will be used for training
    train_data = data[:train_split]
    validation_data = data[train_split:] 

    def get_batch_data(dataset_split):
        data = train_data if dataset_split == 'train' else validation_data
        ix = torch.randint(len(data) - block_size, (batch_size, ))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y

    class AttentionHead(nn.Module):
        def __init__(self, head_size, dropout=0.1):
            super(AttentionHead, self).__init__()
            self.dropout = nn.Dropout(dropout)
            # Inputs k, v, q --> refer to attention is all you need paper
            self.key = nn.Linear(n_embeddings, head_size, bias=False)
            self.value = nn.Linear(n_embeddings, head_size, bias=False)
            self.query = nn.Linear(n_embeddings, head_size, bias=False)

            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        def forward(self, x):
            key = self.key(x)
            query = self.query(x)

            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(query.size(-1))
    
            attention_scores = attention_scores.masked_fill(self.tril[:query.size(1), :query.size(1)] == 0, float("-inf"))
            attention_scores = F.softmax(attention_scores, dim=-1) # along the last dimension
            attention_scores = self.dropout(attention_scores)
            value = self.value(x)
            return torch.matmul(attention_scores, value), attention_scores

    class MultiHeadAttention(nn.Module):
        # n_head --> how many attention heads we want to create
        def __init__(self, n_head, head_size, dropout=0.1):
            super(MultiHeadAttention, self).__init__()
            self.heads = nn.ModuleList([AttentionHead(head_size) for i in range(n_head)])
            self.projection = nn.Linear(n_embeddings, n_embeddings)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # Concat all our heads
            output = torch.cat([h(x)[0] for h in self.heads], dim=-1)
            output = self.projection(output)
            return self.dropout(output)

    class FeedForward(nn.Module):
        # d_model --> number of neurons
        def __init__(self, d_model, dropout=0.1):
            super(FeedForward, self).__init__()
            self.linear_1 = nn.Linear(d_model, 6*d_model)
            self.dropout = nn.Dropout(dropout)
            self.linear_2 = nn.Linear(6*d_model, d_model)

        def forward(self, x):
            x = self.linear_1(x)
            x = F.relu(x)
            x = self.linear_2(x)
            return self.dropout(x)

    class LayerNormalization(nn.Module):
        def __init__(self, d_model, epsilon=1e-5):
            super(LayerNormalization, self).__init__()
            self.gamma = nn.Parameter(torch.ones(d_model))
            self.beta = nn.Parameter(torch.zeros(d_model))
            self.epsilon = epsilon

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True) # x is Pytorch Tensor so can call mean method straight
            std = x.std(dim=-1, keepdim=True) # Standard Deviation
            x = (x - mean) / (std + self.epsilon)
            self.gamma * x * self.beta
            return x

    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_head, dropout=0.1):
            super(TransformerBlock, self).__init__()
            head_size = n_embeddings // n_head
            self.multi_head_attention = MultiHeadAttention(n_head, head_size, dropout)
            self.feed_forward = FeedForward(d_model, dropout)
            self.layer_normalization_1 = LayerNormalization(d_model)
            self.layer_normalization_2 = LayerNormalization(d_model)
            self.dropout_1 = nn.Dropout(dropout)
            self.dropout_2 = nn.Dropout(dropout)
  
        def forward(self, x):
            x_2 = self.layer_normalization_1(x)
            x_2 = self.multi_head_attention(x_2)
            x = x + x_2
            x_2 = self.layer_normalization_2(x)
            x_2 = self.feed_forward(x_2)
            x = x + x_2
            return x

    class MiniFakeGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
            self.position_embedding_table = nn.Embedding(block_size, n_embeddings)
            self.transformer_blocks = nn.Sequential(
                *[TransformerBlock(n_embeddings, n_head, dropout) for i in range(n_layers)]
            )
            self.layer_normalization_forward = nn.LayerNorm(n_embeddings) # Using pytorch implementation instead of ours
            self.linear_mapping_head = nn.Linear(n_embeddings, vocab_size) # linear mapping to attention head

        def forward(self, index, targets=None): # Targets are labels, desired outputs
            B, T = index.shape # Batch and Transformer blocks
            token_embedding = self.token_embedding_table(index)
            positional_embedding = self.position_embedding_table(torch.arange(T, device=device)) # Unique position for each individual input
            x = token_embedding + positional_embedding
            x = self.transformer_blocks(x)
            x = self.layer_normalization_forward(x)
            logits = self.linear_mapping_head(x) # predictions

            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape # dimensions of output based on batches, input size
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets) # Cross Entropy as loss function
                return logits, loss

        def generate(self, index, max_tokens): # max num of token to generate based on input
            for i in range(max_tokens): # generate tokens until max_tokens is reached
                index_state = index[:, -block_size:]
                logits, loss = self(index_state) # forward pass
                logits = logits[:, -1, :] # Focus on last timestep in the sequence
                probabilities = F.softmax(logits, dim=-1) # distribution: probability of next token
                index_next = torch.multinomial(probabilities, num_samples=1)
                index = torch.cat((index, index_next), dim=1)
            return index

    @torch.no_grad() # Not to calculate gradients for optimization purposes
    def estimate_loss(model):
        output = {}
        model.eval()

        for split in ['train', 'eval']:
            losses = torch.zeros(evaluation_iterations)
            for i in range(evaluation_iterations):
                x, y = get_batch_data(split)
                logits, loss = model(x, y)
                losses[i] = loss.item()
            output[split] = losses.mean()
        model.train()
        return output

    def train_model():
        model = MiniFakeGPT()
        m = model.to(device)
        print(sum(p.numel() for p in m.parameters())/1e6)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        progress_bar = st.progress(0)
        train_info = st.empty()

        for iteration in range(max_n_iters):
            if iteration % evaluation_interval == 0 or iteration == max_n_iters - 1:
                losses = estimate_loss(model)
                train_info.write(f"Train Loss: {losses['train']:.2f} | Eval Loss: {losses['eval']:.2f}")
                st.write(f"Train Loss: {losses['train']:.2f} | Eval Loss: {losses['eval']:.2f}")

            progress_bar.progress(iteration / max_n_iters)
            x, y = get_batch_data('train')

            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), 'model.pth')

    def generate_text():
        model = MiniFakeGPT()
        model.load_state_dict(torch.load('model.pth'))
        model = model.to(device)

        input_char = torch.zeros((1, 1), dtype=torch.long, device = device)
        output_text = decode(model.generate(input_char, max_tokens=2000)[0].tolist())

    if st.button("Train Model"):
        train_model()

    if st.button("Generate Text"):
        gen_text = generate_text()
        st.text_area("Generated Text", gen_text, height=200)
