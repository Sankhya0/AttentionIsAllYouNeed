#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import torch
import torch.nn as nn


# In[2]:


class InputEmbedding(nn.Module):

    def __init__(self, d_model: int , vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return math.sqrt(self.d_model)*self.embedding(x)


# In[3]:


class PositionEmbedding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p = dropout)

        pe = torch.zeros(seq_len, d_model)

        num = torch.arange(0, seq_len).unsqueeze(dim = 1).float()

        # For numerical stability we will first take a log of the term of the denominator mentioned in the paper and then take its exponential
        # Also, we will invert the sign of the power, so that it becomes a numerator * denominator operation which is faster than a division
        # Hence, the -ve sign begore the entire expression
        temp_var = math.log(10000)/d_model
        den = torch.exp(-(torch.arange(0, d_model, 2).float() * temp_var))

        pe[:, 0::2] = torch.sin(num * den)
        pe[:, 1::2] = torch.cos(num * den)

        # Since the input will come in batches, we need to make it so that the 
        #broadcasting will add the positional encodings to all the elements in the input batch.
        pe.unsqueeeze(dim = 0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape(1), :]).requires_grad(False)


# In[4]:


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        # this eps is needed to not make the denominator 0 as we will run into division by 0 errors.
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        # usually doing the mean cancels out the dimension on which it is applied, but we want to keep it
        # Basically, it keeps the output dimension the same as the input dimension
        # Note that the dim = -1 mean that we are working on the actual last dimension, i.e. the embedding here
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        # The alpha and bias are broadcasted
        return self.alpha * ((x - mean) / (std + self.eps)) + self.bias
        


# In[ ]:


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias = True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias = True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x1 = self.Linear1(x)
        #x2 = self.relu(x1)
        #x3 = self.Linear2(x2)
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


# In[8]:


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        assert d_model%h == 0, "d_model is not divisible by h"
        self.d_k = d_model//h
        
        self.w_q = nn.Linear(512,512)
        self.w_k = nn.Linear(512,512)
        self.w_v = nn.Linear(512,512)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    # The advantage of adding this decorator is that we can call this function without creating an instance of the class i.e.
    # it is a general function which can be called by just doing class.func(params) without initializing anything and it will
    # just work as a regular function
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # replace all the value in the attention_scores matrix with -1e9 where mask is 0
            assert attention_scores[2:] == mask.shape, "the mask size does not match the matrix shape"
            attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1) #dim --> [batch, h, seq_len, seq_len]

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # return attention scores as well for visualization purposes
        # dim -->[Batch, h, seq_len, d_k]
        return (attention_scores @ value), attention_scores

    def forward(self, q, k ,v, mask):
        query = self.w_q(q) # dim --> [batch, seq_len, d_model]
        key = self.w_k(k) # dim --> [batch, seq_len, d_model]
        value = self.w_v(v) # dim --> [batch, seq_len, d_model]

        # Now we have to make it so that the embedding (d_model) gets divided into h equal parts. To do this we can use the torch.view
        # [batch, seq_len, d_model] ---> [batch, seq_len, h, d_k] ---> [batch, h, seq_len, d_k]
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).permute(0,2,1,3)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).permute(0,2,1,3)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).permute(0,2,1,3)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # [batch, h, seq_len, d_k] --> [batch, seq_len, h, d_k] --> [batch, seq_len, d_model]
        # contiguous() ensures that the tensor's memory layout is stored in a contiguous block. 
        # This is important when using view() because view() only works on contiguous memory layouts.
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)
        


# In[7]:


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) 
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # this acts as a skip connection
        # the x is presernved and then the sublayer takes the normalized x as its input
        # After the application of the sublayer, the value recieved is added to the original x itself
        return x + self.dropout(sublayer(self.norm(x)))


# In[9]:


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        #It creates a list of 2 ResidualConnection modules.
        # nn.ModuleList is used to properly register these layers as submodules of EncoderBlock.
        # The for _ in range(2) ensures that exactly two instances of ResidualConnection are created.

    def forward(self, x, src_mask):
        # We use lambda here because ResidualConnection expects a function that takes only one argument (x),
        # but self_attention_block requires four arguments (q, k, v, mask).
        # The lambda ensures that x is passed correctly while keeping the other arguments fixed.
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x


# In[10]:


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# In[11]:


class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, 
                 cross_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, 
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):

        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x


# In[12]:


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


# In[13]:


class PorjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # [Batch, seq_size, d_model] ---> [Batch, seq_size, vocab_size]
        return torch.log_softmax(self.proj(x), dim = -1)


# In[14]:


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, 
                 decoder: Decoder, 
                 src_embed: InputEmbedding, 
                 tgt_embed: InputEmbedding, 
                 src_pos: PositionEmbedding, 
                 tgt_pos: PositionEmbedding,
                 projection_layer: PorjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encode(src, src_mask)
    
    def decode(self, tgt, encoder_output, tgt_mask, src_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decode(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)


# In[ ]:


def build_transformer(src_vocab_size: int, 
                      tgt_vocab_size: int, 
                      src_seq_len: int, 
                      tgt_seq_len: int, 
                      d_model: int = 512, 
                      N: int = 6,
                      h: int = 8, 
                      dropout: float = 0.1,
                      d_ff: int = 2048):
    # create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # create the src and tgt positional encoding layers
    src_pos = PositionEmbedding(src_seq_len, dropout)
    tgt_pos = PositionEmbedding(tgt_seq_len, dropout)

    # Create the encoder block N time
    encoder_blocks = []

    for _ in range(N):
        encoder_self_attention_blocks = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_blocks, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder block N time
    decoder_blocks = []

    for _ in range(N):
        decoder_self_attention_blocks = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_blocks = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_blocks, decoder_cross_attention_blocks, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # coompile the encoder and decoder blocks into module lists
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer to go from embedding to vocab
    projection_layer = PorjectionLayer(d_model, tgt_vocab_size)

    # build the transformer instance
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize the parameters with Xavier initialization
    for p in transformer.parameters:
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

