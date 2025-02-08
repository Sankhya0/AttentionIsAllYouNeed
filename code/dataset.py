#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset

from typing import Any


# ## Stucture of Dataset
# 
# Note that we are working with the Huggingface en-it opus books dataset.
# 
# 1. In this item has an "id" and a "translation"
# 2. The id is just an int (index)
# 3. The translation is a dictionary in itself
# 4. In our case, the src_lang = "en" and tgt_lang = "it" which will be the keys of the "translation" dictionary
# 5. Therefore, we can use `[translation][src_lang]` to see the text in the ds item.

# In[ ]:


class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:
        # Get the dataset item
        src_target_pair = self.ds[index]

        # Get the source and target text from the item
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # src words --> tokens --> id of the tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # getting the number of padding tokens required
        # note that we are doing -2 to the encoder tokens because the sos and eos have to be accounted for
        # but in the decoder the EOS doesnt have to be there as it would be predicted by the model so we only do a -1
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # If the padding is less than 0, then there is a problem
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Initialize tokens as 1D tensors
        sos_token = torch.tensor(self.tokenizer_src.token_to_id('[SOS]'), dtype=torch.int64)
        eos_token = torch.tensor(self.tokenizer_src.token_to_id('[EOS]'), dtype=torch.int64)
        pad_token = torch.tensor(self.tokenizer_src.token_to_id('[PAD]'), dtype=torch.int64)

        # Add SOS, EOS and padding tokens to the source text
        encoder_input = torch.cat([
            sos_token.unsqueeze(0),  # Ensure it's a 1D tensor
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            eos_token.unsqueeze(0),  # Ensure it's a 1D tensor
            torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ])

        # Add SOS token to the decoder input
        decoder_input = torch.cat([
            sos_token.unsqueeze(0),  # Ensure it's a 1D tensor
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        # Add EOD token to the label (Expected in the output)
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            eos_token.unsqueeze(0),  # Ensure it's a 1D tensor
            torch.tensor([pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        assert encoder_input.shape[0] == self.seq_len
        assert decoder_input.shape[0] == self.seq_len
        assert label.shape[0] == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int(),  # [1, 1, seq_len]
            "decoder_mask": (decoder_input != pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # [1,1,seq_len] & [1,seq_len]
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }


# In[6]:


def causal_mask(size):
    # Keeps only the upper triangular part, setting elements above the main diagonal to 1, while the rest remains 0.
    mask = torch.triu(torch.ones(1,size,size), diagonal = 1).int()

    # we want the opposite, i.e. above the diagonal to be 0
    return mask == 0

