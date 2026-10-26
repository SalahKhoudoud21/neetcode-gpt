import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        vocab = positive + negative
        unique_words = set()
        for sentence in vocab:
            unique_words = unique_words | set(sentence.split(" ")) # union of two sets (basically adding two sets together)
        unique_words = list(unique_words)
        
        unique_words.sort() # sorting in alphabetical order
        
        word_idx_enc = {word: i+1 for i, word in enumerate(unique_words)}
        
        pos_enc = []
        for sentence in positive:
            sent_enc = []
            for word in sentence.split(" "):
                sent_enc.append(word_idx_enc[word])
            pos_enc.append(torch.tensor(sent_enc))
        
        neg_enc = []
        for sentence in negative:
            sent_enc = []
            for word in sentence.split(" "):
                sent_enc.append(word_idx_enc[word])
            neg_enc.append(torch.tensor(sent_enc))
        
        # neg_enc = torch.squeeze(nn.utils.rnn.pad_sequence(neg_enc, batch_first=True, padding_value=0), dim=0)
        # pos_enc = torch.squeeze(nn.utils.rnn.pad_sequence(pos_enc, batch_first=True, padding_value=0), dim=0)
        output = pos_enc + neg_enc

        output = nn.utils.rnn.pad_sequence(output, batch_first=True, padding_value=0)

        return output.tolist() # turns tensor back to list
        

        
        

        
        
