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
        vocab_set = set()
        for sentence in positive + negative:
            for w in sentence.split():
                vocab_set.add(w)
        vocab_sort = sorted(list(vocab_set))
        vocab_dict = {}
        for index, v in enumerate(vocab_sort):
            vocab_dict[v] = index + 1
        
        # print(vocab_dict)
        ts_list = []
        for sentence in positive + negative:
            encode = []
            for w in sentence.split():
                encode.append(vocab_dict[w])
            ts_list.append(torch.tensor(encode))
        tensors = nn.utils.rnn.pad_sequence(ts_list, batch_first=True)
        return tensors


        
