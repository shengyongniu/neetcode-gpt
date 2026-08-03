from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        char_list = sorted(list(set(list(text))))
        stoi, itos = {}, {}
        for index, char in enumerate(char_list):
            stoi[char] = index
            itos[index] = char
        return (stoi, itos)
        

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        encoded = []
        for s in text:
            encoded.append(stoi[s])
        return encoded

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        decoded = ""
        for index in ids:
            decoded += itos[index]
        return decoded
