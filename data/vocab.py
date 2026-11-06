from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        vocab = list(set(text))
        vocab.sort() # sort alphabetically
        
        # string to integer
        stoi = {char: i for i, char in enumerate(vocab)}
        
        # integer to string
        itos = {i: char for i, char in enumerate(vocab)}

        return (stoi, itos)
        
        

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        
        int_arr = [stoi[char] for char in text]

        return int_arr

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        
        text = ''.join([itos[i] for i in ids])

        return text
