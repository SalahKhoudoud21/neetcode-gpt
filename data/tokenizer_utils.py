from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        
        output = [] # shows how each number gets split according to our token vocab
        for number in numbers:
            number = str(number)
            num_to_tokens = []
            i = 0 # left ptr
            while i < len(number):
                val_token = ''
                for j in range(i+1, len(number)+1):
                    curr_token = number[i:j]
                    if curr_token in vocab:
                        val_token = curr_token # keeps track of longest found valid token so far
                
                if not val_token:
                    num_to_tokens.append("UNK")
                    i += 1
                    continue
                
                num_to_tokens.append(val_token)
                i += len(val_token)
            
            output.append(num_to_tokens)
        
        return output
                
            


    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        count = 0
        new_text = text.split()
        i = 0
        while i < len(text):
            val_token = ''
            for j in range(i+1, len(text) + 1):
                curr_token = text[i:j]
                if curr_token in vocab:
                    val_token = curr_token
                
            if val_token:
                count += 1
                i += len(val_token)
                continue
            i += 1
        return count

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        
        num_words = len(text.split()) # without spaces
        token_count = self.count_tokens(text, vocab)
        return round(token_count/num_words, 4)
