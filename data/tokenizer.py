from typing import List
from collections import Counter
class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        chars = list(corpus)
        output = []
        for _ in range(num_merges):
            left = 0
            combos = []
            merged = []
            for right in range(1, len(chars)):
                combos.append((chars[left], chars[right]))
                left += 1
            
            # count frequency of all adjacent token pairs
            freq = Counter(combos) 
            
            # Sort by highest frequency first, breaking ties lexicographically (ascending)
            high_lo = sorted(freq.keys(), key=lambda x: (-freq[x], x))
            output.append([high_lo[0][0], high_lo[0][1]]) # adding merge to list of merges
            most_frequent_pair = ''.join(high_lo[0])
            new_chars = []
            
            # Merge all non-overlapping occurrences left to right
            i = 0
            while i < len(chars):
                if i + 1 < len(chars) and chars[i] + chars[i+1] == most_frequent_pair:
                    new_chars.append(chars[i] + chars[i+1])
                    i += 2
                else:
                    new_chars += [chars[i]]
                    i += 1
                    
            chars = new_chars
      
        return output

            

