from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        from collections import Counter
        char_list = list(corpus)
        merges = []
        for _ in range(num_merges):
            if len(char_list) < 2:
                break
            freq_counter = Counter()
            for i in range(len(char_list)-1):
                freq_counter[(char_list[i],char_list[i+1])] += 1
            # most_common_list = freq_counter.most_common()
            # most_common_list.sort(key = lambda x: (-x[1], x[0])))
            # most_common_pair = most_common_list[0][0]
            most_common_pair = min(freq_counter, key = lambda pair: (-freq_counter[pair], pair))
            merges.append([most_common_pair[0], most_common_pair[1]])
            i = 0
            while i < len(char_list)-1:
                if (char_list[i], char_list[i+1]) == most_common_pair:
                    char_list[i:i+1+1] = ["".join(char_list[i:i+1+1])]
                i += 1

        return merges
