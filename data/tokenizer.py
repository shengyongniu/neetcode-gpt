from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        from collections import Counter

        char_list = list(corpus)
        merges = []

        for _ in range(num_merges):
            if len(char_list) < 2:
                break

            freq_counter = Counter()
            for i in range(len(char_list) - 1):
                freq_counter[(char_list[i], char_list[i + 1])] += 1

            # Highest frequency, lexicographically smallest on ties
            most_common_pair = min(
                freq_counter,
                key=lambda pair: (-freq_counter[pair], pair)
            )

            merges.append([
                most_common_pair[0],
                most_common_pair[1]
            ])

            new_char_list = []
            i = 0

            while i < len(char_list):
                if (
                    i + 1 < len(char_list)
                    and (char_list[i], char_list[i + 1]) == most_common_pair
                ):
                    new_char_list.append(
                        char_list[i] + char_list[i + 1]
                    )
                    i += 2
                else:
                    new_char_list.append(char_list[i])
                    i += 1

            char_list = new_char_list

        return merges