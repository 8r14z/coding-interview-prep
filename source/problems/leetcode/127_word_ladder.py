# https://leetcode.com/problems/word-ladder/description/

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordSet = set(wordList)
        result = 0

        queue = [beginWord]
        visited = set(beginWord)

        while queue:
            newQueue = []
            result += 1
            for word in queue:
                if word == endWord:
                    return result

                for i in range(len(word)):
                    for j in range(26):
                        nextWord = word[:i] + chr(97 + j) + word[i+1:]
                        if nextWord in wordSet and nextWord not in visited:
                            visited.add(nextWord)
                            newQueue.append(nextWord)
            
            queue = newQueue

        return 0