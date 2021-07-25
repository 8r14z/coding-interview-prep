# https://leetcode.com/problems/alien-dictionary/

class Solution:
    def alienOrder(self, words: List[str]) -> str:

        adjencency_list = defaultdict(set)
        in_degree = {c: 0 for word in words for c in word}
    
        for first_word, second_word in zip(words, words[1:]):
            for c, d in zip(first_word, second_word):
                if c != d: 
                    if d not in adjencency_list[c]:
                        adjencency_list[c].add(d)
                        in_degree[d] += 1
                    break
            else:
                if len(second_word) < len(first_word): 
                    return ""
            
        visited = set()
        queue = deque([c for c in in_degree if in_degree[c] == 0])
        
        output = []
        
        while queue:
            node = queue.popleft()
            output.append(node)
            visited.add(node)
            for child in adjencency_list[node]:
                in_degree[child] -= 1
                if not in_degree[child]:
                    queue.append(child)
        
        if len(output) < len(in_degree):
            return ""
    
        return ''.join(output)