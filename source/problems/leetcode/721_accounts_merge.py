# https://leetcode.com/problems/accounts-merge/

class Solution:
    def accountsMerge(self, accounts): 
        email_to_name = {}
        graph = defaultdict(set)
        
        for account in accounts:
            if len(account) < 2: continue
            name = account[0]
            for email in account[1:]:
                graph[email].add(account[1])
                graph[account[1]].add(email)
                email_to_name[email] = name
        
        seen_emails = set()
        ans = []

        for email in graph:
            if email in seen_emails: 
                continue
            seen_emails.add(email)
            stack = [email]
            emails = [email]
            
            while stack:
                node = stack.pop()
                for child in graph[node]:
                    if child not in seen_emails:
                        stack.append(child)
                        emails.append(child)
                        seen_emails.add(child)
            
            emails.sort()
            ans.append([email_to_name[email]] + emails)
            
        return ans
                    
        