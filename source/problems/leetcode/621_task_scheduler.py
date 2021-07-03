# https://leetcode.com/problems/task-scheduler/

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        task_frequencies = sorted(list(Counter(tasks).values()))
        remaining_tasks = [f*-1 for f in task_frequencies]
        heapq.heapify(remaining_tasks)
        
        task_count = 0
        while remaining_tasks:
            scheduled_tasks = []
            for _ in range(n+1):
                if remaining_tasks:
                    task = heapq.heappop(remaining_tasks)
                    scheduled_tasks.append(task)
            
            for task in scheduled_tasks:
                if task + 1 != 0:
                    heapq.heappush(remaining_tasks, task+1)
            
            task_count += (n+1 if remaining_tasks else len(scheduled_tasks))
        
        return task_count
        