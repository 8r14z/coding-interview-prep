
# Enter your solution here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict, defaultdict
        
class CheckoutManager:
    def __init__(self):
        # maintain queues for lines with each queue is a OrdederedDict of customer of their current item count
        self._queues = defaultdict(OrderedDict)
        self._customer_total_count = {}
        self._customer_lines = {}
        
    def customer_enter(self, customer_id, line_number, num_items):
        self._queues[line_number][customer_id] = num_items
        self._customer_total_count[customer_id] = num_items
        self._customer_lines[customer_id] = line_number
        
    def basket_change(self, customer_id, new_num_items):
        customer_total_items = self._customer_total_count[customer_id]
        diff = new_num_items - customer_total_items
        
        # no changes
        if diff == 0:
            return
        
        self._customer_total_count[customer_id] = new_num_items
        
        customer_line = self._customer_lines[customer_id]
        customer_queue = self._queues[customer_line]
        
        customer_queue[customer_id] = customer_queue[customer_id] + diff
        if customer_queue[customer_id] <= 0:
            del customer_queue[customer_id]
            self._on_customer_leave(customer_id)
        elif diff > 0:
            customer_queue.move_to_end(customer_id)
    
    def line_service(self, line_number, num_processed_items):
        queue = self._queues[line_number]
        
        if len(queue) == 0:
            return
            
        self._hande_line_updates(line_number, num_processed_items)
    
    def lines_service(self):
        # alternatively, we can maintain a sorted list of lines
        # however, it's essentially the trade-off between when we want to sort the array
        # either when adding new line (customer_enter) or when servicing all lines.
        # pratically, there are finite number of lines, so this op should be significant
        sorted_lines = sorted(self._queues.keys())
        for line in sorted_lines:
            self.line_service(line, 1)
        
    def _hande_line_updates(self, line_number, num_processed_items):
        # to improve speed, we can maintain a prefix sum of items in a queue and do binary search to find the index of customer we can jump to 
        # so complexity of this op can be down to O(1), however it will make `basket_change` O(n) to re-calc the prefix sum array.
        if num_processed_items <= 0:
            return
            
        queue = self._queues[line_number]
        removed_customers = []
        
        for customer_id, customer_num_items in queue.items():
            if num_processed_items == customer_num_items:
                removed_customers.append(customer_id)
                break
            elif num_processed_items > customer_num_items:
                num_processed_items -= customer_num_items
                removed_customers.append(customer_id)
            else:
                queue[customer_id] = customer_num_items - num_processed_items
                break
        
        for customer in removed_customers:
            queue.pop(customer)
            self._on_customer_leave(customer)
        
    def _on_customer_leave(self, customer_id):
        del self._customer_total_count[customer_id]
        del self._customer_lines[customer_id]
        print(customer_id)

if __name__ == "__main__":
    import sys

    checkout_manager = CheckoutManager()
    line = sys.stdin.readline().split()
    n = int(line[0])
    for _ in range(n):
        line = sys.stdin.readline().split()
        if line[0] == "CustomerEnter":
            customer_id = int(line[1])
            line_number = int(line[2])
            num_items = int(line[3])
            checkout_manager.customer_enter(customer_id, line_number, num_items)
            
        elif line[0] == "BasketChange":
            customer_id = int(line[1])
            new_num_items = int(line[2])
            checkout_manager.basket_change(customer_id, new_num_items)
            
        elif line[0] == "LineService":
            line_number = int(line[1])
            num_processed_items = int(line[2])
            checkout_manager.line_service(line_number, num_processed_items)
            
        elif line[0] == "LinesService":
            checkout_manager.lines_service()
        else:
            raise Exception("Malformed input!")