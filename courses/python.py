# from _typeshed import OpenBinaryMode
# import abc

#  In Python, destructors are not needed as much needed in C++ 
# because Python has a garbage collector that handles memory management automatically.
class OOP:
    class_var = 1 # once this is updated by self.class_var = something. it will become instance var

    def __init__(self, test: int):
        self.instance_var = test
        self.__private_var = 123
        self._protected_var = 1232

    # The variables defined within __init__() are called as the instance variables or objects.
    
    # It is called when all references to the object have been deleted 
    # i.e when an object is garbage collected.
    def __del__(self):
        print('destroyed')

    # equal, ==
    def __eq__(self, other):
        pass

    # not equal. !=
    def __ne__(self, other):
        pass

    # less than, <
    def __lt__(self, other):
        pass

    # less than or equal to, <=
    def __le__(self, other):
        pass
    
    # greater than, >
    def __gt__(self, other):
        pass 

    # greater than or equal to, >=
    def __ge__(self, other):
        pass

    def foo(self):
        print(self.instance_var)    

    @classmethod
    def class_method(self):
        OOP.class_var = 123

    @staticmethod
    def static_method():
        pass

class ChildClass(OOP):
    def __init__(self, test: int, additional_param: int):
        super().__init__(test)
        self.additional_param = additional_param
        self._protected_var = 11
        # self.__private_var = 12323232 this will create another instance method named __private_var for ChildClass
        # All variables which are assigned a value in the class declaration are class variables. 
        # And variables that are assigned values inside methods are instance variables.
    def foo(self):
        # print(self.__private_var)
        print('custom impl')

cde = ChildClass(123, 451)
cde.foo()
cde._protected_var = 10
cde.__private_var = 11
print('----')
print(cde)
print('----')

cde.class_method()
print(OOP.class_var)

# Data structures
array = []
array.append(1)
array.pop()

set = set()
set.add(1)

dictionary = {}
dictionary['key'] = 'value'

for item in array:
    print(item)

from collections import deque
queue = deque([10])
queue.popleft()

range(4) # [0, 4)

# Heap
import heapq
heap_array = [10]
heapq.heapify(heap_array)
min_value = heapq.heappop(heap_array)
heapq.heappush(heap_array, 1)

x = 4^2 # x = 4^2

import bisect
sorted_array = []
search_value = 1
bisect.bisect(sorted_array, search_value) # left <= i < right, same as bisect_right ==> less than any value on right
bisect.bisect_left(sorted_array, search_value) # left < i <= right => greater any value on left


# Python 3.11 https://docs.python.org/3/library/stdtypes.html 

c = ord('C') - ord('A') # 2

array = []
for item in array:
    pass
for i, item in enumerate(array):
    pass

dict = {}
for key in dict:
    pass
for key,value in dict.items():
    pass

# Naming convention
# https://peps.python.org/pep-0008/#prescriptive-naming-conventions

