from _typeshed import OpenBinaryMode
import abc

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
        # self.__private_var = 12323232 this will create another instance method named __private_var for ChildClass
        # All variables which are assigned a value in the class declaration are class variables. 
        # And variables that are assigned values inside methods are instance variables.
    def foo(self):
        # print(self.__private_var)
        print('custom impl')

cde = ChildClass(123, 451)
cde.foo()


cde.class_method()
print(OOP.class_var)