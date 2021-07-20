# https://leetcode.com/problems/robot-bounded-in-circle/

UP = 0
LEFT = 1
DOWN = 2
RIGHT = 3
        
class Solution:
    def get_direction(self, direction, instruction) -> int:
        if instruction not in ['L', 'R']:
            return direction
            
        turn_left = instruction == 'L'
        direction_map = {
            UP: [LEFT, RIGHT],
            DOWN: [RIGHT, LEFT],
            LEFT: [DOWN, UP],
            RIGHT: [UP, DOWN]
        }
        
        return direction_map[direction][0] if turn_left else direction_map[direction][1] 
            
    def move(self, x, y, direction):
        move_map = {
            UP: [0, 1],
            DOWN: [0,-1],
            LEFT: [-1, 0],
            RIGHT: [1, 0],
        }
        
        return x + move_map[direction][0], y + move_map[direction][1]
        
    def isRobotBounded(self, instructions: str) -> bool:
        x = y = direction = 0
    
        for instruction in instructions:
                if instruction == 'L' or instruction == 'R':
                    direction = self.get_direction(direction, instruction)
                else:
                    x, y = self.move(x, y, direction)
        
        return (x == 0 and y == 0) or direction != UP 
        # at the end robot either move back to the origin or change direction
        