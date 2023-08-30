// https://leetcode.com/problems/evaluate-reverse-polish-notation/

class Solution {
    func evalRPN(_ tokens: [String]) -> Int {
        var stack: [Int] = []
        for token in tokens {
            switch token {
                case "+":
                    let a = stack.popLast()!
                    let b = stack.popLast()!
                    stack.append(a+b)
                case "-":
                    let a = stack.popLast()!
                    let b = stack.popLast()!
                    stack.append(b-a)
                case "*":
                    let a = stack.popLast()!
                    let b = stack.popLast()!
                    stack.append(a*b)
                case "/":
                    let a = stack.popLast()!
                    let b = stack.popLast()!
                    stack.append(b/a)
                default:
                    stack.append(Int(token)!)
            }
        }

        assert(stack.count == 1)
        return stack[0]
    }
}