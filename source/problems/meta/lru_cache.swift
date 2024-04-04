class Node {
    let value: Int
    let key: Int
    var next: Node?
    var prev: Node?
    init(_ key: Int, _ value: Int) {
        self.key = key
        self.value = value
    }
}

class LRUCache {
    let capacity: Int
    
    let head = Node(-1,-1)
    let tail = Node(-1,-1)
    var keyValueMap: [Int: Node] = [:]

    init(_ capacity: Int) {
        self.capacity = capacity
        head.next = tail
        tail.prev = head
    }
    
    func get(_ key: Int) -> Int {
        guard keyValueMap[key] != nil else {
            return -1
        }

        let node = keyValueMap[key]!
        remove(node)
        enqueue(node)
        return node.value
    }
    
    func put(_ key: Int, _ value: Int) {
        if keyValueMap[key] != nil {
            let node = keyValueMap[key]!
            remove(node)
        }
        let newNode = Node(key, value)
        keyValueMap[key] = newNode
        enqueue(newNode)

        if keyValueMap.count > capacity {
            if let deletedNode = dequeue() {
                keyValueMap[deletedNode.key] = nil
            }

        }
    }

    private func remove(_ node: Node) {
        let next = node.next
        let prev = node.prev
        prev?.next = next
        next?.prev = prev
    }

    private func dequeue() -> Node? {
        guard head.next !== tail else {
            return nil
        }
        let first = head.next
        head.next = first?.next
        head.next?.prev = head
        return first
    }

    private func enqueue(_ node: Node) {
        let last = tail.prev
        last?.next = node
        node.prev = last
        node.next = tail
        tail.prev = node
    }
}