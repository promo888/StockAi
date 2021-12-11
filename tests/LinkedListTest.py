class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, new_value):
        new_node = Node(new_value)
        if self.head is None:
            self.head = Node(new_value)
        else:
            current_node = self.head
            while current_node.next is not None:
                current_node = current_node.next
            current_node.next = new_node


    def swapNodes(self, left_node, right_node):
        if right_node.data < left_node.data:
            if self.head == left_node: #root leaf
                self.head = right_node
                left_node.next = right_node.next
                right_node.next = left_node #right_node.next
            #left_node, right_node = right_node, left_node
            #left_node.next = right_node # pointer handled in next pair search



    def sortAscData(self):
        if self.head is None: #llist is empty
            return

        left_node = self.head
        if left_node.next is None: #only 1element in list
            return
        right_node = left_node.next

        while right_node.next is not None:
            self.swapNodes(left_node, right_node)
            left_node = right_node
            right_node = right_node.next



if __name__ == '__main__':
    llist = LinkedList()
    llist.append(100)
    llist.append(2)
    llist.append(4)
    llist.append(3)
    llist.append(1)

    print(llist.head.data)
    llist.sortAscData()
    print(llist.head.data)