"""Data structure implementations for testing."""

from typing import Optional


class Stack:
    """A simple stack implementation."""

    def __init__(self) -> None:
        """Initialize empty stack."""
        self._items: list[object] = []

    def push(self, item: object) -> None:
        """Push an item onto the stack."""
        self._items.append(item)

    def pop(self) -> object:
        """Pop an item from the stack."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self) -> object:
        """Peek at the top item without removing it."""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        """Check if the stack is empty."""
        return len(self._items) == 0

    def size(self) -> int:
        """Get the number of items in the stack."""
        return len(self._items)


class Queue:
    """A simple queue implementation."""

    def __init__(self) -> None:
        """Initialize empty queue."""
        self._items: list[object] = []

    def enqueue(self, item: object) -> None:
        """Add an item to the rear of the queue."""
        self._items.append(item)

    def dequeue(self) -> object:
        """Remove and return an item from the front of the queue."""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._items.pop(0)

    def front(self) -> object:
        """Peek at the front item without removing it."""
        if self.is_empty():
            raise IndexError("front from empty queue")
        return self._items[0]

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self._items) == 0

    def size(self) -> int:
        """Get the number of items in the queue."""
        return len(self._items)


class Node:
    """A simple node for linked list."""

    def __init__(self, data: object, next_node: Optional["Node"] = None) -> None:
        """Initialize node with data and optional next reference."""
        self.data = data
        self.next = next_node


class LinkedList:
    """A simple singly linked list implementation."""

    def __init__(self) -> None:
        """Initialize empty linked list."""
        self.head: Node | None = None
        self._size = 0

    def append(self, data: object) -> None:
        """Append data to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self._size += 1

    def prepend(self, data: object) -> None:
        """Prepend data to the beginning of the list."""
        new_node = Node(data, self.head)
        self.head = new_node
        self._size += 1

    def remove(self, data: object) -> bool:
        """Remove the first occurrence of data from the list."""
        if not self.head:
            return False

        if self.head.data == data:
            self.head = self.head.next
            self._size -= 1
            return True

        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                self._size -= 1
                return True
            current = current.next
        return False

    def find(self, data: object) -> bool:
        """Check if data exists in the list."""
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False

    def size(self) -> int:
        """Get the number of items in the list."""
        return self._size

    def is_empty(self) -> bool:
        """Check if the list is empty."""
        return self.head is None

    def to_list(self) -> list[object]:
        """Convert linked list to Python list."""
        result = []
        current = self.head
        while current:
            result.append(current.data)
            current = current.next
        return result
