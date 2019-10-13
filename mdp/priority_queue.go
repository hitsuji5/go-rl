package mdp

import (
	"container/heap"
	// "fmt"
)

// An pqItem is something we manage in a priority queue.
type pqItem struct {
	value    int // The value of the pqItem; arbitrary.
	priority float64    // The priority of the pqItem in the queue.
	// The index is needed by update and is maintained by the heap.Interface methods.
	index int // The index of the pqItem in the heap.
}

// A priorityQueue implements heap.Interface and holds Items.
type priorityQueue []*pqItem

func (pq priorityQueue) Len() int { return len(pq) }

func (pq priorityQueue) Less(i, j int) bool {
	// We want Pop to give us the highest, not lowest, priority so we use greater than here.
	return pq[i].priority > pq[j].priority
}

func (pq priorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *priorityQueue) Push(x interface{}) {
	pqItem := x.(*pqItem)
	pqItem.index = len(*pq)
	*pq = append(*pq, pqItem)
}

func (pq *priorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	pqItem := old[n-1]
	old[n-1] = nil  // avoid memory leak
	pqItem.index = -1 // for safety
	*pq = old[0 : n-1]
	return pqItem
}

type PriorityQueue struct {
	pq *priorityQueue
	items []pqItem
}

func (pq *PriorityQueue) Push(idx int, priority float64) {
	if pq.items[idx].index < 0 {
		pq.items[idx].priority = priority
		heap.Push(pq.pq, &pq.items[idx])
	} else if pq.items[idx].priority < priority {
		pq.items[idx].priority = priority
		heap.Fix(pq.pq, pq.items[idx].index)
	}
}

func (pq *PriorityQueue) Pop() (int, float64) {
	item := heap.Pop(pq.pq).(*pqItem)
	return item.value, item.priority
}

func (pq *PriorityQueue) Size() int {
	return len(*pq.pq)
}

func NewPriorityQueue(length int) *PriorityQueue {
	pq := make(priorityQueue, 0)
	items := make([]pqItem, length)
	for i := range items {
		items[i].value = i
		items[i].index = -1
	}
	heap.Init(&pq)
	return &PriorityQueue{pq: &pq, items:items}
}
