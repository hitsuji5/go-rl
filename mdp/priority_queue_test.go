package mdp

import (
	"testing"
)

func TestPriorityQueue(t *testing.T) {
	cases := []struct {
		p float64
		index int
	}{
		{4, 3},
		{1.11, 1},
		{-100, 4},
		{3, 4},
		{2.0, 5},
	}
	wants := []struct {
		p float64
		index int
	}{
		{4, 3},
		{3, 4},
		{2.0, 5},
		{1.11, 1},
	}

	pq := NewPriorityQueue(10)
	for _, c := range cases {
		pq.Push(c.index, c.p)
	}

	for _, want := range wants {
		index, p := pq.Pop()
		if want.index != index || want.p != p {
			t.Errorf("want:%#v, got index:%d, p:%f", want, index, p)
		}
	}
}