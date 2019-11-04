package mdp

import (
	"testing"
)

var (
	m = NewModel(
		[]int{0, 1, 2, 3},
		[]StateTransition{{0, 1, -1}, {1, 2, -1}, {1, 3, 0}, {2, 3, -1}, {3, 0, -1}},
	)
	vi = NewValueIterator(m)
	
)

func run(vi *ValueIterator, goal int) {
	vi.Init()
	ok := vi.SetAbsorbingState(goal)
	if !ok {
		panic("failed")
	}
	vi.RunValueIteration()
	vi.UpdatePolicy()
}


func TestValueIteration(t *testing.T) {
	cases := []struct {
		goal int
		V []float64
		Policy []float64
	}{
		{goal: 3, V: []float64{-1, 0, -1, 0}, Policy: []float64{1, 0, 1, 1, 0}},
		{goal: 2, V: []float64{-2, -1, 0, -3}, Policy: []float64{1, 1, 0, 0, 1}},
	}
	for _, c := range cases {
		run(vi, c.goal)
		got := vi.V
		want := c.V
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("@%d: got %.3f, want %.3f", i, got[i], want[i])
			}
		}

		got = vi.Policy
		want = c.Policy
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("@%d: got %.3f, want %.3f", i, got[i], want[i])
			}
		}
	}
}


func TestStateActionVisitation(t *testing.T) {
	cases := []struct {
		goal int
		initialState []float64
		stateDist []float64
		actionDist []float64
		
	}{
		{
			goal: 3,
			initialState: []float64{1, 0, 0, 0},
			stateDist: []float64{1, 1, 0, 1},
			actionDist: []float64{1, 0, 1, 0, 0},
		},
		{
			goal: 2,
			initialState: []float64{0, 0, 0, 1},
			stateDist: []float64{1, 1, 1, 1},
			actionDist: []float64{1, 1, 0, 0, 1},
		},
	}
	for _, c := range cases {
		run(vi, c.goal)
		stateDist, actionDist := vi.StateActionVisitation(c.initialState)
		got := stateDist
		want := c.stateDist
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("@%d: got %.3f, want %.3f", i, got[i], want[i])
			}
		}

		got = actionDist
		want = c.actionDist
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("@%d: got %.3f, want %.3f", i, got[i], want[i])
			}
		}
	}
}