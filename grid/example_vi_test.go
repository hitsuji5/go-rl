package grid_test

import (
	"fmt"
	"github.com/misteroda/go-rl/mdp"
	"github.com/misteroda/go-rl/grid"
)

func ExampleValueIterator() {
	width := 3
	height := 3
	g := grid.NewGridModel(width, height)
	goalID, _ := g.StateIDOf(width - 1, height - 1)
	vi := mdp.NewValueIterator(g.Model)
	vi.SetAbsorbingState(goalID)
	vi.RunValueIteration()
	vi.UpdatePolicy()
	startID, _ := g.StateIDOf(0, 0)
	tr, _ := vi.GenerateTrajectory(startID, goalID, 10)
	for _, id := range tr {
		fmt.Println(g.CoordinateOf(id))
	}
	// Output:
	// 0 0 true
	// 1 0 true
	// 2 0 true
	// 2 1 true
	// 2 2 true
}
