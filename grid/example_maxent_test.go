package grid_test

import (
	"fmt"
	"math"
	"math/rand"
	"github.com/misteroda/go-rl/mdp"
	"github.com/misteroda/go-rl/maxent"
	"github.com/misteroda/go-rl/base"
	"github.com/misteroda/go-rl/grid"
)

func generateExpertDeomonstration(vi *mdp.ValueIterator, g *grid.GridWorld, goal [2]int, starts [][2]int, nSample int, alpha float64) *maxent.Demonstration {
	maxStep := 1000
	goalID, _ := g.StateIDOf(goal[0], goal[1])
	vi.Init()
	vi.SetAlpha(alpha)
	vi.SetAbsorbingState(goalID)
	vi.RunValueIteration()
	vi.UpdatePolicy()
	loader := maxent.NewModelDemonstrationLoader(vi, maxStep)
	
	// var startID int
	for _, xy := range starts {
		startID, _ := g.StateIDOf(xy[0], xy[1])
		loader.SetInitialState(goalID, startID, nSample)
	}
	demo, _ := maxent.NewDemonstration(g.Model, goalID, loader)
	return demo
}


func ExampleMaxEntIRL() {
	nFeature := 5
	nEpoch := 100
	nSample := 100
	
	batchSize := 4
	gamma := 0.5
	width := 20
	height := 20
	alpha := 0.01

	g := grid.NewGridModel(width, height)
	feature := g.CreateRandomFeature(nFeature)
	expert := maxent.NewLinearModel(g.Model, feature, false)
	for i := range expert.Theta {
		expert.Theta[i] = rand.Float64()
	}
	expert.Theta.Normalize()
	cost := expert.ComputeCost()
	g.Model.UpdateReward(cost)

	vi := mdp.NewValueIterator(g.Model)	
	demos := make([]*maxent.Demonstration, 0)
	goals := [][2]int{{width / 2, height / 2}, {width / 2, 0}, {0, height / 2}, {width / 2, height - 1}, {width - 1, height / 2}}
	starts := [][2]int{{0, 0}, {0, height - 1}, {width - 1, height - 1}, {width - 1, 0}}
	for _, goal := range goals {
		demos = append(demos, generateExpertDeomonstration(vi, g, goal, starts, nSample, alpha))
	}
	var gTrain *grid.GridWorld
	for _, alpha := range []float64{0.01, 0.02, 0.04} {
		gTrain = grid.NewGridModel(width, height)
		fmt.Println(alpha)
		trainer := maxent.NewLinearModel(gTrain.Model, feature, false)
		// maxent.UpdateCost(g.Model, feature, trainer.Theta, nil)
		trainer.Fit(demos, nEpoch, batchSize, gamma)

		score := base.CosineSimilarity(expert.Theta, trainer.Theta)
		fmt.Printf("<w>\n")
		fmt.Printf("score: %.2f\n", math.Log10(1 - score))
		fmt.Printf("Learned theta: %v\n", trainer.Theta)
		fmt.Printf("True theta: %v\n", expert.Theta)
		// if flag {
		// 	score := base.CosineSimilarity(expert.UniqueCost, trainer.UniqueCost)
		// 	fmt.Printf("<b>\n")
		// 	fmt.Printf("score: %.2f\n", math.Log10(1 - score))
		// 	fmt.Printf("learned: %v\n", trainer.UniqueCost[uniqueIndex - 1: uniqueIndex + 2])
		// 	fmt.Printf("target: %v\n", expert.UniqueCost[uniqueIndex - 1: uniqueIndex + 2])
		// }
		fmt.Printf("<dist>\n [%.2f", math.Log10(1 - trainer.EvalActionDist(demos[0])))
		for _, demo := range demos[1:] {
			fmt.Printf(", %.2f", math.Log10(1 - trainer.EvalActionDist(demo)))
		}
		fmt.Printf("]\n")
	}
}
