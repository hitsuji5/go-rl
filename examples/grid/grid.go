package main

import (
	"fmt"
	"math"
	"math/rand"
	"github.com/misteroda/go-rl/base"
	"github.com/misteroda/go-rl/mdp"
	"github.com/misteroda/go-rl/maxent"
	"github.com/pkg/profile"
)

func CoordinateToState(x, y int) int {
	return x * 100 + y	
}

func stateToCoordinate(state int) (int, int) {
	x := int(state / 100)
	y := state % 100
	return x, y
}

func GridWorld(width, height int) ([]int, []mdp.StateTransition) {
	moves := []struct {
		dx, dy int
	}{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	nMove := len(moves)
	states := make([]int, 0, width * height + 1)
	actions := make([]mdp.StateTransition, 0, width * height * nMove + 1) 
	for x := 1; x <= width; x++ {
		for y := 1; y <= height; y++ {
			state := CoordinateToState(x, y)
			states = append(states, state)
			for _, move := range moves {
				toX, toY := x + move.dx, y + move.dy
				if toX <= 0 || toX > width || toY <=0 || toY > height { continue }
				toState := CoordinateToState(toX, toY)
				actions = append(actions, mdp.StateTransition{state, toState, -1})
			}
		}
	}
	return states, actions
}

func CreateRandomFeature(actions []mdp.StateTransition, nFeature int) [][]float64 {
	features := make([][]float64, len(actions))
	for i := range actions {
		features[i] = make([]float64, nFeature)
		for j := 0; j < nFeature; j++ {
			features[i][j] = rand.Float64()
		}
	}
	return features
}


func GenerateExpertTrajectories(vi *mdp.ValueIterator, initialStates []int, goalState int, nSample int) [][]int {
	trajectories := make([][]int, nSample * len(initialStates))
	for i, initialState := range initialStates {
		for j := 0; j < nSample; j++ {
			tr, ok := vi.GenerateTrajectory(initialState, goalState, 1000)
			if !ok {
				fmt.Printf("Unable to reach the goal: %d\n", goalState)
			}
			trajectories[nSample * i + j] = tr
		}
	}
	return trajectories
}

func main() {
	nFeature := 5
	nEpoch := 100
	nSample := 1000
	batchSize := 4
	gamma := 0.5
	width := 20
	height := 20
	// iters := width + height + 10

	states, actions := GridWorld(width, height)
	features := CreateRandomFeature(actions, nFeature)
	m := mdp.NewModel(states, actions)
	vi := mdp.NewValueIterator(m)
	trueTheta := make([]float64, nFeature)
	for i := range trueTheta {
		trueTheta[i] = rand.Float64()
	}
	base.Vector(trueTheta).Normalize()
	maxent.UpdateCost(m, features, trueTheta)
	fmt.Printf("True theta: %v\n", trueTheta)

	cases := []struct {
		initialStates []int
		goalState int
	}{
		{
			initialStates: []int {CoordinateToState(1, 1)},
			goalState: CoordinateToState(width, height),
		},
		{
			initialStates: []int {CoordinateToState(width, height)},
			goalState: CoordinateToState(1, 1),
		},
		{
			initialStates: []int {CoordinateToState(width, 1)},
			goalState: CoordinateToState(1, height),
		},
		{
			initialStates: []int {CoordinateToState(1, height)},
			goalState: CoordinateToState(width, 1),
		},
	}

	defer profile.Start(profile.ProfilePath(".")).Stop()
	for a := 0.01; a <= 0.02; a *= 2 {
		demos := make([]*maxent.Demonstration, len(cases))
		for i, c := range cases {
			vi.Init()
			vi.SetAlpha(a)
			vi.SetAbsorbingState(c.goalState)
			vi.RunValueIteration()
			vi.UpdatePolicy()
			// fmt.Println(vi)
			trajectories := GenerateExpertTrajectories(vi, c.initialStates, c.goalState, nSample)
			demos[i] = maxent.NewDemonstration(m, c.goalState, trajectories)
		}


		trainer := maxent.NewTrainer(m, features, nFeature)
		maxent.UpdateCost(m, features, trainer.Theta)
		trainer.Fit(demos, nEpoch, batchSize, gamma)

		theta := trainer.Theta
		// rmse := math.Sqrt(ComputeMSE(trueTheta, theta))
		cosSimilarity := base.CosineSimilarity(trueTheta, theta)
		fmt.Printf("a: %.2f, theta_err: %.2f\ntheta: %v\n", a, math.Log10(1 - cosSimilarity), theta)
		fmt.Printf("dist_err : [")
		for i, demo := range demos {
			fmt.Printf("%.2f", math.Log10(1 - trainer.EvalActionDist(demo)))
			if i < len(demos) - 1 {
				fmt.Printf(" ")
			}
		}
		fmt.Printf("]\n")
	}

}
