package grid

import (
	"math/rand"
	"github.com/misteroda/go-rl/mdp"
	"github.com/misteroda/go-rl/maxent"
)
const (
	cost = 1.0
)

type gridMove struct {
	dx, dy int
}

type GridWorld struct {
	width, height int
	Model *mdp.Model
}

func (g *GridWorld) StateIDOf(x, y int) (stateID int, ok bool) {
	if x < 0 || x >= g.width || y < 0 || y >= g.height {
		return
	}
	ok = true
	stateID = x * g.height + y
	return
}

func (g *GridWorld) CoordinateOf(stateID int) (x, y int, ok bool) {
	if stateID >= g.width * g.height {
		return
	}
	ok = true
	x = stateID / g.height
	y = stateID % g.height
	return
}

func NewGridModel(width, height int) *GridWorld {
	moves := []gridMove{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	nMove := len(moves)
	// coordinateOf := make(map[int]GridCoordinate)
	stateIDs := make([]int, 0, width * height)
	g := GridWorld{width: width, height: height}
	actions := make([]mdp.StateTransition, 0, width * height * nMove + 1) 
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			id, _ := g.StateIDOf(x, y)
			stateIDs = append(stateIDs, id)
			for _, move := range moves {
				toX, toY := x + move.dx, y + move.dy
				if toX < 0 || toX >= width || toY < 0 || toY >= height { continue }
				toID, _ := g.StateIDOf(toX, toY)
				actions = append(actions, mdp.StateTransition{
					FromID: id,
					ToID: toID,
					Reward: -cost,
				})
			}
		}
	}
	g.Model = mdp.NewModel(stateIDs, actions)
	return &g
}

func (g *GridWorld) CreateRandomFeature(nFeature int) *maxent.Feature {
	feature := maxent.NewFeature(g.Model.NumActions(), nFeature)
	for i := 0; i < g.Model.NumActions(); i++ {
		// features[i] = make([]float64, nFeature)
		for j := 0; j < nFeature; j++ {
			feature.SetElement(i, j, rand.Float64())
		}
	}
	return feature
}
