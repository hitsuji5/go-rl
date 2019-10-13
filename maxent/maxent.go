package maxent

import (
	// "fmt"
	"math"
	"math/rand"
	"sync"
	"github.com/misteroda/go-rl/base"
	"github.com/misteroda/go-rl/mdp"
)

const (
	gradClip = 3.0
	gradDecay = 0.99
)

type Demonstration struct {
	goalState int
	initialStateDist []float64
	actionDist []float64
	nSample int //Number of trejectories
}

func NewDemonstration(m *mdp.Model, goalState int, trajectories [][]int) *Demonstration {
	nSample := len(trajectories)
	initialStateDist := make([]float64, m.NumStates())
	actionDist := make([]float64, m.NumActions())
	for _, tr := range trajectories {
		state, ok := m.StateOf[tr[0]]
		if !ok { continue }
		initialStateDist[state.Index()]++
		fromState := tr[0]
		for _, toState := range tr[1:] {
			acition, ok := m.Action(fromState, toState)
			if ok { actionDist[acition.Index()]++ }
			fromState = toState
		}
	}
	for i := range initialStateDist {
		initialStateDist[i] /= float64(nSample)
	}
	for i := range actionDist {
		actionDist[i] /= float64(nSample)
	}
	demo := &Demonstration{
		goalState: goalState,
		initialStateDist: initialStateDist,
		actionDist: actionDist,
		nSample: nSample,
	}
	return demo
}

type Trainer struct {
	mdp *mdp.Model
	features [][]float64
	nFeature int
	Theta []float64
}

func NewTrainer(m *mdp.Model, features [][]float64, nFeature int) *Trainer {
	theta := make([]float64, nFeature)
	base.Vector(theta).Fill(1.0 / float64(nFeature))
	return &Trainer{
		mdp: m,
		features: features,
		nFeature: nFeature,
		Theta: theta,
	}
}

func (t *Trainer) EvalActionDist(demo *Demonstration) float64 {
	vi := mdp.NewValueIterator(t.mdp)
	vi.InitAbsorbingState()
	vi.SetAbsorbingState(demo.goalState)
	vi.RunValueIteration()
	vi.UpdatePolicy()
	_, actionDist := vi.StateActionVisitation(demo.initialStateDist)
	return base.CosineSimilarity(actionDist, demo.actionDist)
}

func (t *Trainer) ComputeFeatureExpectation(actionDist []float64) []float64 {
	featureExpectation := make([]float64, t.nFeature)
	for i, d := range actionDist {
		for j := range featureExpectation {
			featureExpectation[j] += d * t.features[i][j]
		}
	}
	return featureExpectation
}

func (t *Trainer) Fit(demonstrations []*Demonstration, nEpoch, numCPU int, gamma float64) {
	// theta := make([]float64, feature.n)
	gradSum := make([]float64, t.nFeature)
	gradMutex := &sync.Mutex{}
	viGroup := make([]*mdp.ValueIterator, numCPU)
	wg := sync.WaitGroup{}
	for i := 0; i < numCPU; i++ {
		viGroup[i] = mdp.NewValueIterator(t.mdp)
	}
	gamma /= float64(numCPU)
	// vi := NewValueIterator(t.mdp)
	for i := 0; i < nEpoch; i++ {
		base.Vector(gradSum).Fill(0.0)
		for _, vi := range viGroup {
			wg.Add(1)
			demo := demonstrations[rand.Intn(len(demonstrations))]
			go func(vi *mdp.ValueIterator, demo *Demonstration) {
				defer wg.Done()
				grad := t.ComputeFeatureExpectationDifference(vi, demo)
				gradMutex.Lock()
				base.Vector(gradSum).Add(base.Vector(grad))
				gradMutex.Unlock()	
			}(vi, demo)
			// base.Vector(gradSum).Add(base.Vector(grad))
		}
		wg.Wait()
		// Update theta with exponentiated gradient ascent
		EponentiatedGradientAscent(t.Theta, gradSum, gamma)
		// Update mdp.cost with new theta
		UpdateCost(t.mdp, t.features, t.Theta)
		gamma *= gradDecay
		// fmt.Printf("gamma : %.3f, grad : %.3f, theta : %v\n", gamma, gradSum, t.Theta)
	}
}


func EponentiatedGradientAscent(theta, grad []float64, gamma float64) {
	for i := range grad {
		theta[i] *= math.Max(-gradClip, (math.Min(gradClip, math.Exp(-gamma * grad[i]))))
	}
	base.Vector(theta).Normalize()
}

func UpdateCost(m *mdp.Model, features [][]float64, theta []float64) {
	for i := 0; i < m.NumActions(); i++ {
		m.SetReward(i, -base.Vector(theta).Dot(base.Vector(features[i])))
	}
}

func (t *Trainer) ComputeFeatureExpectationDifference(vi *mdp.ValueIterator, demo *Demonstration) []float64 {
	vi.InitAbsorbingState()
	vi.SetAbsorbingState(demo.goalState)
	vi.RunValueIteration()
	vi.UpdatePolicy()
	_, actionDist := vi.StateActionVisitation(demo.initialStateDist)
	featureExpectation := t.ComputeFeatureExpectation(actionDist)
	expertFeatureExpectation := t.ComputeFeatureExpectation(demo.actionDist)
	base.Vector(expertFeatureExpectation).Sub(base.Vector(featureExpectation))
	return expertFeatureExpectation
} 