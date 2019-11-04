package maxent

import (
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

type Feature struct {
	N, M int
	values []float64
}

func NewFeature(N, M int) *Feature {
	return &Feature{
		N: N,
		M: M,
		values: make([]float64, N * M),
	}
}

func (f *Feature) Vector(actionIndex int) base.Vector {
	return base.Vector(f.values[f.M * actionIndex: f.M * (actionIndex + 1)])
}

func (f *Feature) Element(i, j int) float64 {
	return f.values[f.M * i + j]
}

func (f *Feature) SetElement(i, j int, v float64) {
	f.values[f.M * i + j] = v
}

type LinearModel struct {
	mdp *mdp.Model
	Feature *Feature
	Theta base.Vector
	UniqueCost base.Vector
}

func NewLinearModel(m *mdp.Model, f *Feature, uniqueCostFlag bool) *LinearModel {
	theta := base.Vector(make([]float64, f.M))
	theta.Fill(1.0 / float64(f.M))

	var uniqueCost base.Vector
	if uniqueCostFlag {
		uniqueCost = base.Vector(make([]float64, f.N))
		uniqueCost.Fill(0.1 / float64(f.N))
	} else {
		uniqueCost = nil
	}
	return &LinearModel{
		mdp: m,
		Feature: f,
		Theta: theta,
		UniqueCost : uniqueCost,
	}
}

func (l *LinearModel) EvalActionDist(demo *Demonstration) float64 {
	vi := mdp.NewValueIterator(l.mdp)
	vi.InitAbsorbingState()
	vi.SetAbsorbingState(demo.goalID)
	vi.RunValueIteration()
	vi.UpdatePolicy()
	_, actionDist := vi.StateActionVisitation(demo.initialStateDist)
	return base.CosineSimilarity(actionDist, demo.actionDist)
}

func (l *LinearModel) ComputeFeatureExpectation(actionDist []float64) []float64 {
	featureExpectation := make([]float64, l.Feature.M)
	for i, d := range actionDist {
		for j := range featureExpectation {
			featureExpectation[j] += d * l.Feature.Element(i, j)
		}
	}
	return featureExpectation
}

func (l *LinearModel) Fit(demonstrations []*Demonstration, nEpoch, numCPU int, gamma float64) {
	// theta := make([]float64, feature.n)
	gradSum := base.Vector(make([]float64, l.Feature.M))

	var uniqueGradSum base.Vector
	if l.UniqueCost != nil {
		uniqueGradSum = base.Vector(make([]float64, l.mdp.NumActions()))
	} else {
		uniqueGradSum = nil
	}
	
	gradMutex := &sync.Mutex{}
	viGroup := make([]*mdp.ValueIterator, numCPU)
	wg := sync.WaitGroup{}
	for i := 0; i < numCPU; i++ {
		viGroup[i] = mdp.NewValueIterator(l.mdp)
	}
	gamma /= float64(numCPU)
	// vi := NewValueIterator(l.mdp)
	for i := 0; i < nEpoch; i++ {
		gradSum.Fill(0.0)
		if uniqueGradSum != nil {
			uniqueGradSum.Fill(0.0)
		}		
		for _, vi := range viGroup {
			wg.Add(1)
			demo := demonstrations[rand.Intn(len(demonstrations))]
			go func(vi *mdp.ValueIterator, demo *Demonstration) {
				defer wg.Done()
				grad, uniqueCost := l.ComputeFeatureExpectationDifference(vi, demo)
				gradMutex.Lock()
				gradSum.Add(base.Vector(grad))
				if uniqueGradSum != nil {
					uniqueGradSum.Add(base.Vector(uniqueCost))
				}
				gradMutex.Unlock()	
			}(vi, demo)
		}
		wg.Wait()
		// Update theta with exponentiated gradient ascent
		if l.UniqueCost != nil {
			l.ExponentiatedGradientAscent(gradSum, gamma)
		} else {
			l.GradientAscent(gradSum, uniqueGradSum, gamma)
		}
		// Update mdp.cost with new theta
		cost := l.ComputeCost()
		l.mdp.UpdateReward(cost)
		gamma *= gradDecay
		// fmt.Printf("gamma : %.3f, grad : %.3f, theta : %v\n", gamma, gradSum, l.Theta)
	}
}


func (l *LinearModel) ExponentiatedGradientAscent(grad []float64, gamma float64) {
	for i, g := range grad {
		l.Theta[i] *= math.Max(-gradClip, (math.Min(gradClip, math.Exp(-gamma * g))))
	}
	l.Theta.Normalize()
}

func (l *LinearModel) GradientAscent(grad []float64, uniqueGrad []float64,gamma float64) {
	for i := range grad {
		l.Theta[i] += -gamma * math.Max(-gradClip, (math.Min(gradClip, grad[i])))
	}
	z := l.Theta.Sum()
	for i := range l.Theta {
		l.Theta[i] /= z
	}
	if uniqueGrad != nil {
		for i, g := range uniqueGrad {
			l.UniqueCost[i] += -gamma * math.Max(-gradClip, (math.Min(gradClip, g)))
			l.UniqueCost[i] /= z
		}
	}
}

func (l *LinearModel) ComputeCost() []float64 {
	cost := make([]float64, l.mdp.NumActions())
	for i := range cost {
		cost[i] = -l.Theta.Dot(l.Feature.Vector(i))
	}
	if l.UniqueCost != nil {
		for i := range cost {
			cost[i] = math.Min(0, cost[i] - l.UniqueCost[i])
		}
	}
	return cost
}

func (l *LinearModel) ComputeFeatureExpectationDifference(vi *mdp.ValueIterator, demo *Demonstration) ([]float64, []float64) {
	vi.InitAbsorbingState()
	vi.SetAbsorbingState(demo.goalID)
	vi.RunValueIteration()
	vi.UpdatePolicy()
	_, actionDist := vi.StateActionVisitation(demo.initialStateDist)
	featureExpectation := l.ComputeFeatureExpectation(actionDist)
	expertFeatureExpectation := l.ComputeFeatureExpectation(demo.actionDist)
	// base.Vector(expertFeatureExpectation).Sub(base.Vector(featureExpectation))
	for i := range expertFeatureExpectation {
		expertFeatureExpectation[i] -= featureExpectation[i]
	}
	if l.UniqueCost == nil {
		return expertFeatureExpectation, nil
	}
	for i := range actionDist {
		actionDist[i] = demo.actionDist[i] - actionDist[i]
	}
	return expertFeatureExpectation, actionDist 
} 
