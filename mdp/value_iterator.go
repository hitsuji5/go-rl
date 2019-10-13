package mdp

import (
	"fmt"
	"math"
	"math/rand"
	"github.com/misteroda/go-rl/base"
)

const (
	pqSize = 10000
	minTDError = 0.001
	minSDError = 0.0001
	maxIterations = 1000000
	numAnnealing = 7
)

// ValueIterator represents Value Iteration algorithm.
type ValueIterator struct {
	model *Model
	V []float64 // state values
	Q []float64 // state-action values
	Policy []float64
	isAbsorbing []bool
	alpha float64 // temperature parameter for softmax operator
}

// NewValueIterator constructs a ValueIterator instance from a given Model.
func NewValueIterator(model *Model) *ValueIterator {
	vi := ValueIterator{
		model: model,
		V: make([]float64, len(model.states)),
		Q: make([]float64, len(model.actions)),
		Policy: make([]float64, len(model.actions)),
		isAbsorbing: make([]bool, len(model.states)),
	}
	return &vi
}

// SetAbsorbingState sets absorbing states in the Model.
// An absorbing state represents the state which terminates an episode.
func (vi *ValueIterator) SetAbsorbingState(stateID int) bool {
	state, ok := vi.model.StateOf[stateID]
	if !ok { return false }
	vi.isAbsorbing[state.idx] = true
	return true
}

// SetAlpha sets a value of alpha.
func (vi *ValueIterator) SetAlpha(alpha float64) {
	vi.alpha = alpha
}

// InitAbsorbingState initializes absorbing states.
func (vi *ValueIterator) InitAbsorbingState() {
	for i := range vi.isAbsorbing {
		vi.isAbsorbing[i] = false
	}
}

// Init initializes ValueIterator instance.
func (vi *ValueIterator) Init() {
	base.Vector(vi.V).Fill(0.0)
	base.Vector(vi.Q).Fill(0.0)
	base.Vector(vi.Policy).Fill(0.0)
	vi.InitAbsorbingState()
	vi.alpha = 0.0
}

// ToActions returns an action space of a given state as []*Action.
func (vi *ValueIterator) ToActions(s *State) []*Action {
	if vi.isAbsorbing[s.idx] {
		return s.actions[:0]
	}
	return s.actions
}

// RunValueIteration runs Value Iteration algorithm and updates V and Q.
func (vi *ValueIterator) RunValueIteration() {
	m := vi.model
	pq := NewPriorityQueue(len(m.states))
	var td float64
	var j int
	tdThreshold := minTDError * math.Pow(2, float64(numAnnealing - 1))
	for i := 0; i < numAnnealing; i++ {
		for stateIdx := range m.states {
			s := &m.states[stateIdx]
			td = vi.bellmanBackup(s)
			if td > tdThreshold && pq.Size() < pqSize {
				pq.Push(stateIdx, td)
			}
		}
		for j = 0; j < maxIterations && pq.Size() > 0; j++ {
			idx, _ := pq.Pop()
			for _, tr := range m.states[idx].transitions {
				s := tr.action.state
				td = vi.bellmanBackup(s)
				if td > tdThreshold && pq.Size() < pqSize {
					pq.Push(s.idx, td)
				}
			}
		}
		// fmt.Printf("Annealing #%d { iters: %d, theta: %.3f, td: %.3f }\n", i, j, tdThreshold, td)
		tdThreshold *= 0.5
	}
}

// UpdatePolicy updates policy based on current state-action values.
func (vi *ValueIterator) UpdatePolicy() {
	m := vi.model
	for stateIdx := range m.states {
		s := &m.states[stateIdx]
		actions := vi.ToActions(s)
		if len(actions) == 0 { continue }
		bestAction := vi.bestAction(actions)

		if vi.alpha == 0 {
			for _, a := range actions {
				vi.Policy[a.idx] = 0
			}
			vi.Policy[bestAction.idx] = 1
		} else {
			maxQ := vi.Q[bestAction.idx]
			z := 0.0
			for _, a := range actions {
				vi.Policy[a.idx] = math.Exp((vi.Q[a.idx] - maxQ) / vi.alpha)
				z += vi.Policy[a.idx]
			}
			for _, a := range s.actions {
				vi.Policy[a.idx] /= z
			}
		}
	}
}


func (vi *ValueIterator) bellmanBackup(s *State) (tdError float64) {
	actions := vi.ToActions(s)
	if len(actions) == 0 { return }
	for _, a := range actions {
		vi.Q[a.idx] = a.transition.r + vi.V[a.transition.state.idx]
	}
	v := vi.softMax(s.actions)
	tdError = math.Abs(v - vi.V[s.idx])	
	vi.V[s.idx] = v
	return
}

func (vi *ValueIterator) softMax(actions []*Action) float64 {
	if len(actions) == 0 {
		panic("model: zero slice length")
	}
	maxQ := vi.Q[actions[0].idx]
	for _, a := range actions[1:] {
		if maxQ < vi.Q[a.idx] {
			maxQ = vi.Q[a.idx]
		}
	}
	if vi.alpha == 0 {
		return maxQ
	}	
	var lse float64
	for _, a := range actions {
		lse += math.Exp((vi.Q[a.idx] - maxQ) / vi.alpha)
	}
	return vi.alpha * math.Log(lse) + maxQ
}

func (vi *ValueIterator) bestAction(actions []*Action) *Action {
	if len(actions) == 0 {
		panic("model: zero slice length")
	}
	maxA := actions[0]
	maxQ := vi.Q[maxA.idx]
	for _, a := range actions {
		if maxQ < vi.Q[a.idx] {
			maxA = a
			maxQ = vi.Q[a.idx]
		}
	}
	return maxA
}

func (vi *ValueIterator) sampleAction(actions []*Action) *Action {
	if len(actions) == 0 {
		panic("model: zero slice length")
	}
	cumP := 0.0
	r := rand.Float64()
	for _, a := range actions[:len(actions)-1] {
		cumP += vi.Policy[a.idx]
		if r < cumP {
			return a
		}
	}
	return actions[len(actions) - 1]
}

// GenerateTrajectory generates trajectory of given a start and a goal state based on current policy.
func (vi *ValueIterator) GenerateTrajectory(startID, goalID, maxSteps int) (tr []int, ok bool) {
	m := vi.model
	startState, ok := m.StateOf[startID]
	if !ok { return }
	goalState, ok := m.StateOf[goalID]
	if !ok { return }
	s := startState
	tr = append(tr, s.name)
	for i := 0; i < maxSteps; i++ {
		actions := vi.ToActions(s)
		if len(s.actions) == 0 { return }
		s = vi.sampleAction(actions).transition.state
		tr = append(tr, s.name)
		if s == goalState {
			ok = true
			return
		}
	}
	ok = false
	return
}


func (vi *ValueIterator) computeStateDist(s *State, actionDist []float64) float64 {
	d := 0.0
	for _, tr := range s.transitions {
		if vi.isAbsorbing[tr.action.state.idx] { continue }
		d += actionDist[tr.action.idx]
	}
	return d
}

// StateActionVisitation computes state-action visitation frequency distribution based on the current policy.
func (vi *ValueIterator) StateActionVisitation(initialStateDist []float64) ([]float64, []float64) {
	m := vi.model
	stateDist := make([]float64, len(m.states))
	copy(stateDist, initialStateDist)
	actionDist := make([]float64, len(m.actions))

	pq := NewPriorityQueue(len(stateDist))
	var sd float64
	var j, idx int
	sdThreshold := minSDError * math.Pow(2, float64(numAnnealing - 1))
	for i := 0; i < numAnnealing; i++ {
		if i == 0 {
			for idx, sd = range initialStateDist {
				if sd > 0 {
					pq.Push(idx, sd)
				}
			}
		} else {
			for stateIdx := range stateDist {
				s := &m.states[stateIdx]
				actions := vi.ToActions(s)
				if len(actions) == 0 || stateDist[stateIdx] < sdThreshold { continue }
				for _, a := range actions {
					actionDist[a.idx] = stateDist[stateIdx] * vi.Policy[a.idx]
					nextState := a.transition.state
					idx := nextState.idx
					sd = stateDist[idx]
					stateDist[idx] = initialStateDist[idx] + vi.computeStateDist(nextState, actionDist)
					sd = math.Abs(sd - stateDist[idx])
					if sd > sdThreshold && pq.Size() < pqSize {
						pq.Push(stateIdx, sd)
					}
				}
			}
		}
		
		for j = 0; j < maxIterations && pq.Size() > 0; j++ {
			stateIdx, _ := pq.Pop()
			s := &m.states[stateIdx]
			actions := vi.ToActions(s)
			if len(actions) == 0 || stateDist[stateIdx] < sdThreshold { continue }
			for _, a := range actions {
				actionDist[a.idx] = stateDist[stateIdx] * vi.Policy[a.idx]
				nextState := a.transition.state
				idx := nextState.idx
				sd = stateDist[idx]
				stateDist[idx] = initialStateDist[idx] + vi.computeStateDist(nextState, actionDist)
				sd = math.Abs(sd - stateDist[idx])
				if sd > sdThreshold && pq.Size() < pqSize {
					pq.Push(idx, sd)
				}
			}
		}
		sdThreshold *= 0.5
	}

	return stateDist, actionDist
}

func (vi *ValueIterator) String() string {
	s := fmt.Sprintf("V: %v\n", vi.V)
	s += fmt.Sprintf("Q: %v\n", vi.Q)
	s += fmt.Sprintf("Policy: %v\n", vi.Policy)
	return s
}
