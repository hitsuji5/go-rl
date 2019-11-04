package maxent

import (
	"fmt"
	"github.com/misteroda/go-rl/mdp"
)

// DemonstrationLoader describes the requirements
// for loading expert demonstraion data.
type DemonstrationLoader interface {
	LoadInitialState(goalID int) ([]InitialState, error)
	LoadTransitionVisitation(goalID int) ([]TransitionVisitation, error)
}

// InitialState represents frequency of initial states 
// in the expert demonstration.
type InitialState struct {
	ID int
	Count int
}
// TransitionVisitation represents visitation frequency 
// of state-state transitions in the expert demonstration.
type TransitionVisitation struct {
	FromID, ToID int
	Count int
}

// ModelDemonstrationLoader implements the expert policy
// for generating demonstration data.
type ModelDemonstrationLoader struct {
	vi *mdp.ValueIterator
	initialStateOf map[int][]InitialState
	maxStep int
}

// NewModelDemonstrationLoader constructs ModelDemonstrationLoader.
// ValueIterator after updating policy must be given as an argument.
func NewModelDemonstrationLoader(vi *mdp.ValueIterator, maxStep int) *ModelDemonstrationLoader {
	return &ModelDemonstrationLoader{
		vi: vi,
		initialStateOf: make(map[int][]InitialState),
		maxStep: maxStep,
	}
}

// SetInitialState sets initial state id and count.
func (m *ModelDemonstrationLoader) SetInitialState(goalID, id, count int) {
	m.initialStateOf[goalID] = append(m.initialStateOf[goalID], InitialState{id, count})
}


func (m *ModelDemonstrationLoader) LoadInitialState(goalID int) ([]InitialState, error) {
	s, ok := m.initialStateOf[goalID]
	if !ok {
		return nil, nil
	}
	return s, nil
}

func (m *ModelDemonstrationLoader) LoadTransitionVisitation(goalID int) ([]TransitionVisitation, error) {
	initialStates, ok := m.initialStateOf[goalID]
	if !ok {
		return nil, nil
	}

	transitionCount := make(map[int]map[int]int)
	for _, s := range initialStates {
		for j := 0; j < s.Count; j++ {
			traj, ok := m.vi.GenerateTrajectory(s.ID, goalID, m.maxStep)
			if !ok {
				fmt.Printf("Unable to reach the goal: %d\n", goalID)
				continue
			}
			fromID := traj[0]
			for _, toID := range traj[1:] {
				if _, ok := transitionCount[fromID]; !ok {
					transitionCount[fromID] = make(map[int]int)
				}
				transitionCount[fromID][toID]++
				fromID = toID
			}
		}
	}
	transitions := make([]TransitionVisitation, 0)
	for fromID := range transitionCount {
		for toID, count := range transitionCount[fromID] {
			transitions = append(transitions, TransitionVisitation{
				FromID: fromID,
				ToID: toID,
				Count: count,
			})
		}
	}
	return transitions, nil
}

// Demonstration represents expert demonstration data
// for training MaxEnt IRL model.
type Demonstration struct {
	goalID int
	initialStateDist []float64
	actionDist []float64
	nSample int //Number of trejectories
}

// NewDemonstration constructs Demonstraion.
func NewDemonstration(m *mdp.Model, goalID int, loader DemonstrationLoader) (*Demonstration, error) {
	initialStates, err := loader.LoadInitialState(goalID)
	if err != nil {
		return nil, err
	}

	transitions, err := loader.LoadTransitionVisitation(goalID)
	if err != nil {
		return nil, err
	}

	initialStateDist := make([]float64, m.NumStates())
	var nSample int
	for _, s := range initialStates {
		state, ok := m.StateOf[s.ID]
		if !ok { continue }
		initialStateDist[state.Index()] += float64(s.Count)
		nSample += s.Count
	}
	
	actionDist := make([]float64, m.NumActions())
	for _, t := range transitions {
		action, ok := m.ActionByID(t.FromID, t.ToID)
		if !ok { continue }
		actionDist[action.Index()] += float64(t.Count)
	}
	for i := range initialStateDist {
		initialStateDist[i] /= float64(nSample)
	}
	for i := range actionDist {
		actionDist[i] /= float64(nSample)
	}
	demo := &Demonstration{
		goalID: goalID,
		initialStateDist: initialStateDist,
		actionDist: actionDist,
		nSample: nSample,
	}
	return demo, nil
}
