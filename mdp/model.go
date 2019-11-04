package mdp

// Model is the model parameters of Markov Descision Processes of your problem.
type Model struct {
	StateOf map[int]*State
	states []State
	actions []Action
	transitions []Transition
}

// NumStates returns a number of states in MDP.
func (m *Model) NumStates() int {
	return len(m.states)
}

// NumActions returns a number of actions in MDP.
func (m *Model) NumActions() int {
	return len(m.actions)
}

// UpdateReward update the reward of all actions.
func (m *Model) UpdateReward(reward []float64) bool {
	if len(reward) != m.NumActions() {
		return false
	}
	for i := range m.actions {
		if m.actions[i].transition == nil { continue }
		m.actions[i].transition.r = reward[i]
	}
	return true
}

// State represents a state of Model.
type State struct {
	id int
	index int
	actions []*Action
	transitions []*Transition
}

// Index returns the array index of the state.
func (s *State) Index() int {
	return s.index
}

// Action represents an action of Model.
type Action struct {
	index int
	state *State
	transition *Transition
}

// Index returns the array index of the action.
func (a *Action) Index() int {
	return a.index
}

// Transition represents a action-state transition of Model.
// Since the Model is deterministic, an action corresponds to the state one-to-one.
// If stochastic Model, Action has multiple transitions and their probability.
type Transition struct {
	action *Action
	state *State
	r float64 // reward
}

// StateTransition is a stete-state transition in a deterministic Model.
type StateTransition struct {
	FromID, ToID int
	Reward float64 // reward
}

// NewModel constructs a Model instance and returns a pointer to it.
func NewModel(stateIDs []int, stateTransitions []StateTransition) *Model {
	// construct stateID => stateIdx Map
	StateOf := make(map[int]*State)
	states := make([]State, len(stateIDs))
	for i, id := range stateIDs {
		states[i] = State{
			id: id,
			index : i,
			actions: make([]*Action, 0),
			transitions: make([]*Transition, 0),
		}
		StateOf[id] = &states[i]
	}
	actions := make([]Action, len(stateTransitions))
	transitions := make([]Transition, len(stateTransitions))
	for i, st := range stateTransitions {
		toState, ok := StateOf[st.ToID]
		if !ok {
			continue
		}
		state, ok := StateOf[st.FromID]
		if !ok {
			continue
		}
		actions[i].state = state
		actions[i].index = i
		actions[i].transition = &transitions[i]
		transitions[i].state = toState
		transitions[i].action = &actions[i]
		transitions[i].r = st.Reward
		state.actions = append(state.actions, &actions[i])
		toState.transitions = append(toState.transitions, &transitions[i])
	}

	m := &Model{
		states: states,
		actions: actions,
		transitions: transitions,
		StateOf: StateOf,
	}
	return m
}

// ActionByID returns the action satisfied with a given state transition.
func (m *Model) ActionByID(fromStateID, toStateID int) (a *Action, ok bool) {
	fromState, ok := m.StateOf[fromStateID]
	if !ok { return }
	toState, ok := m.StateOf[toStateID]
	if !ok { return }
	for _, action := range fromState.actions {
		if action.transition.state == toState {
			a = action
			ok = true
			return
		}
	}
	return
}
