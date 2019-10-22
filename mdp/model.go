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
func (m *Model) UpdateReward(reward func(a *Action) float64) {
	for i := range m.actions {
		if m.actions[i].transition == nil { continue }
		m.actions[i].transition.r = reward(&m.actions[i])
	}
}

// State represents a state of Model.
type State struct {
	name int
	idx int
	actions []*Action
	transitions []*Transition
}

// Index returns the array index of the state.
func (s *State) Index() int {
	return s.idx
}

// Action represents an action of Model.
type Action struct {
	idx int
	state *State
	transition *Transition
}

// Index returns the array index of the action.
func (a *Action) Index() int {
	return a.idx
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
	FromState, ToState int
	Reward float64 // reward
}

// NewModel constructs a Model instance and returns a pointer to it.
func NewModel(stateNames []int, stateTransitions []StateTransition) *Model {
	// construct stateID => stateIdx Map
	StateOf := make(map[int]*State)
	states := make([]State, len(stateNames))
	for i, name := range stateNames {
		states[i] = State{
			name: name,
			idx : i,
			actions: make([]*Action, 0),
			transitions: make([]*Transition, 0),
		}
		StateOf[name] = &states[i]
	}
	actions := make([]Action, len(stateTransitions))
	transitions := make([]Transition, len(stateTransitions))
	for idx, st := range stateTransitions {
		toState, ok := StateOf[st.ToState]
		if !ok {
			continue
		}
		state, ok := StateOf[st.FromState]
		if !ok {
			continue
		}
		actions[idx].state = state
		actions[idx].idx = idx
		actions[idx].transition = &transitions[idx]
		transitions[idx].state = toState
		transitions[idx].action = &actions[idx]
		transitions[idx].r = st.Reward
		state.actions = append(state.actions, &actions[idx])
		toState.transitions = append(toState.transitions, &transitions[idx])
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
