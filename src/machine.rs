//! # State Machine — Runtime Autonomous State Machine
//!
//! A runtime state machine with named states, guarded transitions,
//! and a transition history log.
//!
//! ## Primitive Grounding
//!
//! | Symbol | Role | Weight |
//! |--------|------|--------|
//! | ς | State (current position) | 0.35 (dominant) |
//! | → | Causality (transitions) | 0.25 |
//! | ∂ | Boundary (guards) | 0.20 |
//! | κ | Comparison (guard eval) | 0.10 |
//! | π | Persistence (history) | 0.10 |

use crate::guard::{Guard, GuardContext};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for a state machine instance.
pub type MachineId = String;

/// A named state within a machine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StateId(pub String);

impl StateId {
    /// Create a new state ID.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get the state name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for StateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Whether a state is initial, normal, terminal, or error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StateKind {
    /// Entry point of the machine.
    Initial,
    /// Normal intermediate state.
    Normal,
    /// Terminal state — no outgoing transitions.
    Terminal,
    /// Error state — may have recovery transitions.
    Error,
}

/// Definition of a state within a machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDef {
    /// State identifier.
    pub id: StateId,
    /// Kind of state.
    pub kind: StateKind,
    /// Human-readable description.
    pub description: Option<String>,
}

/// A guarded transition between two states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionDef {
    /// Human-readable name for this transition.
    pub name: String,
    /// Source state.
    pub from: StateId,
    /// Target state.
    pub to: StateId,
    /// Guard that must pass for this transition to fire.
    pub guard: Guard,
    /// Priority (lower = higher priority). Used when multiple transitions can fire.
    pub priority: u32,
}

/// A record of a transition that fired.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionRecord {
    /// When the transition fired.
    pub timestamp: DateTime<Utc>,
    /// Name of the transition.
    pub transition_name: String,
    /// Source state.
    pub from: StateId,
    /// Target state.
    pub to: StateId,
    /// Guard description at the time of firing.
    pub guard_description: String,
}

/// Error type for machine operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MachineError {
    /// State not found in the machine.
    StateNotFound(String),
    /// No initial state defined.
    NoInitialState,
    /// Multiple initial states defined.
    MultipleInitialStates,
    /// Transition references a nonexistent state.
    InvalidTransition(String),
    /// Machine is in a terminal state — no transitions possible.
    TerminalState(String),
    /// No transitions can fire from the current state.
    NoValidTransition,
    /// Machine not yet started.
    NotStarted,
}

impl std::fmt::Display for MachineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StateNotFound(s) => write!(f, "state not found: {s}"),
            Self::NoInitialState => write!(f, "no initial state defined"),
            Self::MultipleInitialStates => write!(f, "multiple initial states defined"),
            Self::InvalidTransition(s) => write!(f, "invalid transition: {s}"),
            Self::TerminalState(s) => write!(f, "machine is in terminal state: {s}"),
            Self::NoValidTransition => write!(f, "no valid transition from current state"),
            Self::NotStarted => write!(f, "machine has not been started"),
        }
    }
}

impl std::error::Error for MachineError {}

/// A runtime autonomous state machine.
///
/// Manages states, transitions, and a transition history.
/// The `tick` method evaluates all outgoing guards and fires
/// the highest-priority transition that passes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Machine {
    /// Unique machine ID.
    pub id: MachineId,
    /// Human-readable name.
    pub name: String,
    /// State definitions.
    states: HashMap<StateId, StateDef>,
    /// Transition definitions.
    transitions: Vec<TransitionDef>,
    /// Current state (None if not started).
    current_state: Option<StateId>,
    /// Transition history (most recent last).
    history: Vec<TransitionRecord>,
    /// Maximum history entries to retain.
    max_history: usize,
    /// When the machine was created.
    pub created_at: DateTime<Utc>,
    /// When the machine last transitioned.
    pub last_transition_at: Option<DateTime<Utc>>,
    /// Total number of transitions fired.
    pub transition_count: u64,
}

impl Machine {
    /// Create a new machine with the given ID and name.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            states: HashMap::new(),
            transitions: Vec::new(),
            current_state: None,
            history: Vec::new(),
            max_history: 1000,
            created_at: Utc::now(),
            last_transition_at: None,
            transition_count: 0,
        }
    }

    /// Set maximum history size.
    #[must_use]
    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Add a state definition.
    pub fn add_state(&mut self, id: impl Into<String>, kind: StateKind) -> &mut Self {
        let state_id = StateId::new(id);
        self.states.insert(
            state_id.clone(),
            StateDef {
                id: state_id,
                kind,
                description: None,
            },
        );
        self
    }

    /// Add a state with a description.
    pub fn add_state_with_desc(
        &mut self,
        id: impl Into<String>,
        kind: StateKind,
        description: impl Into<String>,
    ) -> &mut Self {
        let state_id = StateId::new(id);
        self.states.insert(
            state_id.clone(),
            StateDef {
                id: state_id,
                kind,
                description: Some(description.into()),
            },
        );
        self
    }

    /// Add a guarded transition.
    pub fn add_transition(
        &mut self,
        name: impl Into<String>,
        from: impl Into<String>,
        to: impl Into<String>,
        guard: Guard,
    ) -> &mut Self {
        self.transitions.push(TransitionDef {
            name: name.into(),
            from: StateId::new(from),
            to: StateId::new(to),
            guard,
            priority: self.transitions.len() as u32,
        });
        self
    }

    /// Add a guarded transition with explicit priority.
    pub fn add_transition_with_priority(
        &mut self,
        name: impl Into<String>,
        from: impl Into<String>,
        to: impl Into<String>,
        guard: Guard,
        priority: u32,
    ) -> &mut Self {
        self.transitions.push(TransitionDef {
            name: name.into(),
            from: StateId::new(from),
            to: StateId::new(to),
            guard,
            priority,
        });
        self
    }

    /// Validate the machine definition.
    pub fn validate(&self) -> Result<(), MachineError> {
        // Check for initial state
        let initial_count = self
            .states
            .values()
            .filter(|s| s.kind == StateKind::Initial)
            .count();
        if initial_count == 0 {
            return Err(MachineError::NoInitialState);
        }
        if initial_count > 1 {
            return Err(MachineError::MultipleInitialStates);
        }

        // Check transition references
        for t in &self.transitions {
            if !self.states.contains_key(&t.from) {
                return Err(MachineError::InvalidTransition(format!(
                    "'{}' references unknown source state '{}'",
                    t.name, t.from
                )));
            }
            if !self.states.contains_key(&t.to) {
                return Err(MachineError::InvalidTransition(format!(
                    "'{}' references unknown target state '{}'",
                    t.name, t.to
                )));
            }
        }

        Ok(())
    }

    /// Start the machine by moving to the initial state.
    pub fn start(&mut self) -> Result<&StateId, MachineError> {
        self.validate()?;

        let initial = self
            .states
            .values()
            .find(|s| s.kind == StateKind::Initial)
            .ok_or(MachineError::NoInitialState)?;

        self.current_state = Some(initial.id.clone());
        Ok(self
            .current_state
            .as_ref()
            .ok_or(MachineError::NotStarted)?)
    }

    /// Get the current state.
    #[must_use]
    pub fn current_state(&self) -> Option<&StateId> {
        self.current_state.as_ref()
    }

    /// Get the current state definition.
    #[must_use]
    pub fn current_state_def(&self) -> Option<&StateDef> {
        self.current_state
            .as_ref()
            .and_then(|id| self.states.get(id))
    }

    /// Whether the machine is in a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.current_state_def()
            .is_some_and(|s| s.kind == StateKind::Terminal)
    }

    /// Whether the machine has been started.
    #[must_use]
    pub fn is_started(&self) -> bool {
        self.current_state.is_some()
    }

    /// Get the transition history.
    #[must_use]
    pub fn history(&self) -> &[TransitionRecord] {
        &self.history
    }

    /// Get all state definitions.
    #[must_use]
    pub fn states(&self) -> &HashMap<StateId, StateDef> {
        &self.states
    }

    /// Get all transition definitions.
    #[must_use]
    pub fn transitions(&self) -> &[TransitionDef] {
        &self.transitions
    }

    /// Evaluate all outgoing transitions from the current state and
    /// fire the highest-priority one whose guard passes.
    ///
    /// Returns the transition record if a transition fired, or an error
    /// if no transition is valid.
    ///
    /// This is the autonomous tick — the → (Causality) primitive.
    pub fn tick(&mut self, ctx: &GuardContext) -> Result<TransitionRecord, MachineError> {
        let current = self
            .current_state
            .as_ref()
            .ok_or(MachineError::NotStarted)?
            .clone();

        // Check terminal
        if let Some(def) = self.states.get(&current) {
            if def.kind == StateKind::Terminal {
                return Err(MachineError::TerminalState(current.to_string()));
            }
        }

        // Collect and sort outgoing transitions by priority
        let mut candidates: Vec<&TransitionDef> = self
            .transitions
            .iter()
            .filter(|t| t.from == current)
            .collect();
        candidates.sort_by_key(|t| t.priority);

        // Fire the first guard that passes
        for t in candidates {
            if t.guard.evaluate(ctx) {
                let record = TransitionRecord {
                    timestamp: Utc::now(),
                    transition_name: t.name.clone(),
                    from: current.clone(),
                    to: t.to.clone(),
                    guard_description: t.guard.describe(),
                };

                self.current_state = Some(t.to.clone());
                self.last_transition_at = Some(record.timestamp);
                self.transition_count += 1;

                // Append to history with bounded ring
                self.history.push(record.clone());
                while self.history.len() > self.max_history {
                    self.history.remove(0);
                }

                return Ok(record);
            }
        }

        Err(MachineError::NoValidTransition)
    }

    /// Force a transition to a specific state, bypassing guards.
    ///
    /// Used for manual overrides and error recovery.
    pub fn force_transition(
        &mut self,
        target: impl Into<String>,
    ) -> Result<TransitionRecord, MachineError> {
        let current = self
            .current_state
            .as_ref()
            .ok_or(MachineError::NotStarted)?
            .clone();

        let target_id = StateId::new(target);
        if !self.states.contains_key(&target_id) {
            return Err(MachineError::StateNotFound(target_id.to_string()));
        }

        let record = TransitionRecord {
            timestamp: Utc::now(),
            transition_name: format!("force:{}->{}", current, target_id),
            from: current,
            to: target_id.clone(),
            guard_description: "forced (no guard)".to_string(),
        };

        self.current_state = Some(target_id);
        self.last_transition_at = Some(record.timestamp);
        self.transition_count += 1;

        self.history.push(record.clone());
        while self.history.len() > self.max_history {
            self.history.remove(0);
        }

        Ok(record)
    }

    /// Get a summary of the machine's current state.
    #[must_use]
    pub fn summary(&self) -> MachineSummary {
        MachineSummary {
            id: self.id.clone(),
            name: self.name.clone(),
            current_state: self.current_state.as_ref().map(|s| s.0.clone()),
            is_terminal: self.is_terminal(),
            transition_count: self.transition_count,
            state_count: self.states.len(),
            transition_def_count: self.transitions.len(),
            last_transition_at: self.last_transition_at,
            created_at: self.created_at,
        }
    }
}

/// Serializable summary of a machine's status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineSummary {
    /// Machine ID.
    pub id: MachineId,
    /// Machine name.
    pub name: String,
    /// Current state name.
    pub current_state: Option<String>,
    /// Whether in a terminal state.
    pub is_terminal: bool,
    /// Total transitions fired.
    pub transition_count: u64,
    /// Number of defined states.
    pub state_count: usize,
    /// Number of defined transitions.
    pub transition_def_count: usize,
    /// Last transition timestamp.
    pub last_transition_at: Option<DateTime<Utc>>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
}

// ═══════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guard::ComparisonOp;

    fn build_traffic_light() -> Machine {
        let mut m = Machine::new("traffic-light-1", "Traffic Light");
        m.add_state("green", StateKind::Initial);
        m.add_state("yellow", StateKind::Normal);
        m.add_state("red", StateKind::Normal);

        m.add_transition("green->yellow", "green", "yellow", Guard::Always);
        m.add_transition("yellow->red", "yellow", "red", Guard::Always);
        m.add_transition("red->green", "red", "green", Guard::Always);
        m
    }

    #[test]
    fn machine_lifecycle() {
        let mut m = build_traffic_light();

        // Start
        let initial = m.start();
        assert!(initial.is_ok());
        assert_eq!(m.current_state().map(StateId::name), Some("green"));

        // Tick through
        let r1 = m.tick(&GuardContext::new());
        assert!(r1.is_ok());
        assert_eq!(m.current_state().map(StateId::name), Some("yellow"));

        let r2 = m.tick(&GuardContext::new());
        assert!(r2.is_ok());
        assert_eq!(m.current_state().map(StateId::name), Some("red"));

        let r3 = m.tick(&GuardContext::new());
        assert!(r3.is_ok());
        assert_eq!(m.current_state().map(StateId::name), Some("green"));

        assert_eq!(m.transition_count, 3);
        assert_eq!(m.history().len(), 3);
    }

    #[test]
    fn guarded_transitions() {
        let mut m = Machine::new("alert-system", "Alert System");
        m.add_state("normal", StateKind::Initial);
        m.add_state("elevated", StateKind::Normal);
        m.add_state("critical", StateKind::Normal);
        m.add_state("resolved", StateKind::Terminal);

        m.add_transition(
            "escalate",
            "normal",
            "elevated",
            Guard::threshold("error_rate", ComparisonOp::GreaterThan, 0.05),
        );
        m.add_transition(
            "critical",
            "elevated",
            "critical",
            Guard::threshold("error_rate", ComparisonOp::GreaterThan, 0.20),
        );
        m.add_transition(
            "resolve",
            "elevated",
            "resolved",
            Guard::threshold("error_rate", ComparisonOp::LessThan, 0.01),
        );

        let _ = m.start();

        // Low error rate — no transition
        let ctx_low = GuardContext::new().with_metric("error_rate", 0.01);
        assert!(m.tick(&ctx_low).is_err());
        assert_eq!(m.current_state().map(StateId::name), Some("normal"));

        // High error rate — escalate
        let ctx_high = GuardContext::new().with_metric("error_rate", 0.10);
        assert!(m.tick(&ctx_high).is_ok());
        assert_eq!(m.current_state().map(StateId::name), Some("elevated"));

        // Very high — go critical
        let ctx_critical = GuardContext::new().with_metric("error_rate", 0.30);
        assert!(m.tick(&ctx_critical).is_ok());
        assert_eq!(m.current_state().map(StateId::name), Some("critical"));
    }

    #[test]
    fn terminal_state_blocks_tick() {
        let mut m = Machine::new("one-way", "One Way");
        m.add_state("start", StateKind::Initial);
        m.add_state("end", StateKind::Terminal);
        m.add_transition("finish", "start", "end", Guard::Always);

        let _ = m.start();
        assert!(m.tick(&GuardContext::new()).is_ok());
        assert!(m.is_terminal());

        // Can't tick from terminal
        let result = m.tick(&GuardContext::new());
        assert!(matches!(result, Err(MachineError::TerminalState(_))));
    }

    #[test]
    fn force_transition() {
        let mut m = build_traffic_light();
        let _ = m.start();

        let record = m.force_transition("red");
        assert!(record.is_ok());
        assert_eq!(m.current_state().map(StateId::name), Some("red"));
        assert!(
            record
                .as_ref()
                .is_ok_and(|r| r.transition_name.starts_with("force:"))
        );
    }

    #[test]
    fn validation_catches_missing_initial() {
        let mut m = Machine::new("bad", "Bad Machine");
        m.add_state("only", StateKind::Normal);
        assert!(matches!(m.validate(), Err(MachineError::NoInitialState)));
    }

    #[test]
    fn validation_catches_bad_transition_ref() {
        let mut m = Machine::new("bad", "Bad Machine");
        m.add_state("start", StateKind::Initial);
        m.add_transition("broken", "start", "nonexistent", Guard::Always);
        assert!(matches!(
            m.validate(),
            Err(MachineError::InvalidTransition(_))
        ));
    }

    #[test]
    fn priority_ordering() {
        let mut m = Machine::new("priority-test", "Priority Test");
        m.add_state("start", StateKind::Initial);
        m.add_state("low", StateKind::Normal);
        m.add_state("high", StateKind::Normal);

        // Both guards pass, but "to_high" has lower priority number (higher priority)
        m.add_transition_with_priority("to_low", "start", "low", Guard::Always, 10);
        m.add_transition_with_priority("to_high", "start", "high", Guard::Always, 1);

        let _ = m.start();
        let result = m.tick(&GuardContext::new());
        assert!(result.is_ok());
        assert_eq!(m.current_state().map(StateId::name), Some("high"));
    }

    #[test]
    fn machine_summary() {
        let mut m = build_traffic_light();
        let _ = m.start();
        let _ = m.tick(&GuardContext::new());

        let summary = m.summary();
        assert_eq!(summary.id, "traffic-light-1");
        assert_eq!(summary.current_state.as_deref(), Some("yellow"));
        assert_eq!(summary.transition_count, 1);
        assert_eq!(summary.state_count, 3);
        assert!(!summary.is_terminal);
    }

    #[test]
    fn serialization_roundtrip() {
        let mut m = build_traffic_light();
        let _ = m.start();
        let _ = m.tick(&GuardContext::new());

        let json = serde_json::to_string(&m).unwrap_or_default();
        let restored: Machine =
            serde_json::from_str(&json).unwrap_or_else(|_| Machine::new("fallback", "Fallback"));
        assert_eq!(restored.current_state().map(StateId::name), Some("yellow"));
        assert_eq!(restored.transition_count, 1);
    }
}
