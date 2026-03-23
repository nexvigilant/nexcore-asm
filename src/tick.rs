//! # Autonomous Tick Engine
//!
//! Manages a fleet of state machines and ticks them all against
//! a shared context. This is the ν (Frequency) + ρ (Recursion) primitive:
//! periodic evaluation of all machines.
//!
//! The `TickEngine` is the autonomic nervous system — it evaluates
//! guards across all registered machines and fires transitions
//! without human intervention.

use crate::guard::GuardContext;
use crate::machine::{Machine, MachineError, MachineId, TransitionRecord};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of a single tick across all machines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickResult {
    /// When the tick was executed.
    pub timestamp: DateTime<Utc>,
    /// Transitions that fired during this tick.
    pub transitions: Vec<TickTransition>,
    /// Machines that had no valid transition.
    pub no_transition: Vec<MachineId>,
    /// Machines that are in terminal state.
    pub terminal: Vec<MachineId>,
    /// Errors encountered during tick.
    pub errors: Vec<TickError>,
    /// Total tick count for this engine.
    pub tick_number: u64,
}

/// A transition that fired during a tick, tagged with its machine ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickTransition {
    /// Machine that transitioned.
    pub machine_id: MachineId,
    /// The transition record.
    pub record: TransitionRecord,
}

/// An error encountered during a tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickError {
    /// Machine that errored.
    pub machine_id: MachineId,
    /// Error description.
    pub error: String,
}

/// The autonomous tick engine.
///
/// Maintains a fleet of state machines and evaluates them all
/// on each `tick()` call. Machines that aren't started are
/// auto-started on first tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickEngine {
    /// Registered machines.
    machines: HashMap<MachineId, Machine>,
    /// Total tick count.
    tick_count: u64,
    /// When the engine was created.
    pub created_at: DateTime<Utc>,
    /// When the last tick occurred.
    pub last_tick_at: Option<DateTime<Utc>>,
}

impl TickEngine {
    /// Create a new tick engine.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machines: HashMap::new(),
            tick_count: 0,
            created_at: Utc::now(),
            last_tick_at: None,
        }
    }

    /// Register a machine with the engine.
    pub fn register(&mut self, machine: Machine) -> Result<(), MachineError> {
        machine.validate()?;
        self.machines.insert(machine.id.clone(), machine);
        Ok(())
    }

    /// Unregister a machine by ID.
    pub fn unregister(&mut self, id: &str) -> Option<Machine> {
        self.machines.remove(id)
    }

    /// Get a machine by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&Machine> {
        self.machines.get(id)
    }

    /// Get a mutable reference to a machine by ID.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Machine> {
        self.machines.get_mut(id)
    }

    /// List all registered machine IDs.
    #[must_use]
    pub fn machine_ids(&self) -> Vec<&MachineId> {
        self.machines.keys().collect()
    }

    /// Number of registered machines.
    #[must_use]
    pub fn machine_count(&self) -> usize {
        self.machines.len()
    }

    /// Total tick count.
    #[must_use]
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Tick all machines against the provided context.
    ///
    /// For each registered machine:
    /// 1. If not started, start it.
    /// 2. If terminal, skip (record in `terminal` list).
    /// 3. Evaluate guards and fire the highest-priority valid transition.
    /// 4. Record all outcomes.
    pub fn tick(&mut self, ctx: &GuardContext) -> TickResult {
        let now = Utc::now();
        self.tick_count += 1;
        self.last_tick_at = Some(now);

        let mut transitions = Vec::new();
        let mut no_transition = Vec::new();
        let mut terminal = Vec::new();
        let mut errors = Vec::new();

        let machine_ids: Vec<MachineId> = self.machines.keys().cloned().collect();

        for id in machine_ids {
            let Some(machine) = self.machines.get_mut(&id) else {
                continue;
            };

            // Auto-start if needed
            if !machine.is_started() {
                if let Err(e) = machine.start() {
                    errors.push(TickError {
                        machine_id: id,
                        error: e.to_string(),
                    });
                    continue;
                }
            }

            // Skip terminal machines
            if machine.is_terminal() {
                terminal.push(id);
                continue;
            }

            // Evaluate
            match machine.tick(ctx) {
                Ok(record) => {
                    transitions.push(TickTransition {
                        machine_id: id,
                        record,
                    });
                }
                Err(MachineError::NoValidTransition) => {
                    no_transition.push(id);
                }
                Err(e) => {
                    errors.push(TickError {
                        machine_id: id,
                        error: e.to_string(),
                    });
                }
            }
        }

        TickResult {
            timestamp: now,
            transitions,
            no_transition,
            terminal,
            errors,
            tick_number: self.tick_count,
        }
    }

    /// Tick a single machine by ID.
    pub fn tick_one(
        &mut self,
        id: &str,
        ctx: &GuardContext,
    ) -> Result<TransitionRecord, MachineError> {
        let machine = self
            .machines
            .get_mut(id)
            .ok_or_else(|| MachineError::StateNotFound(id.to_string()))?;

        if !machine.is_started() {
            machine.start()?;
        }

        machine.tick(ctx)
    }

    /// Get summaries of all registered machines.
    #[must_use]
    pub fn summaries(&self) -> Vec<crate::machine::MachineSummary> {
        self.machines.values().map(Machine::summary).collect()
    }
}

impl Default for TickEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guard::{ComparisonOp, Guard};
    use crate::machine::StateKind;

    fn alert_machine(id: &str) -> Machine {
        let mut m = Machine::new(id, format!("Alert {id}"));
        m.add_state("normal", StateKind::Initial);
        m.add_state("elevated", StateKind::Normal);
        m.add_state("critical", StateKind::Terminal);

        m.add_transition(
            "escalate",
            "normal",
            "elevated",
            Guard::threshold("threat", ComparisonOp::GreaterThan, 0.5),
        );
        m.add_transition(
            "critical",
            "elevated",
            "critical",
            Guard::threshold("threat", ComparisonOp::GreaterThan, 0.9),
        );
        m.add_transition(
            "resolve",
            "elevated",
            "normal",
            Guard::threshold("threat", ComparisonOp::LessThan, 0.2),
        );
        m
    }

    #[test]
    fn tick_engine_basic() {
        let mut engine = TickEngine::new();
        engine.register(alert_machine("alert-1")).ok();
        engine.register(alert_machine("alert-2")).ok();

        assert_eq!(engine.machine_count(), 2);

        // Low threat — no transitions
        let ctx = GuardContext::new().with_metric("threat", 0.1);
        let result = engine.tick(&ctx);
        assert!(result.transitions.is_empty());
        assert_eq!(result.no_transition.len(), 2);

        // High threat — both escalate
        let ctx = GuardContext::new().with_metric("threat", 0.7);
        let result = engine.tick(&ctx);
        assert_eq!(result.transitions.len(), 2);
        for t in &result.transitions {
            assert_eq!(t.record.to.name(), "elevated");
        }
    }

    #[test]
    fn tick_engine_mixed_states() {
        let mut engine = TickEngine::new();
        engine.register(alert_machine("a")).ok();
        engine.register(alert_machine("b")).ok();

        // Escalate both
        let ctx = GuardContext::new().with_metric("threat", 0.7);
        engine.tick(&ctx);

        // Push "a" to critical, resolve "b"
        let ctx = GuardContext::new().with_metric("threat", 0.95);
        let result = engine.tick(&ctx);

        // "a" should transition to critical
        let a_trans = result.transitions.iter().find(|t| t.machine_id == "a");
        assert!(a_trans.is_some_and(|t| t.record.to.name() == "critical"));

        // Next tick — "a" is terminal, "b" might transition or stay
        let ctx = GuardContext::new().with_metric("threat", 0.1);
        let result = engine.tick(&ctx);
        assert!(result.terminal.contains(&"a".to_string()));
    }

    #[test]
    fn tick_engine_auto_start() {
        let mut engine = TickEngine::new();
        let m = alert_machine("auto");
        // Don't start it manually
        engine.register(m).ok();

        let ctx = GuardContext::new().with_metric("threat", 0.7);
        let result = engine.tick(&ctx);

        // Should auto-start and then transition
        assert_eq!(result.transitions.len(), 1);
    }

    #[test]
    fn tick_one() {
        let mut engine = TickEngine::new();
        engine.register(alert_machine("single")).ok();

        let ctx = GuardContext::new().with_metric("threat", 0.7);
        let result = engine.tick_one("single", &ctx);
        assert!(result.is_ok());

        let result = engine.tick_one("nonexistent", &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn summaries() {
        let mut engine = TickEngine::new();
        engine.register(alert_machine("s1")).ok();
        engine.register(alert_machine("s2")).ok();

        let _ = engine.tick(&GuardContext::new().with_metric("threat", 0.7));
        let summaries = engine.summaries();
        assert_eq!(summaries.len(), 2);
    }
}
