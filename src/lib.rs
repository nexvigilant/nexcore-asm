//! # nexcore-asm — Autonomous State Machine
//!
//! Runtime state machines with guarded transitions, autonomous ticking,
//! and serializable state for cross-session persistence.
//!
//! ## Core Thesis
//!
//! **Autonomous states are states that transition themselves.**
//!
//! The gap between compile-time typestate (which enforces valid transitions)
//! and runtime autonomous behavior (which fires transitions without human
//! intervention) is bridged by three components:
//!
//! 1. **Guards** (`guard.rs`) — κ (Comparison): predicates that evaluate
//!    whether a transition should fire given current context.
//!
//! 2. **Machines** (`machine.rs`) — ς (State) + → (Causality): state
//!    definitions with guarded transitions and transition history.
//!
//! 3. **Tick Engine** (`tick.rs`) — ν (Frequency) + ρ (Recursion): periodic
//!    evaluation of all machines, auto-firing valid transitions.
//!
//! ## Primitive Grounding
//!
//! | Symbol | Role | Component |
//! |--------|------|-----------|
//! | ς | State | `Machine::current_state` |
//! | → | Causality | `Machine::tick` fires transitions |
//! | ∂ | Boundary | `Guard` gates transitions |
//! | κ | Comparison | `Guard::evaluate` |
//! | ν | Frequency | `TickEngine::tick` periodic evaluation |
//! | ρ | Recursion | Engine loops over machines |
//! | π | Persistence | `Machine` is `Serialize`/`Deserialize` |
//! | N | Quantity | Priority ordering, history size |
//!
//! ## Conservation Law
//!
//! `∃ = ∂(×(ς, ∅))` — A machine exists when a boundary (guard)
//! is applied to the product of its current state and its absence
//! (the states it is NOT in). Remove any term and the machine
//! collapses to either "everywhere" (no boundary) or "nowhere"
//! (no state).
//!
//! ## Quick Start
//!
//! ```rust
//! use nexcore_asm::prelude::*;
//!
//! // Define a machine
//! let mut machine = Machine::new("alert-1", "Alert System");
//! machine.add_state("normal", StateKind::Initial);
//! machine.add_state("elevated", StateKind::Normal);
//! machine.add_state("critical", StateKind::Terminal);
//!
//! // Add guarded transitions
//! machine.add_transition(
//!     "escalate", "normal", "elevated",
//!     Guard::threshold("error_rate", ComparisonOp::GreaterThan, 0.05),
//! );
//! machine.add_transition(
//!     "critical_escalation", "elevated", "critical",
//!     Guard::threshold("error_rate", ComparisonOp::GreaterThan, 0.20),
//! );
//!
//! // Start and tick
//! machine.start().unwrap();
//! let ctx = GuardContext::new().with_metric("error_rate", 0.10);
//! let record = machine.tick(&ctx).unwrap();
//! assert_eq!(record.to.name(), "elevated");
//! ```
//!
//! ## Fleet Management
//!
//! ```rust
//! use nexcore_asm::prelude::*;
//!
//! let mut engine = TickEngine::new();
//!
//! // Register multiple machines
//! let mut m1 = Machine::new("m1", "Machine 1");
//! m1.add_state("idle", StateKind::Initial);
//! m1.add_state("active", StateKind::Normal);
//! m1.add_transition("activate", "idle", "active", Guard::Always);
//! engine.register(m1).unwrap();
//!
//! // Tick all machines at once
//! let result = engine.tick(&GuardContext::new());
//! assert_eq!(result.transitions.len(), 1);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![warn(missing_docs)]

extern crate alloc;

pub mod guard;
pub mod machine;
pub mod tick;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::guard::{ComparisonOp, Guard, GuardContext};
    pub use crate::machine::{Machine, MachineError, MachineSummary, StateId, StateKind};
    pub use crate::tick::{TickEngine, TickResult, TickTransition};
}
