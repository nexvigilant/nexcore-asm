//! # Guards — Transition Gate Predicates
//!
//! Guards evaluate whether a transition should fire given the current context.
//! They are the κ (Comparison) primitive applied to state transitions.
//!
//! ## Built-in Guards
//!
//! | Guard | Fires When |
//! |-------|------------|
//! | [`ThresholdGuard`] | A named metric crosses a threshold |
//! | [`AlwaysGuard`] | Unconditionally (for manual/forced transitions) |
//! | [`CompositeGuard`] | All inner guards pass (AND composition) |

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Context provided to guards for evaluation.
///
/// Contains the current metric values and any additional metadata
/// that guards can inspect to decide whether a transition should fire.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GuardContext {
    /// Named metric values (e.g., "error_rate" → 0.05).
    pub metrics: HashMap<String, f64>,
    /// Named flags (e.g., "manual_override" → true).
    pub flags: HashMap<String, bool>,
    /// Named labels (e.g., "current_operator" → "system").
    pub labels: HashMap<String, String>,
}

impl GuardContext {
    /// Create an empty context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a metric value.
    #[must_use]
    pub fn with_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }

    /// Add a flag.
    #[must_use]
    pub fn with_flag(mut self, name: impl Into<String>, value: bool) -> Self {
        self.flags.insert(name.into(), value);
        self
    }

    /// Add a label.
    #[must_use]
    pub fn with_label(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(name.into(), value.into());
        self
    }
}

/// Comparison operator for threshold guards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOp {
    /// Metric > threshold.
    GreaterThan,
    /// Metric >= threshold.
    GreaterOrEqual,
    /// Metric < threshold.
    LessThan,
    /// Metric <= threshold.
    LessOrEqual,
    /// Metric == threshold (within f64 epsilon).
    Equal,
}

impl fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GreaterThan => write!(f, ">"),
            Self::GreaterOrEqual => write!(f, ">="),
            Self::LessThan => write!(f, "<"),
            Self::LessOrEqual => write!(f, "<="),
            Self::Equal => write!(f, "=="),
        }
    }
}

/// A guard that can be evaluated, serialized, and composed.
///
/// Guards are the κ (Comparison) primitive: they compare current state
/// against a condition and return a boolean verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Guard {
    /// Fires when a named metric crosses a threshold.
    Threshold {
        /// Name of the metric to check.
        metric: String,
        /// Comparison operator.
        op: ComparisonOp,
        /// Threshold value.
        threshold: f64,
    },

    /// Fires when a named flag is set to the expected value.
    Flag {
        /// Name of the flag to check.
        flag: String,
        /// Expected value.
        expected: bool,
    },

    /// Always fires. Used for manual/forced transitions.
    Always,

    /// Never fires. Used as a placeholder or to disable a transition.
    Never,

    /// All inner guards must pass (logical AND).
    All {
        /// Guards that must all evaluate to true.
        guards: Vec<Guard>,
    },

    /// At least one inner guard must pass (logical OR).
    Any {
        /// Guards where at least one must evaluate to true.
        guards: Vec<Guard>,
    },
}

impl Guard {
    /// Create a threshold guard.
    #[must_use]
    pub fn threshold(metric: impl Into<String>, op: ComparisonOp, threshold: f64) -> Self {
        Self::Threshold {
            metric: metric.into(),
            op,
            threshold,
        }
    }

    /// Create a flag guard.
    #[must_use]
    pub fn flag(flag: impl Into<String>, expected: bool) -> Self {
        Self::Flag {
            flag: flag.into(),
            expected,
        }
    }

    /// Create an AND-composite guard.
    #[must_use]
    pub fn all(guards: Vec<Guard>) -> Self {
        Self::All { guards }
    }

    /// Create an OR-composite guard.
    #[must_use]
    pub fn any(guards: Vec<Guard>) -> Self {
        Self::Any { guards }
    }

    /// Evaluate this guard against a context.
    ///
    /// Returns `true` if the guard condition is satisfied.
    #[must_use]
    pub fn evaluate(&self, ctx: &GuardContext) -> bool {
        match self {
            Self::Threshold {
                metric,
                op,
                threshold,
            } => {
                let Some(&value) = ctx.metrics.get(metric.as_str()) else {
                    return false;
                };
                match op {
                    ComparisonOp::GreaterThan => value > *threshold,
                    ComparisonOp::GreaterOrEqual => value >= *threshold,
                    ComparisonOp::LessThan => value < *threshold,
                    ComparisonOp::LessOrEqual => value <= *threshold,
                    ComparisonOp::Equal => (value - *threshold).abs() < f64::EPSILON,
                }
            }
            Self::Flag { flag, expected } => {
                ctx.flags.get(flag.as_str()).copied().unwrap_or(false) == *expected
            }
            Self::Always => true,
            Self::Never => false,
            Self::All { guards } => guards.iter().all(|g| g.evaluate(ctx)),
            Self::Any { guards } => guards.iter().any(|g| g.evaluate(ctx)),
        }
    }

    /// Human-readable description of this guard.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Threshold {
                metric,
                op,
                threshold,
            } => format!("{metric} {op} {threshold}"),
            Self::Flag { flag, expected } => format!("{flag} == {expected}"),
            Self::Always => "always".to_string(),
            Self::Never => "never".to_string(),
            Self::All { guards } => {
                let descs: Vec<String> = guards.iter().map(Guard::describe).collect();
                format!("({})", descs.join(" AND "))
            }
            Self::Any { guards } => {
                let descs: Vec<String> = guards.iter().map(Guard::describe).collect();
                format!("({})", descs.join(" OR "))
            }
        }
    }
}

impl fmt::Display for Guard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

// ═══════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_guard_greater_than() {
        let guard = Guard::threshold("error_rate", ComparisonOp::GreaterThan, 0.05);
        let ctx = GuardContext::new().with_metric("error_rate", 0.1);
        assert!(guard.evaluate(&ctx));

        let ctx_low = GuardContext::new().with_metric("error_rate", 0.01);
        assert!(!guard.evaluate(&ctx_low));
    }

    #[test]
    fn threshold_guard_missing_metric_returns_false() {
        let guard = Guard::threshold("cpu", ComparisonOp::GreaterThan, 0.9);
        let ctx = GuardContext::new();
        assert!(!guard.evaluate(&ctx));
    }

    #[test]
    fn flag_guard_evaluates_correctly() {
        let guard = Guard::flag("manual_override", true);

        let ctx_true = GuardContext::new().with_flag("manual_override", true);
        assert!(guard.evaluate(&ctx_true));

        let ctx_false = GuardContext::new().with_flag("manual_override", false);
        assert!(!guard.evaluate(&ctx_false));

        // Missing flag defaults to false
        let ctx_missing = GuardContext::new();
        assert!(!guard.evaluate(&ctx_missing));
    }

    #[test]
    fn always_and_never_guards() {
        let ctx = GuardContext::new();
        assert!(Guard::Always.evaluate(&ctx));
        assert!(!Guard::Never.evaluate(&ctx));
    }

    #[test]
    fn composite_all_guard() {
        let guard = Guard::all(vec![
            Guard::threshold("error_rate", ComparisonOp::GreaterThan, 0.05),
            Guard::threshold("latency", ComparisonOp::GreaterThan, 100.0),
        ]);

        // Both pass
        let ctx = GuardContext::new()
            .with_metric("error_rate", 0.1)
            .with_metric("latency", 200.0);
        assert!(guard.evaluate(&ctx));

        // Only one passes
        let ctx_partial = GuardContext::new()
            .with_metric("error_rate", 0.1)
            .with_metric("latency", 50.0);
        assert!(!guard.evaluate(&ctx_partial));
    }

    #[test]
    fn composite_any_guard() {
        let guard = Guard::any(vec![
            Guard::threshold("error_rate", ComparisonOp::GreaterThan, 0.05),
            Guard::flag("emergency", true),
        ]);

        // Flag fires
        let ctx = GuardContext::new()
            .with_metric("error_rate", 0.01)
            .with_flag("emergency", true);
        assert!(guard.evaluate(&ctx));

        // Neither fires
        let ctx_none = GuardContext::new()
            .with_metric("error_rate", 0.01)
            .with_flag("emergency", false);
        assert!(!guard.evaluate(&ctx_none));
    }

    #[test]
    fn guard_describe() {
        let guard = Guard::threshold("cpu", ComparisonOp::GreaterOrEqual, 0.9);
        assert_eq!(guard.describe(), "cpu >= 0.9");

        let composite = Guard::all(vec![Guard::Always, Guard::flag("ready", true)]);
        assert_eq!(composite.describe(), "(always AND ready == true)");
    }

    #[test]
    fn guard_serialization_roundtrip() {
        let guard = Guard::all(vec![
            Guard::threshold("error_rate", ComparisonOp::GreaterThan, 0.05),
            Guard::flag("enabled", true),
        ]);
        let json = serde_json::to_string(&guard).unwrap_or_default();
        let deserialized: Guard = serde_json::from_str(&json).unwrap_or(Guard::Never);
        let ctx = GuardContext::new()
            .with_metric("error_rate", 0.1)
            .with_flag("enabled", true);
        assert!(deserialized.evaluate(&ctx));
    }
}
