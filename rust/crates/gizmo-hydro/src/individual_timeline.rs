//! Public-GIZMO-compatible individual-particle power-of-two scheduling.
//!
//! This module models the integer-time state machine independently of any
//! hydro operator.  Particle bin zero is only an initialization sentinel:
//! normal selected steps contain at least two ticks and therefore use bin one
//! or higher.

use std::error::Error;
use std::fmt;

use crate::LEGACY_TIMEBASE_TICKS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndividualTimelineError {
    InvalidTimeline,
    InvalidBound,
    InvalidStep,
    NoTwoTickStepRemaining,
    Finished,
    NoOccupiedBin,
    MismatchedLength { expected: usize, actual: usize },
    InactiveParticle { index: usize },
}

impl fmt::Display for IndividualTimelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTimeline => write!(formatter, "invalid individual-particle timeline"),
            Self::InvalidBound => write!(formatter, "invalid physical timestep bound"),
            Self::InvalidStep => write!(formatter, "invalid integer particle timestep"),
            Self::NoTwoTickStepRemaining => {
                write!(formatter, "fewer than two integer-time ticks remain")
            }
            Self::Finished => write!(formatter, "timeline is already finished"),
            Self::NoOccupiedBin => write!(formatter, "timeline has no occupied time bin"),
            Self::MismatchedLength { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected} particle entries, found {actual}"
                )
            }
            Self::InactiveParticle { index } => {
                write!(
                    formatter,
                    "particle {index} is not active at the current tick"
                )
            }
        }
    }
}

impl Error for IndividualTimelineError {}

/// Integer schedule retained for each particle by the public C integrator.
#[derive(Clone, Debug, PartialEq)]
pub struct IndividualParticleTimeline {
    time_begin: f64,
    time_max: f64,
    current_tick: u64,
    time_bins: Vec<u8>,
    begin_ticks: Vec<u64>,
    step_ticks: Vec<u64>,
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
impl IndividualParticleTimeline {
    /// Assign the initial particle bins from independent physical bounds.
    ///
    /// Like `get_timestep()` followed by `find_timesteps()`, each bound is
    /// truncated to integer ticks and then rounded down to a power of two.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid timeline/bound or a bound below the
    /// public code's timeline.
    pub fn from_initial_bounds(
        time_begin: f64,
        time_max: f64,
        physical_bounds: &[f64],
    ) -> Result<Self, IndividualTimelineError> {
        if !time_begin.is_finite()
            || !time_max.is_finite()
            || time_max <= time_begin
            || physical_bounds.is_empty()
        {
            return Err(IndividualTimelineError::InvalidTimeline);
        }
        let tick_duration = (time_max - time_begin) / LEGACY_TIMEBASE_TICKS as f64;
        let mut time_bins = Vec::with_capacity(physical_bounds.len());
        let mut step_ticks = Vec::with_capacity(physical_bounds.len());
        for &bound in physical_bounds {
            let ticks = quantize_bound(bound, tick_duration, LEGACY_TIMEBASE_TICKS)?;
            time_bins.push(bin_for_step(ticks));
            step_ticks.push(ticks);
        }
        Ok(Self {
            time_begin,
            time_max,
            current_tick: 0,
            begin_ticks: vec![0; physical_bounds.len()],
            time_bins,
            step_ticks,
        })
    }

    /// Construct a schedule from already capped and quantized C time steps.
    ///
    /// This is the lossless bridge for physics-specific timestep selection:
    /// callers can preserve the raw/bounded diagnostics while this type owns
    /// only the event schedule.
    ///
    /// # Errors
    ///
    /// Returns an error unless every step is a power of two in `[2, 2^60]`.
    pub fn from_initial_steps(
        time_begin: f64,
        time_max: f64,
        steps: &[u64],
    ) -> Result<Self, IndividualTimelineError> {
        if !time_begin.is_finite()
            || !time_max.is_finite()
            || time_max <= time_begin
            || steps.is_empty()
        {
            return Err(IndividualTimelineError::InvalidTimeline);
        }
        if steps
            .iter()
            .any(|&step| !(2..=LEGACY_TIMEBASE_TICKS).contains(&step) || !step.is_power_of_two())
        {
            return Err(IndividualTimelineError::InvalidStep);
        }
        Ok(Self {
            time_begin,
            time_max,
            current_tick: 0,
            time_bins: steps.iter().map(|&step| bin_for_step(step)).collect(),
            begin_ticks: vec![0; steps.len()],
            step_ticks: steps.to_vec(),
        })
    }

    #[must_use]
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    #[must_use]
    pub fn current_time(&self) -> f64 {
        self.time_begin + self.tick_duration() * self.current_tick as f64
    }

    #[must_use]
    pub fn time_bins(&self) -> &[u8] {
        &self.time_bins
    }

    #[must_use]
    pub fn begin_ticks(&self) -> &[u64] {
        &self.begin_ticks
    }

    #[must_use]
    pub fn step_ticks(&self) -> &[u64] {
        &self.step_ticks
    }

    #[must_use]
    pub fn active_mask(&self) -> Vec<bool> {
        self.step_ticks
            .iter()
            .map(|&step| self.current_tick % step == 0)
            .collect()
    }

    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.current_tick >= LEGACY_TIMEBASE_TICKS
    }

    /// Return the next global synchronization point (the earliest occupied
    /// particle-bin endpoint).
    ///
    /// # Errors
    ///
    /// Returns an error at the end of the timeline or for an empty schedule.
    pub fn next_sync_tick(&self) -> Result<u64, IndividualTimelineError> {
        if self.is_finished() {
            return Err(IndividualTimelineError::Finished);
        }
        self.step_ticks
            .iter()
            .map(|&step| {
                let quotient = self.current_tick / step;
                (quotient + 1) * step
            })
            .filter(|&tick| tick <= LEGACY_TIMEBASE_TICKS)
            .min()
            .ok_or(IndividualTimelineError::NoOccupiedBin)
    }

    /// Move to the next global synchronization point and return its active set.
    ///
    /// # Errors
    ///
    /// Returns an error if no future occupied-bin endpoint exists.
    pub fn advance_to_next_sync(&mut self) -> Result<Vec<bool>, IndividualTimelineError> {
        self.current_tick = self.next_sync_tick()?;
        Ok(self.active_mask())
    }

    /// Reassign the steps of the particles active at the current tick.
    ///
    /// A step may grow only into a bin synchronized at this tick. This is the
    /// `TimeBinActive[bin]` growth restriction from `find_timesteps()`.
    /// Inactive particles are rejected instead of silently changing schedule.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched inputs, invalid bounds, or an inactive
    /// selected particle.
    pub fn reassign_active_bounds(
        &mut self,
        physical_bounds: &[Option<f64>],
    ) -> Result<(), IndividualTimelineError> {
        if physical_bounds.len() != self.step_ticks.len() {
            return Err(IndividualTimelineError::MismatchedLength {
                expected: self.step_ticks.len(),
                actual: physical_bounds.len(),
            });
        }
        let remaining = LEGACY_TIMEBASE_TICKS - self.current_tick;
        let tick_duration = self.tick_duration();
        for (index, bound) in physical_bounds.iter().enumerate() {
            let Some(bound) = bound else {
                continue;
            };
            if self.current_tick % self.step_ticks[index] != 0 {
                return Err(IndividualTimelineError::InactiveParticle { index });
            }
            let mut step = quantize_bound(*bound, tick_duration, remaining)?;
            while self.current_tick % step != 0 {
                step >>= 1;
            }
            self.begin_ticks[index] = self.current_tick;
            self.step_ticks[index] = step;
            self.time_bins[index] = bin_for_step(step);
        }
        Ok(())
    }

    fn tick_duration(&self) -> f64 {
        (self.time_max - self.time_begin) / LEGACY_TIMEBASE_TICKS as f64
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn quantize_bound(
    physical_bound: f64,
    tick_duration: f64,
    maximum_ticks: u64,
) -> Result<u64, IndividualTimelineError> {
    if !physical_bound.is_finite() || physical_bound <= 0.0 {
        return Err(IndividualTimelineError::InvalidBound);
    }
    let requested = (physical_bound / tick_duration).floor();
    if !requested.is_finite() {
        return Err(IndividualTimelineError::InvalidBound);
    }
    if maximum_ticks < 2 {
        return Err(IndividualTimelineError::NoTwoTickStepRemaining);
    }
    // `get_timestep()` promotes requests of zero or one integer tick to two.
    let integer = (requested as u64).max(2).min(maximum_ticks);
    let next_power = integer.next_power_of_two();
    Ok(if next_power > integer {
        next_power >> 1
    } else {
        next_power
    })
}

fn bin_for_step(step: u64) -> u8 {
    debug_assert!(step.is_power_of_two());
    u8::try_from(step.trailing_zeros()).expect("u64 has at most 64 trailing zeroes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::cast_precision_loss)]
    fn physical_ticks(ticks: u64) -> f64 {
        ticks as f64 / LEGACY_TIMEBASE_TICKS as f64
    }

    #[test]
    fn initial_bounds_are_independently_truncated_and_power_of_two_floored() {
        let timeline = IndividualParticleTimeline::from_initial_bounds(
            0.0,
            1.0,
            &[physical_ticks(9), physical_ticks(4), physical_ticks(2)],
        )
        .unwrap();
        assert_eq!(timeline.step_ticks(), &[8, 4, 2]);
        assert_eq!(timeline.time_bins(), &[3, 2, 1]);
        assert_eq!(timeline.active_mask(), [true, true, true]);
    }

    #[test]
    fn sub_tick_request_is_promoted_to_public_two_tick_minimum() {
        let timeline =
            IndividualParticleTimeline::from_initial_bounds(0.0, 1.0, &[physical_ticks(1) * 0.5])
                .unwrap();
        assert_eq!(timeline.step_ticks(), &[2]);
        assert_eq!(timeline.time_bins(), &[1]);
    }

    #[test]
    fn next_sync_activates_only_divisible_bins() {
        let mut timeline = IndividualParticleTimeline::from_initial_bounds(
            0.0,
            1.0,
            &[physical_ticks(8), physical_ticks(4), physical_ticks(2)],
        )
        .unwrap();
        assert_eq!(
            timeline.advance_to_next_sync().unwrap(),
            [false, false, true]
        );
        assert_eq!(timeline.current_tick(), 2);
        assert_eq!(
            timeline.advance_to_next_sync().unwrap(),
            [false, true, true]
        );
        assert_eq!(timeline.current_tick(), 4);
    }

    #[test]
    fn step_growth_is_reduced_to_a_bin_synchronized_now() {
        let mut timeline = IndividualParticleTimeline::from_initial_bounds(
            0.0,
            1.0,
            &[physical_ticks(8), physical_ticks(2)],
        )
        .unwrap();
        timeline.advance_to_next_sync().unwrap();
        timeline
            .reassign_active_bounds(&[None, Some(physical_ticks(8))])
            .unwrap();
        assert_eq!(timeline.step_ticks(), &[8, 2]);
        assert_eq!(timeline.begin_ticks(), &[0, 2]);
    }

    #[test]
    fn inactive_particle_cannot_be_reassigned() {
        let mut timeline = IndividualParticleTimeline::from_initial_bounds(
            0.0,
            1.0,
            &[physical_ticks(8), physical_ticks(2)],
        )
        .unwrap();
        timeline.advance_to_next_sync().unwrap();
        assert_eq!(
            timeline.reassign_active_bounds(&[Some(physical_ticks(4)), None]),
            Err(IndividualTimelineError::InactiveParticle { index: 0 })
        );
    }
}
