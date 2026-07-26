//! Public-GIZMO-compatible individual-particle power-of-two scheduling.
//!
//! This module models the integer-time state machine independently of any
//! hydro operator. Particle bin zero is represented by a zero step and is
//! always active; normal selected steps contain at least two ticks and use bin
//! one through bin 59.

use std::error::Error;
use std::fmt;

use crate::LEGACY_TIMEBASE_TICKS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndividualTimelineError {
    InvalidTimeline,
    InvalidBound,
    InvalidStep,
    BeyondTimelineEnd,
    Finished,
    NoOccupiedBin,
    MismatchedLength { expected: usize, actual: usize },
    MissingActiveBound { index: usize },
    UnexpectedInactiveBound { index: usize },
}

impl fmt::Display for IndividualTimelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTimeline => write!(formatter, "invalid individual-particle timeline"),
            Self::InvalidBound => write!(formatter, "invalid physical timestep bound"),
            Self::InvalidStep => write!(formatter, "invalid integer particle timestep"),
            Self::BeyondTimelineEnd => write!(formatter, "particle step crosses TimeMax"),
            Self::Finished => write!(formatter, "timeline is already finished"),
            Self::NoOccupiedBin => write!(formatter, "timeline has no occupied time bin"),
            Self::MismatchedLength { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected} particle entries, found {actual}"
                )
            }
            Self::MissingActiveBound { index } => {
                write!(formatter, "active particle {index} has no timestep bound")
            }
            Self::UnexpectedInactiveBound { index } => {
                write!(formatter, "inactive particle {index} has a timestep bound")
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
    /// Construct public C's bin-zero state before initial timestep assignment.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid timeline or zero particles.
    pub fn new_unassigned(
        time_begin: f64,
        time_max: f64,
        particle_count: usize,
    ) -> Result<Self, IndividualTimelineError> {
        if !time_begin.is_finite()
            || !time_max.is_finite()
            || time_max <= time_begin
            || particle_count == 0
        {
            return Err(IndividualTimelineError::InvalidTimeline);
        }
        Ok(Self {
            time_begin,
            time_max,
            current_tick: 0,
            time_bins: vec![0; particle_count],
            begin_ticks: vec![0; particle_count],
            step_ticks: vec![0; particle_count],
        })
    }

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
            let ticks = quantize_bound(bound, tick_duration)?;
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
    /// Returns an error unless every step is a power of two in `[2, 2^59]`.
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
            .any(|&step| !(2..LEGACY_TIMEBASE_TICKS).contains(&step) || !step.is_power_of_two())
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
    pub fn duration_for_ticks(&self, ticks: u64) -> f64 {
        self.tick_duration() * ticks as f64
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
            .map(|&step| step == 0 || self.current_tick % step == 0)
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
        if self.step_ticks.contains(&0) {
            return Ok(self.current_tick);
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
    /// Every active particle must have a bound and every inactive particle
    /// must have `None`, matching the dense C active-list traversal.
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
        if self.is_finished() {
            return Err(IndividualTimelineError::Finished);
        }
        let active = self.active_mask();
        let tick_duration = self.tick_duration();
        let remaining = LEGACY_TIMEBASE_TICKS - self.current_tick;
        let mut replacements = Vec::new();
        for (index, (&is_active, bound)) in active.iter().zip(physical_bounds).enumerate() {
            match (is_active, bound) {
                (true, None) => {
                    return Err(IndividualTimelineError::MissingActiveBound { index });
                }
                (false, Some(_)) => {
                    return Err(IndividualTimelineError::UnexpectedInactiveBound { index });
                }
                (false, None) => {}
                (true, Some(bound)) => {
                    let mut step = quantize_bound(*bound, tick_duration)?;
                    while self.current_tick % step != 0 {
                        step >>= 1;
                    }
                    if step > remaining {
                        return Err(IndividualTimelineError::BeyondTimelineEnd);
                    }
                    let begin = self.begin_ticks[index]
                        .checked_add(self.step_ticks[index])
                        .ok_or(IndividualTimelineError::BeyondTimelineEnd)?;
                    replacements.push((index, begin, step, bin_for_step(step)));
                }
            }
        }
        for (index, begin, step, bin) in replacements {
            self.begin_ticks[index] = begin;
            self.step_ticks[index] = step;
            self.time_bins[index] = bin;
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
fn quantize_bound(physical_bound: f64, tick_duration: f64) -> Result<u64, IndividualTimelineError> {
    if !physical_bound.is_finite() || physical_bound <= 0.0 {
        return Err(IndividualTimelineError::InvalidBound);
    }
    let requested = (physical_bound / tick_duration).floor();
    if !requested.is_finite() {
        return Err(IndividualTimelineError::InvalidBound);
    }
    // `get_timestep()` promotes requests of zero or one integer tick to two.
    let integer = (requested as u64).max(2);
    if integer >= LEGACY_TIMEBASE_TICKS {
        return Err(IndividualTimelineError::InvalidStep);
    }
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
    fn bin_zero_is_active_until_initial_assignment() {
        let mut timeline = IndividualParticleTimeline::new_unassigned(0.0, 1.0, 2).unwrap();
        assert_eq!(timeline.time_bins(), &[0, 0]);
        assert_eq!(timeline.step_ticks(), &[0, 0]);
        assert_eq!(timeline.active_mask(), [true, true]);
        assert_eq!(timeline.next_sync_tick().unwrap(), 0);
        timeline
            .reassign_active_bounds(&[Some(physical_ticks(4)), Some(physical_ticks(2))])
            .unwrap();
        assert_eq!(timeline.step_ticks(), &[4, 2]);
        assert_eq!(timeline.begin_ticks(), &[0, 0]);
    }

    #[test]
    fn bin_sixty_is_rejected() {
        assert_eq!(
            IndividualParticleTimeline::from_initial_steps(0.0, 1.0, &[LEGACY_TIMEBASE_TICKS]),
            Err(IndividualTimelineError::InvalidStep)
        );
        assert_eq!(
            IndividualParticleTimeline::from_initial_bounds(0.0, 1.0, &[1.0]),
            Err(IndividualTimelineError::InvalidStep)
        );
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
            Err(IndividualTimelineError::UnexpectedInactiveBound { index: 0 })
        );
    }

    #[test]
    fn failed_reassignment_is_transactional() {
        let mut timeline = IndividualParticleTimeline::from_initial_bounds(
            0.0,
            1.0,
            &[physical_ticks(2), physical_ticks(2)],
        )
        .unwrap();
        let before = timeline.clone();
        assert_eq!(
            timeline.reassign_active_bounds(&[Some(physical_ticks(4)), None]),
            Err(IndividualTimelineError::MissingActiveBound { index: 1 })
        );
        assert_eq!(timeline, before);
    }
}
