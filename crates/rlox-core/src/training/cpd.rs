// Change-Point Detection (CPD) algorithms for non-stationary RL.
//
// Implements streaming change-point detectors that operate on reward/loss
// signals to detect when the underlying MDP has shifted. These are pure
// data-plane operations (O(1) per observation, no autograd) and belong
// in the Rust data plane per the Polars pattern.

/// Two-sided CUSUM (Cumulative Sum) change-point detector.
///
/// Tracks cumulative deviations from a reference level. When the cumulative
/// sum exceeds a threshold `h`, a change-point alarm is raised.
///
/// # Algorithm
///
/// For each observation x_t:
///   S_t^+ = max(0, S_{t-1}^+ + x_t - mu_0 - delta)  (upward shift)
///   S_t^- = max(0, S_{t-1}^- - x_t + mu_0 - delta)  (downward shift)
///
/// Alarm when S_t^+ > h OR S_t^- > h.
///
/// # Parameters
///
/// - `mu_0`: Reference level (estimated from burn-in or provided)
/// - `delta`: Allowance parameter (minimum shift to detect). Controls sensitivity.
/// - `h`: Decision threshold. Higher h = fewer false alarms, slower detection.
///
/// # Usage in RL
///
/// Feed the detector with a streaming signal (e.g., episode rewards, TD errors,
/// policy loss). When it fires, the environment dynamics or reward function
/// have likely shifted, triggering policy adaptation.
#[derive(Debug, Clone)]
pub struct CusumDetector {
    /// Reference level (target mean under null hypothesis)
    mu_0: f64,
    /// Allowance parameter: minimum shift magnitude to detect
    delta: f64,
    /// Decision threshold: alarm fires when S > h
    h: f64,
    /// Upward CUSUM statistic
    s_pos: f64,
    /// Downward CUSUM statistic
    s_neg: f64,
    /// Total observations processed
    count: u64,
    /// Number of alarms fired
    alarm_count: u64,
    /// Whether to estimate mu_0 from the first `burnin` samples
    burnin: u64,
    /// Sum of burnin samples (for estimating mu_0)
    burnin_sum: f64,
    /// Whether burnin estimation is complete
    burnin_done: bool,
}

impl CusumDetector {
    /// Create a CUSUM detector with a known reference level.
    ///
    /// # Parameters
    /// - `mu_0`: Expected mean of the signal under the null (no-change) hypothesis
    /// - `delta`: Minimum detectable shift (sensitivity). Typical: 0.5 * expected_shift
    /// - `h`: Detection threshold. Typical: 4-8 for moderate false alarm rates
    pub fn new(mu_0: f64, delta: f64, h: f64) -> Self {
        assert!(delta >= 0.0, "delta must be non-negative, got {delta}");
        assert!(h > 0.0, "h must be positive, got {h}");
        Self {
            mu_0,
            delta,
            h,
            s_pos: 0.0,
            s_neg: 0.0,
            count: 0,
            alarm_count: 0,
            burnin: 0,
            burnin_sum: 0.0,
            burnin_done: true,
        }
    }

    /// Create a CUSUM detector that estimates mu_0 from the first `burnin` samples.
    ///
    /// During the burn-in period, no alarms will fire. After burn-in, mu_0 is set
    /// to the mean of the observed samples and detection begins.
    pub fn with_burnin(burnin: u64, delta: f64, h: f64) -> Self {
        assert!(burnin > 0, "burnin must be > 0, got {burnin}");
        assert!(delta >= 0.0, "delta must be non-negative, got {delta}");
        assert!(h > 0.0, "h must be positive, got {h}");
        Self {
            mu_0: 0.0,
            delta,
            h,
            s_pos: 0.0,
            s_neg: 0.0,
            count: 0,
            alarm_count: 0,
            burnin,
            burnin_sum: 0.0,
            burnin_done: false,
        }
    }

    /// Feed one observation. Returns `true` if a change-point alarm fires.
    pub fn update(&mut self, value: f64) -> bool {
        self.count += 1;

        // Handle burn-in phase
        if !self.burnin_done {
            self.burnin_sum += value;
            if self.count >= self.burnin {
                self.mu_0 = self.burnin_sum / self.count as f64;
                self.burnin_done = true;
            }
            return false;
        }

        // Two-sided CUSUM update
        self.s_pos = (self.s_pos + value - self.mu_0 - self.delta).max(0.0);
        self.s_neg = (self.s_neg - value + self.mu_0 - self.delta).max(0.0);

        let alarm = self.s_pos > self.h || self.s_neg > self.h;
        if alarm {
            self.alarm_count += 1;
        }
        alarm
    }

    /// Feed a batch of observations. Returns the index of the first alarm (if any).
    pub fn batch_update(&mut self, values: &[f64]) -> Option<usize> {
        for (i, &v) in values.iter().enumerate() {
            if self.update(v) {
                return Some(i);
            }
        }
        None
    }

    /// Reset the CUSUM statistics without changing parameters.
    ///
    /// Call this after handling an alarm to begin detecting the next change-point.
    /// If burn-in was used, pass the new reference level explicitly, or call
    /// `reset_with_burnin()` to re-estimate from scratch.
    pub fn reset(&mut self) {
        self.s_pos = 0.0;
        self.s_neg = 0.0;
    }

    /// Reset and re-estimate mu_0 from the next `burnin` samples.
    pub fn reset_with_burnin(&mut self) {
        self.s_pos = 0.0;
        self.s_neg = 0.0;
        self.count = 0;
        self.burnin_sum = 0.0;
        self.burnin_done = self.burnin == 0;
    }

    /// Set a new reference level manually (e.g., after adaptation).
    pub fn set_mu(&mut self, mu_0: f64) {
        self.mu_0 = mu_0;
        self.reset();
    }

    /// Current upward CUSUM statistic.
    pub fn s_pos(&self) -> f64 {
        self.s_pos
    }

    /// Current downward CUSUM statistic.
    pub fn s_neg(&self) -> f64 {
        self.s_neg
    }

    /// Reference level.
    pub fn mu_0(&self) -> f64 {
        self.mu_0
    }

    /// Total observations processed.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Total alarms fired.
    pub fn alarm_count(&self) -> u64 {
        self.alarm_count
    }

    /// Whether burn-in is complete.
    pub fn is_ready(&self) -> bool {
        self.burnin_done
    }
}

/// Page-Hinkley change-point detector (one-sided, detects increases).
///
/// Simpler alternative to CUSUM. Tracks the cumulative deviation from
/// the running mean and alarms when it exceeds a threshold.
///
/// # Algorithm
///
/// m_t = sum_{i=1}^t (x_i - mean_t - delta)
/// M_t = min_{1<=i<=t} m_i
/// Alarm when m_t - M_t > lambda
#[derive(Debug, Clone)]
pub struct PageHinkleyDetector {
    delta: f64,
    lambda: f64,
    sum: f64,
    count: u64,
    running_mean: f64,
    cumsum: f64,
    min_cumsum: f64,
    alarm_count: u64,
}

impl PageHinkleyDetector {
    /// Create a Page-Hinkley detector.
    ///
    /// # Parameters
    /// - `delta`: Allowance parameter (tolerance for drift)
    /// - `lambda`: Detection threshold
    pub fn new(delta: f64, lambda: f64) -> Self {
        assert!(lambda > 0.0, "lambda must be positive, got {lambda}");
        Self {
            delta,
            lambda,
            sum: 0.0,
            count: 0,
            running_mean: 0.0,
            cumsum: 0.0,
            min_cumsum: 0.0,
            alarm_count: 0,
        }
    }

    /// Feed one observation. Returns `true` if alarm fires.
    pub fn update(&mut self, value: f64) -> bool {
        self.count += 1;
        self.sum += value;
        self.running_mean = self.sum / self.count as f64;

        self.cumsum += value - self.running_mean - self.delta;
        if self.cumsum < self.min_cumsum {
            self.min_cumsum = self.cumsum;
        }

        let alarm = (self.cumsum - self.min_cumsum) > self.lambda;
        if alarm {
            self.alarm_count += 1;
        }
        alarm
    }

    /// Reset detector state.
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
        self.running_mean = 0.0;
        self.cumsum = 0.0;
        self.min_cumsum = 0.0;
    }

    /// Total observations processed.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Total alarms fired.
    pub fn alarm_count(&self) -> u64 {
        self.alarm_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cusum_no_alarm_on_stationary() {
        let mut det = CusumDetector::new(0.0, 0.5, 5.0);
        // Feed 100 samples from a stationary signal at mean=0
        for _ in 0..100 {
            assert!(!det.update(0.1));
            assert!(!det.update(-0.1));
        }
        assert_eq!(det.alarm_count(), 0);
    }

    #[test]
    fn cusum_detects_upward_shift() {
        let mut det = CusumDetector::new(0.0, 0.5, 5.0);
        // Large upward shift should trigger alarm
        let mut alarm_step = None;
        for i in 0..100 {
            if det.update(3.0) {
                alarm_step = Some(i);
                break;
            }
        }
        assert!(alarm_step.is_some(), "should detect upward shift");
        assert!(alarm_step.unwrap() < 10, "should detect quickly");
    }

    #[test]
    fn cusum_detects_downward_shift() {
        let mut det = CusumDetector::new(0.0, 0.5, 5.0);
        let mut alarm_step = None;
        for i in 0..100 {
            if det.update(-3.0) {
                alarm_step = Some(i);
                break;
            }
        }
        assert!(alarm_step.is_some(), "should detect downward shift");
    }

    #[test]
    fn cusum_burnin_delays_detection() {
        let mut det = CusumDetector::with_burnin(20, 0.5, 5.0);
        // During burnin, no alarms
        for _ in 0..19 {
            assert!(!det.update(100.0)); // even extreme values
        }
        assert!(!det.is_ready()); // Wait, 19 < 20... let me check the logic
                                  // Actually count starts at 1, so after 19 updates count=19 < burnin=20
                                  // After one more:
        assert!(!det.update(100.0)); // count=20, burnin completes, no alarm on this step
        assert!(det.is_ready());
    }

    #[test]
    fn cusum_reset_clears_statistics() {
        let mut det = CusumDetector::new(0.0, 0.5, 5.0);
        for _ in 0..5 {
            det.update(3.0);
        }
        assert!(det.s_pos() > 0.0);
        det.reset();
        assert!((det.s_pos()).abs() < 1e-10);
        assert!((det.s_neg()).abs() < 1e-10);
    }

    #[test]
    fn cusum_batch_update_returns_first_alarm() {
        let mut det = CusumDetector::new(0.0, 0.5, 2.0);
        let values = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let idx = det.batch_update(&values);
        assert!(idx.is_some());
        assert!(idx.unwrap() < 3);
    }

    #[test]
    fn cusum_set_mu_resets_and_updates_reference() {
        let mut det = CusumDetector::new(0.0, 0.5, 5.0);
        for _ in 0..10 {
            det.update(5.0);
        }
        det.set_mu(5.0);
        assert!((det.s_pos()).abs() < 1e-10);
        // Now at the new reference, no alarm
        for _ in 0..50 {
            assert!(!det.update(5.1));
            assert!(!det.update(4.9));
        }
    }

    #[test]
    fn page_hinkley_no_alarm_stationary() {
        let mut det = PageHinkleyDetector::new(0.5, 10.0);
        for _ in 0..100 {
            assert!(!det.update(1.0));
        }
        assert_eq!(det.alarm_count(), 0);
    }

    #[test]
    fn page_hinkley_detects_increase() {
        let mut det = PageHinkleyDetector::new(0.1, 5.0);
        // Stationary phase
        for _ in 0..50 {
            det.update(0.0);
        }
        // Shift up
        let mut detected = false;
        for _ in 0..50 {
            if det.update(5.0) {
                detected = true;
                break;
            }
        }
        assert!(detected, "should detect upward shift");
    }

    #[test]
    fn page_hinkley_reset_clears() {
        let mut det = PageHinkleyDetector::new(0.1, 5.0);
        for _ in 0..10 {
            det.update(10.0);
        }
        det.reset();
        assert_eq!(det.count(), 0);
    }

    #[test]
    #[should_panic(expected = "h must be positive")]
    fn cusum_invalid_h_panics() {
        CusumDetector::new(0.0, 0.5, 0.0);
    }

    #[test]
    #[should_panic(expected = "lambda must be positive")]
    fn page_hinkley_invalid_lambda_panics() {
        PageHinkleyDetector::new(0.1, 0.0);
    }
}
