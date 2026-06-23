//! Image augmentation for visual RL (DrQ-v2 style random shift).
//!
//! Provides the [`ImageAugmentation`] trait and concrete implementations
//! for composable, reproducible image augmentations on flat `(B, C, H, W)` arrays.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::RloxError;

/// Trait for composable image augmentations.
///
/// Implementors transform a flat batch of images `(B, C, H, W)` stored as
/// contiguous f32 arrays. The trait enables adding new augmentations
/// (color jitter, cutout, etc.) without modifying existing code.
pub trait ImageAugmentation: Send + Sync {
    /// Apply the augmentation to a batch of images.
    ///
    /// `images` is a flat array of length `batch_size * channels * height * width`.
    /// Returns a new flat array of the same length.
    fn augment_batch(
        &self,
        images: &[f32],
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        seed: u64,
    ) -> Result<Vec<f32>, RloxError>;

    /// Human-readable name for logging/debugging.
    fn name(&self) -> &str;
}

/// DrQ-v2 random shift augmentation.
///
/// Pads the image with zeros, then randomly crops back to original size.
/// Effectively translates the image by up to `pad` pixels in each direction.
pub struct RandomShift {
    pub pad: usize,
}

impl ImageAugmentation for RandomShift {
    fn augment_batch(
        &self,
        images: &[f32],
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        seed: u64,
    ) -> Result<Vec<f32>, RloxError> {
        random_shift_batch(images, batch_size, channels, height, width, self.pad, seed)
    }

    fn name(&self) -> &str {
        "RandomShift"
    }
}

/// Random shift: pad image with zeros, then crop a random (H, W) window.
///
/// For each image in the batch, a random offset `(dy, dx)` is sampled
/// uniformly from `[0, 2*pad]`. The output pixel at `(y, x)` is taken
/// from the padded image at `(y + dy, x + dx)`, which corresponds to
/// the original image pixel at `(y + dy - pad, x + dx - pad)` if in
/// bounds, or zero otherwise.
///
/// # Arguments
/// * `images` - flat `(B * C * H * W)` f32 array
/// * `pad` - number of zero-pad pixels on each side
/// * `seed` - ChaCha8 RNG seed for reproducibility
///
/// # Returns
/// New flat array of same length as input.
#[inline]
pub fn random_shift_batch(
    images: &[f32],
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    pad: usize,
    seed: u64,
) -> Result<Vec<f32>, RloxError> {
    let expected_len = batch_size * channels * height * width;
    if images.len() != expected_len {
        return Err(RloxError::ShapeMismatch {
            expected: format!(
                "B*C*H*W = {}*{}*{}*{} = {}",
                batch_size, channels, height, width, expected_len
            ),
            got: format!("images.len() = {}", images.len()),
        });
    }

    if expected_len == 0 {
        return Ok(Vec::new());
    }

    if pad == 0 {
        return Ok(images.to_vec());
    }

    let img_size = channels * height * width;
    let mut output = vec![0.0f32; expected_len];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    for b in 0..batch_size {
        // Random offset in the padded image
        let dy = rng.random_range(0..=(2 * pad));
        let dx = rng.random_range(0..=(2 * pad));

        let img_offset = b * img_size;

        for c in 0..channels {
            let ch_offset = img_offset + c * height * width;
            for y in 0..height {
                let src_y = y as isize + dy as isize - pad as isize;
                if src_y < 0 || src_y >= height as isize {
                    continue;
                }
                let src_y = src_y as usize;

                // For output x, source is src_x = x + dx - pad.
                // Valid range: 0 <= src_x < width, i.e.:
                //   x >= pad - dx  (lower bound, clamped to 0)
                //   x <  width + pad - dx  (upper bound, clamped to width)
                let x_lo = pad.saturating_sub(dx);
                let x_hi = if dx > pad {
                    width.saturating_sub(dx - pad)
                } else {
                    width
                };

                if x_lo < x_hi {
                    let src_x_start = x_lo + dx - pad;
                    let row_len = x_hi - x_lo;
                    let src_base = ch_offset + src_y * width + src_x_start;
                    let dst_base = ch_offset + y * width + x_lo;
                    output[dst_base..dst_base + row_len]
                        .copy_from_slice(&images[src_base..src_base + row_len]);
                }
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_shift_preserves_shape() {
        let images = vec![1.0f32; 2 * 3 * 8 * 8];
        let output = random_shift_batch(&images, 2, 3, 8, 8, 2, 42).unwrap();
        assert_eq!(output.len(), 2 * 3 * 8 * 8);
    }

    #[test]
    fn test_random_shift_different_seeds_differ() {
        let images: Vec<f32> = (0..2 * 3 * 8 * 8).map(|i| i as f32 / 100.0).collect();
        let a = random_shift_batch(&images, 2, 3, 8, 8, 2, 42).unwrap();
        let b = random_shift_batch(&images, 2, 3, 8, 8, 2, 99).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn test_random_shift_pad_zero_is_identity() {
        let images: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let output = random_shift_batch(&images, 1, 1, 4, 4, 0, 42).unwrap();
        assert_eq!(output, images);
    }

    #[test]
    fn test_random_shift_values_bounded() {
        let images: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let output = random_shift_batch(&images, 1, 1, 4, 4, 2, 42).unwrap();
        for &v in &output {
            assert!((0.0..=1.0).contains(&v), "value out of bounds: {v}");
        }
    }

    #[test]
    fn test_random_shift_single_pixel() {
        let images = vec![5.0f32];
        let output = random_shift_batch(&images, 1, 1, 1, 1, 1, 42).unwrap();
        assert_eq!(output.len(), 1);
        // The single pixel might land or be zero-padded
        assert!(output[0] == 0.0 || output[0] == 5.0);
    }

    #[test]
    fn test_random_shift_large_pad_mostly_zeros() {
        let images = vec![1.0f32; 4];
        let output = random_shift_batch(&images, 1, 1, 2, 2, 10, 42).unwrap();
        let num_zeros = output.iter().filter(|&&v| v == 0.0).count();
        // With pad=10 on a 2x2 image, most positions will be zero
        assert!(
            num_zeros >= 2,
            "expected mostly zeros with large pad, got {num_zeros}/4 zeros"
        );
    }

    #[test]
    fn test_empty_batch_returns_empty() {
        let output = random_shift_batch(&[], 0, 3, 8, 8, 2, 42).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_shape_validation_rejects_mismatched_input() {
        let images = vec![1.0f32; 10]; // wrong length
        let result = random_shift_batch(&images, 2, 3, 8, 8, 2, 42);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, RloxError::ShapeMismatch { .. }),
            "expected ShapeMismatch, got {err:?}"
        );
    }

    #[test]
    fn test_trait_object_safety() {
        let aug: Box<dyn ImageAugmentation> = Box::new(RandomShift { pad: 4 });
        assert_eq!(aug.name(), "RandomShift");
    }

    #[test]
    fn test_random_shift_deterministic_with_same_seed() {
        let images: Vec<f32> = (0..2 * 3 * 8 * 8).map(|i| i as f32 / 100.0).collect();
        let a = random_shift_batch(&images, 2, 3, 8, 8, 2, 42).unwrap();
        let b = random_shift_batch(&images, 2, 3, 8, 8, 2, 42).unwrap();
        assert_eq!(a, b);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_shift_batch_size_preserved(
                b in 1usize..8,
                c in 1usize..4,
                h in 2usize..16,
                w in 2usize..16,
                pad in 0usize..4,
            ) {
                let images = vec![0.5f32; b * c * h * w];
                let output = random_shift_batch(&images, b, c, h, w, pad, 42).unwrap();
                prop_assert_eq!(output.len(), b * c * h * w);
            }

            #[test]
            fn prop_shift_deterministic_with_seed(
                b in 1usize..4,
                c in 1usize..3,
                h in 2usize..8,
                w in 2usize..8,
                seed in 0u64..1000,
            ) {
                let images = vec![1.0f32; b * c * h * w];
                let a = random_shift_batch(&images, b, c, h, w, 2, seed).unwrap();
                let b_out = random_shift_batch(&images, b, c, h, w, 2, seed).unwrap();
                prop_assert_eq!(a, b_out);
            }

            #[test]
            fn prop_shift_values_in_input_range(
                b in 1usize..4,
                c in 1usize..3,
                h in 2usize..8,
                w in 2usize..8,
            ) {
                let n = b * c * h * w;
                let images: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32)).collect();
                let min_val = images.iter().cloned().fold(f32::INFINITY, f32::min);
                let output = random_shift_batch(&images, b, c, h, w, 2, 42).unwrap();
                for &v in &output {
                    // Values are either from the original image or zero (padding)
                    prop_assert!((0.0..=1.0).contains(&v),
                        "value {v} not in [0.0, 1.0], min_val={min_val}");
                }
            }
        }
    }
}
