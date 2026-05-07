use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuDeviceProfile {
    pub global_memory_bytes: u64,
    pub max_alloc_bytes: u64,
    pub compute_units: u32,
    pub max_threads_per_group: u32,
    pub local_memory_bytes: u64,
    pub subgroup_size: u32,
    pub unified_memory: bool,
    pub low_power: bool,
    pub removable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct GpuTuningShape {
    pub batch_size: usize,
    pub by_segment: bool,
    pub precompute_refs: bool,
}

const DISCRETE_GPU_MEMORY_TARGET_PERCENT: u128 = 90;
const DISCRETE_GPU_MAX_ALLOC_TARGET_PERCENT: u128 = 95;

pub fn recommended_gpu_tuning_shapes(
    profile: Option<GpuDeviceProfile>,
    memory_cost_kib: u32,
    parallelism: u32,
) -> Vec<GpuTuningShape> {
    let max_batch = recommended_max_batch(profile, memory_cost_kib, parallelism);
    let anchor = recommended_anchor_batch(profile, max_batch);
    let mut batches = Vec::new();
    for batch_size in powers_of_two_up_to(max_batch) {
        add_batch_candidate(&mut batches, batch_size, max_batch);
    }
    add_batch_candidate(&mut batches, anchor, max_batch);
    add_scaled_batch_candidate(&mut batches, anchor, 3, 4, max_batch);
    add_scaled_batch_candidate(&mut batches, anchor, 3, 2, max_batch);
    add_scaled_batch_candidate(&mut batches, anchor, 5, 2, max_batch);
    add_scaled_batch_candidate(&mut batches, max_batch, 1, 2, max_batch);
    add_scaled_batch_candidate(&mut batches, max_batch, 3, 4, max_batch);
    if let Some(profile) = profile
        && profile.compute_units > 0
    {
        let units = profile.compute_units as usize;
        for candidate in [
            units / 2,
            units,
            units * 2,
            units * 3,
            units * 4,
            units * 6,
            units * 8,
        ] {
            add_batch_candidate(&mut batches, candidate, max_batch);
        }
    }
    batches.sort_unstable();
    batches.dedup();
    batches.sort_by_key(|batch| batch.abs_diff(anchor));

    let mut shapes = Vec::new();
    for batch_size in batches {
        for (by_segment, precompute_refs) in strategy_order(profile) {
            shapes.push(GpuTuningShape {
                batch_size,
                by_segment,
                precompute_refs,
            });
        }
    }
    if shapes.is_empty() {
        shapes.push(GpuTuningShape {
            batch_size: 1,
            by_segment: false,
            precompute_refs: false,
        });
    }
    shapes
}

pub fn estimated_argon2_batch_memory_bytes(memory_cost_kib: u32, batch_size: usize) -> u128 {
    u128::from(memory_cost_kib.max(1)) * 1024 * batch_size.max(1) as u128
}

fn recommended_max_batch(
    profile: Option<GpuDeviceProfile>,
    memory_cost_kib: u32,
    _parallelism: u32,
) -> usize {
    let bytes_per_job = estimated_argon2_batch_memory_bytes(memory_cost_kib, 1);
    let Some(profile) = profile else {
        return 1024;
    };
    let memory_percent = if profile.low_power {
        35
    } else if profile.unified_memory {
        45
    } else {
        DISCRETE_GPU_MEMORY_TARGET_PERCENT
    };
    let alloc_percent = if profile.low_power || profile.unified_memory {
        75
    } else {
        DISCRETE_GPU_MAX_ALLOC_TARGET_PERCENT
    };
    let high_memory_device = profile.global_memory_bytes >= 12 * 1024 * 1024 * 1024
        || profile.max_alloc_bytes >= 8 * 1024 * 1024 * 1024;
    let batch_cap = if profile.low_power {
        512
    } else if profile.unified_memory {
        1024
    } else if high_memory_device && profile.compute_units >= 48 {
        4096
    } else if profile.compute_units >= 32 || profile.max_threads_per_group >= 1024 {
        2048
    } else {
        1024
    };
    let mut usable = if profile.global_memory_bytes > 0 {
        u128::from(profile.global_memory_bytes) * memory_percent / 100
    } else {
        bytes_per_job * 1024
    };
    if profile.max_alloc_bytes > 0 {
        usable = usable.min(u128::from(profile.max_alloc_bytes) * alloc_percent / 100);
    }
    let max_by_memory = (usable / bytes_per_job).clamp(1, batch_cap) as usize;
    max_by_memory.max(1)
}

fn recommended_anchor_batch(profile: Option<GpuDeviceProfile>, max_batch: usize) -> usize {
    let Some(profile) = profile else {
        return max_batch.clamp(1, 64);
    };
    let thread_anchor = if profile.max_threads_per_group > 0 {
        let subgroup = profile.subgroup_size.max(16) as usize;
        (profile.max_threads_per_group as usize / subgroup).max(1)
    } else {
        0
    };
    let compute_anchor = if profile.compute_units > 0 {
        let multiplier = if profile.low_power || profile.unified_memory {
            1
        } else if profile.removable {
            4
        } else {
            2
        };
        (profile.compute_units as usize).saturating_mul(multiplier)
    } else if profile.max_threads_per_group > 0 {
        thread_anchor
    } else {
        64
    };
    compute_anchor.max(thread_anchor).clamp(1, max_batch.max(1))
}

fn powers_of_two_up_to(max_batch: usize) -> Vec<usize> {
    let mut values = Vec::new();
    let mut value = 1usize;
    while value <= max_batch {
        values.push(value);
        value = value.saturating_mul(2);
        if value == 0 {
            break;
        }
    }
    if !values.contains(&max_batch) {
        values.push(max_batch);
    }
    values
}

fn add_batch_candidate(batches: &mut Vec<usize>, candidate: usize, max_batch: usize) {
    if candidate > 0 && candidate <= max_batch {
        batches.push(candidate);
    }
}

fn add_scaled_batch_candidate(
    batches: &mut Vec<usize>,
    base: usize,
    numerator: usize,
    denominator: usize,
    max_batch: usize,
) {
    if denominator == 0 {
        return;
    }
    add_batch_candidate(
        batches,
        base.saturating_mul(numerator) / denominator,
        max_batch,
    );
}

fn strategy_order(profile: Option<GpuDeviceProfile>) -> [(bool, bool); 3] {
    let Some(profile) = profile else {
        return [(true, true), (true, false), (false, false)];
    };
    if profile.low_power
        || (profile.unified_memory && profile.global_memory_bytes <= 8 * 1024 * 1024 * 1024)
        || (profile.local_memory_bytes > 0 && profile.local_memory_bytes < 32 * 1024)
    {
        return [(false, false), (true, false), (true, true)];
    }
    let high_parallel_device = profile.compute_units >= 16 || profile.max_threads_per_group >= 512;
    if high_parallel_device {
        [(true, true), (true, false), (false, false)]
    } else {
        [(false, false), (true, false), (true, true)]
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GpuDeviceProfile, estimated_argon2_batch_memory_bytes, recommended_gpu_tuning_shapes,
    };

    #[test]
    fn recommended_gpu_tuning_shapes_respects_memory_limit() {
        let shapes = recommended_gpu_tuning_shapes(
            Some(GpuDeviceProfile {
                global_memory_bytes: 512 * 1024 * 1024,
                max_alloc_bytes: 256 * 1024 * 1024,
                compute_units: 8,
                max_threads_per_group: 256,
                local_memory_bytes: 32 * 1024,
                subgroup_size: 32,
                unified_memory: false,
                low_power: false,
                removable: false,
            }),
            64 * 1024,
            1,
        );

        assert!(shapes.iter().all(|shape| shape.batch_size <= 3));
        assert!(shapes.iter().any(|shape| shape.batch_size == 1));
    }

    #[test]
    fn recommended_gpu_tuning_shapes_prefers_compute_anchor_first() {
        let shapes = recommended_gpu_tuning_shapes(
            Some(GpuDeviceProfile {
                global_memory_bytes: 24 * 1024 * 1024 * 1024,
                max_alloc_bytes: 24 * 1024 * 1024 * 1024,
                compute_units: 64,
                max_threads_per_group: 1024,
                local_memory_bytes: 64 * 1024,
                subgroup_size: 32,
                unified_memory: false,
                low_power: false,
                removable: false,
            }),
            64 * 1024,
            1,
        );

        assert_eq!(shapes[0].batch_size, 128);
        assert!(shapes[0].by_segment);
        assert!(shapes[0].precompute_refs);
    }

    #[test]
    fn recommended_gpu_tuning_shapes_treats_argon2_memory_as_total_per_job() {
        let shapes = recommended_gpu_tuning_shapes(
            Some(GpuDeviceProfile {
                global_memory_bytes: 24 * 1024 * 1024 * 1024,
                max_alloc_bytes: 24 * 1024 * 1024 * 1024,
                compute_units: 82,
                max_threads_per_group: 1024,
                local_memory_bytes: 64 * 1024,
                subgroup_size: 32,
                unified_memory: false,
                low_power: false,
                removable: false,
            }),
            64 * 1024,
            2,
        );
        let max_batch = shapes
            .iter()
            .map(|shape| shape.batch_size)
            .max()
            .expect("recommended shapes");

        assert!(max_batch >= 240);
        assert_eq!(
            estimated_argon2_batch_memory_bytes(64 * 1024, 16),
            1024 * 1024 * 1024
        );
    }

    #[test]
    fn recommended_gpu_tuning_shapes_keeps_large_discrete_gpu_above_sm_count() {
        let shapes = recommended_gpu_tuning_shapes(
            Some(GpuDeviceProfile {
                global_memory_bytes: 24 * 1024 * 1024 * 1024,
                max_alloc_bytes: 24 * 1024 * 1024 * 1024,
                compute_units: 82,
                max_threads_per_group: 1024,
                local_memory_bytes: 64 * 1024,
                subgroup_size: 32,
                unified_memory: false,
                low_power: false,
                removable: false,
            }),
            256 * 1024,
            1,
        );
        let max_batch = shapes
            .iter()
            .map(|shape| shape.batch_size)
            .max()
            .expect("recommended shapes");

        assert!(max_batch >= 82);
    }

    #[test]
    fn recommended_gpu_tuning_shapes_conserves_low_power_unified_devices() {
        let shapes = recommended_gpu_tuning_shapes(
            Some(GpuDeviceProfile {
                global_memory_bytes: 8 * 1024 * 1024 * 1024,
                max_alloc_bytes: 2 * 1024 * 1024 * 1024,
                compute_units: 16,
                max_threads_per_group: 512,
                local_memory_bytes: 16 * 1024,
                subgroup_size: 32,
                unified_memory: true,
                low_power: true,
                removable: false,
            }),
            64 * 1024,
            1,
        );

        assert!(shapes.iter().all(|shape| shape.batch_size <= 24));
        assert_eq!(shapes[0].batch_size, 16);
        assert!(!shapes[0].by_segment);
        assert!(!shapes[0].precompute_refs);
    }

    #[test]
    fn recommended_gpu_tuning_shapes_adds_refinement_batches() {
        let shapes = recommended_gpu_tuning_shapes(
            Some(GpuDeviceProfile {
                global_memory_bytes: 48 * 1024 * 1024 * 1024,
                max_alloc_bytes: 48 * 1024 * 1024 * 1024,
                compute_units: 96,
                max_threads_per_group: 1024,
                local_memory_bytes: 128 * 1024,
                subgroup_size: 32,
                unified_memory: false,
                low_power: false,
                removable: false,
            }),
            8 * 1024,
            1,
        );
        let batches = shapes
            .iter()
            .map(|shape| shape.batch_size)
            .collect::<std::collections::BTreeSet<_>>();

        assert!(batches.contains(&192));
        assert!(batches.contains(&288));
        assert!(batches.contains(&384));
        assert!(batches.iter().any(|batch| *batch > 2048));
    }
}
