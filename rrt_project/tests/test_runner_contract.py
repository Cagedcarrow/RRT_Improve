from rrt_project.benchmark.runner import CSV_FIELDS


def test_runner_csv_contract_fields():
    must = {
        "ablation_mode",
        "planned_success",
        "sample_source_counts",
        "alpha_trace",
        "alpha_trace_stats",
        "bridge_candidate_stats",
        "cpu_gpu_consistency",
        "hard_validation_passed",
        "hard_validation_collision_idx",
        "image_path",
    }
    assert must.issubset(set(CSV_FIELDS))
