#!/usr/bin/env python3.12
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


PLATFORMS = ("5kp", "6kp")
BACKENDS = (
    "cute_sm120_fp8",
    "cute_sm120_mxfp8_32",
    "deep_gemm_mxfp8_32",
)
CC_VALUES = (1, 4, 8, 16, 32, 64, 128)
VALID_ROUNDS = ("r1", "r3")
METRICS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
)
FILE_RE = re.compile(
    r"bench_Qwen3\.5-35B-A3B-FP8_tp1_"
    r"(cute_sm120_fp8|cute_sm120_mxfp8_32|deep_gemm_mxfp8_32)_"
    r"cc(1|4|8|16|32|64|128)_(r1|r3)\.json$"
)
DISPATCH = {
    "cute_sm120_fp8": "Using CUTE_SM120_FP8 Fp8 MoE backend",
    "cute_sm120_mxfp8_32": (
        "Using CUTE_SM120_MXFP8_32 Fp8 MoE backend"
    ),
    "deep_gemm_mxfp8_32": (
        "Using DEEP_GEMM_MXFP8_32 Fp8 MoE backend"
    ),
}
COMMON_IDENTITY_FIELDS = (
    "SIX_KD_COMMIT",
    "SIX_KD_ARCHIVE_SHA256",
    "SIX_KD_CUTLASS_COMMIT",
    "VLLM_COMMIT",
    "VLLM_TREE_ID",
    "DEEPGEMM_VERSION",
    "DEEPGEMM_SOURCE_SHA256",
    "CUTLASS_DSL_VERSION",
    "CUTLASS_DSL_MODULE_PATH",
    "FLASHINFER_BASE_COMMIT",
    "CONTAINER_IMAGE_ID",
    "TORCH_VERSION",
    "TORCH_CUDA_VERSION",
    "PY_SITE_MANIFEST_SHA256",
    "VLLM_BINARY_MANIFEST_SHA256",
    "MODEL_CONFIG_SHA256",
    "MODEL_MANIFEST_SHA256",
    "SOURCE_MANIFEST_SHA256",
)
ARM_IDENTITY_FIELDS = tuple(
    field for field in COMMON_IDENTITY_FIELDS
) + ("DRIVER_VERSION",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_identity(path: Path) -> dict:
    result = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def validate_sha_set(marker: Path, paths: dict) -> None:
    entries = {}
    for line in marker.read_text().splitlines():
        digest, recorded_path = line.split(maxsplit=1)
        name = Path(recorded_path.lstrip("*")).name
        if name in entries:
            raise RuntimeError(f"{marker}: duplicate {name}")
        entries[name] = digest
    if set(entries) != set(paths):
        raise RuntimeError(
            f"{marker}: expected {sorted(paths)}, got {sorted(entries)}"
        )
    for name, path in paths.items():
        if entries[name] != sha256(path):
            raise RuntimeError(f"{marker}: checksum mismatch for {name}")


def validate_json_marker(path: Path) -> None:
    marker = path.with_suffix(path.suffix + ".valid.sha256")
    validate_sha_set(marker, {path.name: path})


def validate_platform_identity(root: Path, platform: str) -> dict:
    identity = read_identity(root / "identity.env")
    required = {
        "RUN_ID",
        "PLATFORM",
        "GPU_UUID",
        "EXPECTED_SM_COUNT",
        "MODEL_CONFIG_SHA256",
        "HARNESS_MANIFEST_SHA256",
        *COMMON_IDENTITY_FIELDS,
    }
    missing = required - set(identity)
    if missing:
        raise RuntimeError(f"{platform}: missing identity fields {missing}")
    if identity["PLATFORM"] != platform:
        raise RuntimeError(
            f"{root}: PLATFORM={identity['PLATFORM']} != {platform}"
        )
    manifest_files = {
        "DEEPGEMM_SOURCE_SHA256": root / "deepgemm_source.sha256",
        "SOURCE_MANIFEST_SHA256": root / "flashinfer_sync.sha256",
        "HARNESS_MANIFEST_SHA256": root / "harness.sha256",
        "MODEL_MANIFEST_SHA256": root / "model_manifest.sha256",
        "PY_SITE_MANIFEST_SHA256": root / "python_site.sha256",
        "VLLM_BINARY_MANIFEST_SHA256": root / "vllm_binary.sha256",
    }
    for field, path in manifest_files.items():
        if sha256(path) != identity[field]:
            raise RuntimeError(f"{platform}: {field} file mismatch")
    correctness = {
        name: root / name
        for name in (
            "correctness_build.log",
            "correctness_binary.sha256",
            "correctness_fp8.log",
            "correctness_mxfp8.log",
        )
    }
    validate_sha_set(root / "correctness_gate.sha256", correctness)
    binary_parts = correctness["correctness_binary.sha256"].read_text(
    ).strip().split(maxsplit=1)
    if (
        len(binary_parts) != 2
        or len(binary_parts[0]) != 64
        or Path(binary_parts[1].lstrip("*")).name != "libth_op.so"
    ):
        raise RuntimeError(f"{platform}: malformed correctness binary identity")
    pass_suffix = (
        f"SIX_KD_COMMIT={identity['SIX_KD_COMMIT']} "
        f"SIX_KD_ARCHIVE_SHA256={identity['SIX_KD_ARCHIVE_SHA256']} "
        f"SIX_KD_CUTLASS_COMMIT={identity['SIX_KD_CUTLASS_COMMIT']} "
        f"TESTED_BINARY_SHA256={binary_parts[0]}"
    )
    pass_markers = {
        "correctness_fp8.log": f"EXP00_CORRECTNESS_FP8_PASS {pass_suffix}",
        "correctness_mxfp8.log": (
            f"EXP00_CORRECTNESS_MXFP8_PASS {pass_suffix}"
        ),
    }
    for name, marker in pass_markers.items():
        if marker not in correctness[name].read_text().splitlines():
            raise RuntimeError(f"{platform}: exact PASS missing from {name}")
    return identity


def validate_arm(
    root: Path,
    platform: str,
    backend: str,
    round_name: str,
    platform_identity: dict,
) -> dict:
    prefix = f"Qwen3.5-35B-A3B-FP8_tp1_{backend}_{round_name}"
    paths = {
        f"identity_{prefix}.env": root / "raw" / f"identity_{prefix}.env",
        f"runtime_{prefix}.env": root / "raw" / f"runtime_{prefix}.env",
        f"evidence_{prefix}.txt": root / "raw" / f"evidence_{prefix}.txt",
        f"binary_{prefix}.sha256": (
            root / "raw" / f"binary_{prefix}.sha256"
        ),
        f"serve_{prefix}.log": root / "raw" / f"serve_{prefix}.log",
        f"engine_config_{prefix}.txt": (
            root / "raw" / f"engine_config_{prefix}.txt"
        ),
    }
    marker = root / "raw" / f"arm_{backend}_{round_name}.complete.sha256"
    validate_sha_set(marker, paths)
    arm_identity = read_identity(paths[f"identity_{prefix}.env"])
    expected = {
        "RUN_ID": platform_identity["RUN_ID"],
        "PLATFORM": platform,
        "BACKEND": backend,
        "ROUND": round_name,
        "ACTUAL_GPU_UUID": platform_identity["GPU_UUID"],
        "GPU_SM_COUNT": platform_identity["EXPECTED_SM_COUNT"],
        "MODEL_CONFIG_SHA256": platform_identity["MODEL_CONFIG_SHA256"],
    }
    expected.update(
        {
            field: platform_identity[field]
            for field in ARM_IDENTITY_FIELDS
        }
    )
    errors = [
        f"{field}={arm_identity.get(field)!r} != {value!r}"
        for field, value in expected.items()
        if arm_identity.get(field) != value
    ]
    if errors:
        raise RuntimeError(
            f"{platform}/{backend}/{round_name}: " + "; ".join(errors)
        )
    binary_path = paths[f"binary_{prefix}.sha256"]
    binary_manifest_sha = sha256(binary_path)
    if arm_identity.get("BACKEND_BINARY_SHA256") != binary_manifest_sha:
        raise RuntimeError(
            f"{platform}/{backend}/{round_name}: binary manifest mismatch"
        )
    binary_lines = binary_path.read_text().splitlines()
    if len(binary_lines) != 1:
        raise RuntimeError(f"{binary_path}: expected one binary")
    elf_sha, elf_name = binary_lines[0].split(maxsplit=1)
    if len(elf_sha) != 64 or not elf_name:
        raise RuntimeError(f"{binary_path}: malformed ELF identity")
    expected_dispatch = DISPATCH[backend]
    for kind in ("serve", "evidence"):
        path = paths[f"{kind}_{prefix}.{'log' if kind == 'serve' else 'txt'}"]
        if expected_dispatch not in path.read_text(errors="replace"):
            raise RuntimeError(
                f"{platform}/{backend}/{round_name}: dispatch missing "
                f"from {path.name}"
            )
    bench_hashes = set(
        re.findall(
            r"^([0-9a-f]{64})\s+\S*/bench_5kp\.sh$",
            paths[f"evidence_{prefix}.txt"].read_text(errors="replace"),
            flags=re.MULTILINE,
        )
    )
    if len(bench_hashes) != 1:
        raise RuntimeError(
            f"{platform}/{backend}/{round_name}: executed bench SHA missing"
        )
    return {
        "identity": arm_identity,
        "binary_manifest_sha256": binary_manifest_sha,
        "elf_sha256": elf_sha,
        "elf_name": elf_name,
        "executed_bench_sha256": bench_hashes.pop(),
        "engine_config": paths[f"engine_config_{prefix}.txt"].read_bytes(),
    }


def validate_record(
    data: dict,
    platform: str,
    backend: str,
    cc: int,
    round_name: str,
    platform_identity: dict,
    arm: dict,
) -> None:
    num_prompts = max(4 * cc, 16)
    expected = {
        "num_prompts": num_prompts,
        "completed": num_prompts,
        "failed": 0,
        "max_concurrency": cc,
        "total_input_tokens": num_prompts * 8000,
        "total_output_tokens": num_prompts * 1000,
        "exp00_run_id": platform_identity["RUN_ID"],
        "exp00_platform": platform,
        "exp00_backend": backend,
        "exp00_cc": str(cc),
        "exp00_round": round_name,
        "exp00_gpu_uuid": platform_identity["GPU_UUID"],
        "exp00_six_kd_commit": platform_identity["SIX_KD_COMMIT"],
        "exp00_six_kd_archive_sha256": platform_identity[
            "SIX_KD_ARCHIVE_SHA256"
        ],
        "exp00_vllm_commit": platform_identity["VLLM_COMMIT"],
        "exp00_vllm_tree_id": platform_identity["VLLM_TREE_ID"],
        "exp00_deepgemm_version": platform_identity["DEEPGEMM_VERSION"],
        "exp00_deepgemm_source_sha256": platform_identity[
            "DEEPGEMM_SOURCE_SHA256"
        ],
        "exp00_model_config_sha256": platform_identity[
            "MODEL_CONFIG_SHA256"
        ],
        "exp00_model_manifest_sha256": platform_identity[
            "MODEL_MANIFEST_SHA256"
        ],
        "exp00_source_manifest_sha256": platform_identity[
            "SOURCE_MANIFEST_SHA256"
        ],
        "exp00_flashinfer_base_commit": platform_identity[
            "FLASHINFER_BASE_COMMIT"
        ],
        "exp00_backend_binary_sha256": arm["binary_manifest_sha256"],
    }
    errors = [
        f"{key}={data.get(key)!r} != {value!r}"
        for key, value in expected.items()
        if data.get(key) != value
    ]
    for field in ("exp00_manifest_sha256",):
        value = data.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            errors.append(f"{field}={value!r} is not a SHA256")
    for metric in METRICS:
        value = data.get(metric)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            errors.append(f"{metric}={value!r} is not finite and positive")
    duration = data.get("duration")
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(duration)
        or duration <= 0
    ):
        errors.append(f"duration={duration!r} is not finite and positive")
    else:
        arithmetic = {
            "request_throughput": num_prompts / duration,
            "output_throughput": num_prompts * 1000 / duration,
            "total_token_throughput": num_prompts * 9000 / duration,
        }
        for metric, expected_value in arithmetic.items():
            if not math.isclose(
                data.get(metric, math.nan),
                expected_value,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                errors.append(
                    f"{metric} is inconsistent with duration and token totals"
                )
    if errors:
        raise RuntimeError(
            f"{platform}/{backend}/cc{cc}/{round_name}: "
            + "; ".join(errors)
        )


def load_platform(platform: str, root: Path) -> tuple:
    platform_identity = validate_platform_identity(root, platform)
    arms = {}
    for backend in BACKENDS:
        for round_name in VALID_ROUNDS:
            arms[(backend, round_name)] = validate_arm(
                root,
                platform,
                backend,
                round_name,
                platform_identity,
            )
    for round_name in VALID_ROUNDS:
        configs = {
            arms[(backend, round_name)]["engine_config"]
            for backend in BACKENDS
        }
        if len(configs) != 1:
            raise RuntimeError(
                f"{platform}/{round_name}: engine config mismatch"
            )
    cute_elf_hashes = {
        arms[(backend, round_name)]["elf_sha256"]
        for backend in ("cute_sm120_fp8", "cute_sm120_mxfp8_32")
        for round_name in VALID_ROUNDS
    }
    if len(cute_elf_hashes) != 1:
        raise RuntimeError(
            f"{platform}: FP8/MXFP8 CuTe ELF mismatch: {cute_elf_hashes}"
        )

    rows = []
    keys = set()
    for path in sorted((root / "raw").glob("bench_*.json")):
        match = FILE_RE.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"{platform}: unexpected JSON {path.name}")
        backend, cc_text, round_name = match.groups()
        cc = int(cc_text)
        key = (backend, cc, round_name)
        if key in keys:
            raise RuntimeError(f"{platform}: duplicate {key}")
        keys.add(key)
        validate_json_marker(path)
        data = json.loads(path.read_text())
        arm = arms[(backend, round_name)]
        validate_record(
            data,
            platform,
            backend,
            cc,
            round_name,
            platform_identity,
            arm,
        )
        row = {
            "platform": platform,
            "backend": backend,
            "cc": cc,
            "round": round_name,
            "num_prompts": data["num_prompts"],
            "manifest_sha256": data["exp00_manifest_sha256"],
            "backend_binary_sha256": arm["binary_manifest_sha256"],
            "backend_elf_sha256": arm["elf_sha256"],
            "gpu_uuid": data["exp00_gpu_uuid"],
        }
        row.update({metric: data[metric] for metric in METRICS})
        rows.append(row)

    expected_keys = {
        (backend, cc, round_name)
        for backend in BACKENDS
        for cc in CC_VALUES
        for round_name in VALID_ROUNDS
    }
    if keys != expected_keys:
        raise RuntimeError(
            f"{platform}: missing={sorted(expected_keys - keys)}, "
            f"extra={sorted(keys - expected_keys)}"
        )
    return rows, platform_identity, arms


def validate_cross_platform(
    rows: list,
    identities: dict,
    arms: dict,
) -> None:
    for field in COMMON_IDENTITY_FIELDS:
        values = {identities[platform][field] for platform in PLATFORMS}
        if len(values) != 1:
            raise RuntimeError(f"{field} mismatch across platforms: {values}")
    for cc in CC_VALUES:
        hashes = {
            row["manifest_sha256"] for row in rows if row["cc"] == cc
        }
        if len(hashes) != 1:
            raise RuntimeError(
                f"cross-platform cc{cc}: request manifest mismatch: {hashes}"
            )
    deep_gemm_elf_hashes = {
        arms[(platform, "deep_gemm_mxfp8_32", round_name)]["elf_sha256"]
        for platform in PLATFORMS
        for round_name in VALID_ROUNDS
    }
    if len(deep_gemm_elf_hashes) != 1:
        raise RuntimeError(
            f"DeepGEMM ELF mismatch across platforms: {deep_gemm_elf_hashes}"
        )
    executed_bench_hashes = {
        arm["executed_bench_sha256"] for arm in arms.values()
    }
    if len(executed_bench_hashes) != 1:
        raise RuntimeError(
            f"executed benchmark harness mismatch: {executed_bench_hashes}"
        )


def validate_request_manifests(
    manifest_root: Path,
    tokenizer_path: Path,
    model_manifest_path: Path,
    rows: list,
) -> list:
    from transformers import AutoTokenizer

    recorded_model_hashes = {}
    for line in model_manifest_path.read_text().splitlines():
        digest, recorded_path = line.split(maxsplit=1)
        recorded_model_hashes[
            Path(recorded_path.lstrip("*")).name
        ] = digest
    tokenizer_artifacts = (
        "config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    for name in tokenizer_artifacts:
        path = tokenizer_path / name
        if (
            name not in recorded_model_hashes
            or sha256(path) != recorded_model_hashes[name]
        ):
            raise RuntimeError(f"tokenizer artifact mismatch: {name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    audit_rows = []
    for cc in CC_VALUES:
        path = manifest_root / f"request_manifest_cc{cc}.jsonl"
        expected_hashes = {
            row["manifest_sha256"] for row in rows if row["cc"] == cc
        }
        if len(expected_hashes) != 1 or sha256(path) not in expected_hashes:
            raise RuntimeError(f"cc{cc}: request manifest digest mismatch")
        expected_requests = max(4 * cc, 16)
        request_count = 0
        for line in path.read_text().splitlines():
            request = json.loads(line)
            if set(request) != {"prompt", "output_tokens"}:
                raise RuntimeError(
                    f"{path}: unexpected request fields {set(request)}"
                )
            if request["output_tokens"] != 1000:
                raise RuntimeError(f"{path}: output_tokens != 1000")
            input_tokens = len(
                tokenizer.encode(
                    request["prompt"],
                    add_special_tokens=False,
                )
            )
            if input_tokens != 8000:
                raise RuntimeError(
                    f"{path}: request {request_count} has "
                    f"{input_tokens} input tokens"
                )
            request_count += 1
        if request_count != expected_requests:
            raise RuntimeError(
                f"{path}: {request_count} requests != {expected_requests}"
            )
        audit_rows.append(
            {
                "cc": cc,
                "request_count": request_count,
                "input_tokens_per_request": 8000,
                "output_tokens_per_request": 1000,
                "manifest_sha256": sha256(path),
            }
        )
    return audit_rows


def write_csv(path: Path, rows: list, fields: list) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    identities = {}
    all_arms = {}
    for platform in PLATFORMS:
        platform_rows, identity, arms = load_platform(
            platform,
            args.results_root / platform,
        )
        rows.extend(platform_rows)
        identities[platform] = identity
        for key, value in arms.items():
            all_arms[(platform, *key)] = value
    validate_cross_platform(rows, identities, all_arms)
    manifest_audit = validate_request_manifests(
        args.manifest_root,
        args.tokenizer_path,
        args.results_root / "5kp" / "model_manifest.sha256",
        rows,
    )

    round_fields = [
        "platform",
        "backend",
        "cc",
        "round",
        "num_prompts",
        *METRICS,
        "manifest_sha256",
        "backend_binary_sha256",
        "backend_elf_sha256",
        "gpu_uuid",
    ]
    rows.sort(
        key=lambda row: (
            PLATFORMS.index(row["platform"]),
            BACKENDS.index(row["backend"]),
            row["cc"],
            VALID_ROUNDS.index(row["round"]),
        )
    )
    write_csv(args.results_root / "benchmark_rounds.csv", rows, round_fields)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["platform"], row["backend"], row["cc"])].append(row)
    median_rows = []
    for (platform, backend, cc), group in grouped.items():
        output_values = [row["output_throughput"] for row in group]
        median_row = {
            "platform": platform,
            "backend": backend,
            "cc": cc,
            "num_prompts": group[0]["num_prompts"],
        }
        for metric in METRICS:
            median_row[metric] = statistics.median(
                row[metric] for row in group
            )
        median_row["output_throughput_cv_pct"] = (
            statistics.stdev(output_values)
            / statistics.mean(output_values)
            * 100.0
        )
        median_rows.append(median_row)
    median_rows.sort(
        key=lambda row: (
            PLATFORMS.index(row["platform"]),
            BACKENDS.index(row["backend"]),
            row["cc"],
        )
    )
    median_fields = [
        "platform",
        "backend",
        "cc",
        "num_prompts",
        *METRICS,
        "output_throughput_cv_pct",
    ]
    write_csv(
        args.results_root / "benchmark_medians.csv",
        median_rows,
        median_fields,
    )

    medians = {
        (row["platform"], row["backend"], row["cc"]): row
        for row in median_rows
    }
    comparison_rows = []
    for platform in PLATFORMS:
        for cc in CC_VALUES:
            fp8 = medians[(platform, "cute_sm120_fp8", cc)]
            cute_mx = medians[(platform, "cute_sm120_mxfp8_32", cc)]
            dg_mx = medians[(platform, "deep_gemm_mxfp8_32", cc)]
            comparison_rows.append(
                {
                    "platform": platform,
                    "cc": cc,
                    "num_prompts": fp8["num_prompts"],
                    "cute_fp8_output_tok_s": fp8["output_throughput"],
                    "cute_mxfp8_32_output_tok_s": cute_mx["output_throughput"],
                    "dg_mxfp8_32_output_tok_s": dg_mx["output_throughput"],
                    "cute_fp8_vs_dg_pct": (
                        fp8["output_throughput"]
                        / dg_mx["output_throughput"]
                        - 1.0
                    )
                    * 100.0,
                    "cute_mxfp8_vs_dg_pct": (
                        cute_mx["output_throughput"]
                        / dg_mx["output_throughput"]
                        - 1.0
                    )
                    * 100.0,
                    "cute_mxfp8_vs_fp8_pct": (
                        cute_mx["output_throughput"]
                        / fp8["output_throughput"]
                        - 1.0
                    )
                    * 100.0,
                    "cute_fp8_cv_pct": fp8["output_throughput_cv_pct"],
                    "cute_mxfp8_cv_pct": cute_mx[
                        "output_throughput_cv_pct"
                    ],
                    "dg_mxfp8_cv_pct": dg_mx[
                        "output_throughput_cv_pct"
                    ],
                }
            )
    write_csv(
        args.results_root / "benchmark.csv",
        comparison_rows,
        list(comparison_rows[0]),
    )
    write_csv(
        args.results_root / "request_manifest_audit.csv",
        manifest_audit,
        list(manifest_audit[0]),
    )

    artifact_rows = []
    for platform in PLATFORMS:
        for backend in BACKENDS:
            artifact_rows.append(
                {
                    "platform": platform,
                    "backend": backend,
                    "backend_elf_sha256": all_arms[
                        (platform, backend, "r1")
                    ]["elf_sha256"],
                    "same_in_r1_r3": (
                        all_arms[(platform, backend, "r1")]["elf_sha256"]
                        == all_arms[(platform, backend, "r3")]["elf_sha256"]
                    ),
                    "executed_bench_sha256": all_arms[
                        (platform, backend, "r1")
                    ]["executed_bench_sha256"],
                    "root_harness_manifest_sha256": identities[platform][
                        "HARNESS_MANIFEST_SHA256"
                    ],
                }
            )
    write_csv(
        args.results_root / "artifact_identity.csv",
        artifact_rows,
        list(artifact_rows[0]),
    )
    print(
        "SUMMARY_DONE "
        f"rounds={len(rows)} medians={len(median_rows)} "
        f"comparisons={len(comparison_rows)} "
        f"manifest_audits={len(manifest_audit)}"
    )


if __name__ == "__main__":
    main()
