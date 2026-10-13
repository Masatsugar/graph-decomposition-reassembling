"""Validate shared-factory PPO, saved distributions and process restoration.

Run ``python -m scripts.validate_experiment --require-cuda`` for honest GPU
verification, or ``--device cpu`` for CPU validation. Every case and every
restoration runs in a separate process; jobs are sequential on a single GPU.
Artifacts include resolved settings, asset hashes, finite learner statistics,
parameter changes, log-probability agreement, memory, and sampled molecules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from moldr.experiment import ExperimentConfig
from moldr.factory import make_environment, make_ppo_config, resolve_config
from moldr.provenance import load_saved_config, save_run, write_json
from moldr.sampling import Sampler

CASES = ("actor", "actor_classic", "actor_value", "auxiliary_only", "mpnn_frozen", "mpnn_trainable", "mpnn_vae", "legacy", "old_e2e")
FRAGMENTS = ["CC", "CCO", "CCN", "CC(=O)O", "C1CC1", "C1CCCCC1", "c1ccccc1", "c1ccncc1"]


def _ray_init(device):
    import ray

    ray.init(
        num_cpus=2, num_gpus=int(device == "cuda"), include_dashboard=False,
        log_to_driver=False, object_store_memory=512 * 2**20,
        _temp_dir=f"/tmp/moldr-validation-ray-{os.getpid()}",
    )


def _model_outputs(policy, observation):
    from ray.rllib.utils.torch_utils import convert_to_torch_tensor

    if isinstance(observation, dict):
        batched = {key: np.expand_dims(value, 0) for key, value in observation.items()}
    else:
        batched = np.expand_dims(observation, 0)
    tensors = convert_to_torch_tensor(batched, device=policy.device)
    with torch.no_grad():
        logits, _ = policy.model({"obs": tensors}, [], None)
        value = policy.model.value_function()
    return logits.cpu().numpy(), value.cpu().numpy()


def _parameter_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _distribution_check(algo):
    """Compare actual saved rollout distributions BEFORE any PPO update."""
    from ray.rllib.policy.sample_batch import SampleBatch
    from ray.rllib.utils.torch_utils import convert_to_torch_tensor

    batch = algo.env_runner_group.local_env_runner.sample()
    if hasattr(batch, "policy_batches"):
        batch = batch.policy_batches["default_policy"]
    policy = algo.get_policy()
    observation = convert_to_torch_tensor(batch[SampleBatch.OBS], device=policy.device)
    actions = torch.as_tensor(batch[SampleBatch.ACTIONS], device=policy.device, dtype=torch.long)
    with torch.no_grad():
        logits, _ = policy.model({"obs": observation}, [], None)
        logp = torch.distributions.Categorical(logits=logits).log_prob(actions).cpu().numpy()
    saved = np.asarray(batch[SampleBatch.ACTION_LOGP])
    saved_logits = np.asarray(batch[SampleBatch.ACTION_DIST_INPUTS])
    np.testing.assert_allclose(logp, saved, atol=2e-5, rtol=1e-5)
    np.testing.assert_allclose(logits.cpu().numpy(), saved_logits, atol=2e-5, rtol=1e-5)
    phase_b = None
    if isinstance(observation, dict):
        valid = observation["action_mask"].gather(1, actions.unsqueeze(1))
        assert torch.all(valid > 0), "Rollout selected a masked action"
        phase_b = int((observation["phase"][:, 1] > .5).sum().item())
        assert phase_b > 0, "The rollout must exercise attachment choices"
    return {"rows": int(batch.count), "phase_b_rows": phase_b,
            "maximum_logp_error": float(np.max(np.abs(logp - saved))),
            "maximum_logits_error": float(np.max(np.abs(logits.cpu().numpy() - saved_logits)))}


def _make_config(case, args, directory):
    config = ExperimentConfig()
    config.fragments.smiles = list(FRAGMENTS)
    config.environment.atom_limit = 20
    config.environment.max_attach = 128
    config.environment.max_episode_steps = 12
    config.environment.score_threshold = 1.1
    config.training.seed = args.seed
    config.training.num_workers = 0
    config.training.num_cpus = 2
    config.training.num_gpus = float(args.device == "cuda")
    config.training.train_batch_size = args.batch_size
    config.training.minibatch_size = args.minibatch_size
    config.training.num_epochs = args.epochs
    config.training.iterations = args.iterations
    config.training.rollout_fragment_length = args.batch_size
    config.output.directory = str(directory)
    if case == "actor_classic":
        config.selection.context = "classic"
    if case in ("actor_value", "mpnn_frozen", "mpnn_trainable", "mpnn_vae"):
        config.selection.mode = "actor_value"
        config.selection.candidate_value = True
    if case == "auxiliary_only":
        config.selection.candidate_value = True
    if case.startswith("mpnn"):
        config.representation.name = "mpnn"
        config.representation.trainable = case != "mpnn_frozen"
        config.representation.gradient_checkpointing = config.representation.trainable
        config.representation.embedding_cache_size = 32 if case == "mpnn_frozen" else 0
    if case == "mpnn_vae":
        from moldr.generative.producer import build_proposer_from_smiles

        proposer = build_proposer_from_smiles(FRAGMENTS, n_blocks=8, hidden_dim=16, epochs=5, seed=args.seed)
        proposer_directory = directory / "source_proposer"
        proposer.save(proposer_directory)
        config.fragments.source = "vae"
        config.fragments.proposer_dir = str(proposer_directory)
        # This tiny proposer is trained without aligned molecular contexts.
        # Its validation therefore uses the supported unconditional baseline.
        config.fragments.recond_every = "never"
        config.fragments.smiles = []
    if case == "legacy":
        config.environment.mode = "legacy"
    return resolve_config(config)


def _old_config(args):
    from guacamol.standard_benchmarks import qed_benchmark

    from moldr.e2e.config import get_e2e_config

    return (get_e2e_config(
        qed_benchmark(), FRAGMENTS, num_workers=0, num_gpus=int(args.device == "cuda"),
        max_attach=128, length=20, max_episode_steps=12, score_threshold=1.1,
    ).training(train_batch_size=args.batch_size, minibatch_size=args.minibatch_size, num_epochs=args.epochs)
            .debugging(seed=args.seed, log_level="ERROR")
            .env_runners(rollout_fragment_length=args.batch_size)
            .reporting(min_sample_timesteps_per_iteration=args.batch_size, min_time_s_per_iteration=0))


def _validate_case(args):
    import ray
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(1)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = None
    if args.worker_case != "old_e2e":
        config, blocks, catalog = _make_config(args.worker_case, args, directory)
        save_run(directory, config, blocks=blocks, catalog=catalog)
        ppo = make_ppo_config(config)
    else:
        ppo = _old_config(args)
    _ray_init(args.device)
    algo = None
    try:
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        algo = ppo.build()
        policy = algo.get_policy()
        device = str(next(policy.model.parameters()).device)
        assert device.startswith(args.device), (device, args.device)
        distribution = _distribution_check(algo)
        before = {name: value.detach().cpu().clone() for name, value in policy.model.named_parameters()}
        records = []
        for iteration in range(args.iterations):
            start = time.perf_counter()
            result = algo.train()
            if args.device == "cuda":
                torch.cuda.synchronize()
            stats = result["info"]["learner"]["default_policy"]["learner_stats"]
            stats = {key: float(stats[key]) for key in ("total_loss", "policy_loss", "vf_loss", "entropy", "kl")}
            assert all(np.isfinite(v) for v in stats.values()), stats
            assert all(torch.isfinite(p).all() for p in policy.model.parameters())
            towers = getattr(policy, "model_gpu_towers", [policy.model])
            gradients = [p.grad for tower in towers for p in tower.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            records.append({"iteration": iteration + 1, "seconds": time.perf_counter() - start,
                            "learner_stats": stats, "finite_gradient_tensors": len(gradients),
                            "model_metrics": {key: float(value) for key, value in policy.model.metrics().items()}})
        changes = {name: float((value.detach().cpu() - before[name]).abs().sum())
                   for name, value in policy.model.named_parameters()}
        assert sum(changes.values()) > 0
        encoder_changes = sum(v for k, v in changes.items() if ".encoder." in k)
        u_changes = sum(v for k, v in changes.items() if "candidate_value_head" in k)
        if config is not None and config.representation.name == "mpnn":
            assert (encoder_changes > 0) == config.representation.trainable
        if config is not None and config.selection.candidate_value:
            assert u_changes > 0, "PPO must update the candidate-value head"
        if args.worker_case == "old_e2e":
            from moldr.e2e.env import MolEnvE2E

            env = MolEnvE2E(ppo.env_config)
        else:
            env = make_environment(config)
        observation, _ = env.reset(seed=123)
        logits, values = _model_outputs(policy, observation)
        probe = observation if isinstance(observation, dict) else {"legacy_observation": observation}
        np.savez(directory / "probe.npz", **probe, expected_logits=logits, expected_values=values)
        if isinstance(observation, dict) and "phase" in observation:
            attachment_observation, _, terminated, _, _ = env.step(0)
            assert not terminated and attachment_observation["phase"][1] > .5
            attachment_logits, attachment_values = _model_outputs(policy, attachment_observation)
            np.savez(directory / "probe_attachment.npz", **attachment_observation,
                     expected_logits=attachment_logits, expected_values=attachment_values)
        sample = Sampler(env, algo).episode(seed=123)
        assert Chem.MolFromSmiles(sample.smiles) is not None
        checkpoint = algo.save(str(directory / "checkpoint")).checkpoint.path
        report = {"case": args.worker_case, "device": device, "checkpoint": str(checkpoint),
                  "validation_settings": {"seed": args.seed, "iterations": args.iterations,
                                          "batch_size": args.batch_size, "minibatch_size": args.minibatch_size,
                                          "epochs": args.epochs},
                  "distribution_before_learning": distribution, "iterations": records,
                  "parameter_absolute_change": sum(changes.values()), "encoder_absolute_change": encoder_changes,
                  "candidate_value_absolute_change": u_changes, "sample": sample.to_dict(),
                  "model_class": type(policy.model).__name__,
                  "parameter_sha256": _parameter_digest(policy.model),
                  "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else None,
                  "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if args.device == "cuda" else None}
        if config is not None and config.environment.mode == "e2e":
            report["observation_schema"] = {
                "version": 2, "context_aggregation": policy.model.net.heads.context_aggregation,
                "graph_record_dtype": str(observation["base_atoms"].dtype) if config.representation.name == "mpnn" else None,
            }
        if args.worker_case == "mpnn_vae":
            assert env.mol2vec is not None and env._proposer.vae.context_dim == 300
            assert observation["candidate_atoms"].shape[0] == config.environment.max_attach
            assert np.all(observation["fragment_ids"][observation["action_mask"] > 0] >= 0)
            report["independent_frozen_vae_encoder"] = True
            report["conditioning_cadence"] = config.fragments.recond_every
            report["stable_vocabulary_ids"] = True
        write_json(directory / "validation.json", report)
        print(json.dumps({"case": args.worker_case, "device": device, "trained": True,
                          "encoder_change": encoder_changes, "candidate_value_change": u_changes}), flush=True)
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


def _restore_case(args):
    import ray
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(1)
    directory = args.output.resolve()
    report = json.loads((directory / "validation.json").read_text())
    _ray_init(args.device)
    algo = None
    try:
        if report["case"] == "old_e2e":
            from ray.rllib.algorithms.algorithm import Algorithm
            from ray.tune.registry import register_env

            from moldr.e2e.config import ENV_NAME, e2e_env_creator
            from moldr.e2e.env import MolEnvE2E
            from moldr.e2e.model import MoldrPointerModel, register_pointer_model

            register_env(ENV_NAME, e2e_env_creator)
            register_pointer_model()
            algo = Algorithm.from_checkpoint(report["checkpoint"])
            assert isinstance(algo.get_policy().model, MoldrPointerModel)
            env = MolEnvE2E(algo.config.env_config)
            restored_by = "Algorithm.from_checkpoint (old moldr_pointer registration)"
        else:
            config, _ = load_saved_config(report["checkpoint"])
            algo = make_ppo_config(config).build()
            algo.restore(report["checkpoint"])
            env = make_environment(config)
            restored_by = "saved config and shared factory"
        probe_paths = [directory / "probe.npz"]
        if (directory / "probe_attachment.npz").is_file():
            probe_paths.append(directory / "probe_attachment.npz")
        exact_logits, exact_values, maximum_logits_error, maximum_value_error = True, True, 0., 0.
        for probe_path in probe_paths:
            with np.load(probe_path) as probe:
                if "legacy_observation" in probe:
                    observation = probe["legacy_observation"]
                else:
                    observation = {key: probe[key] for key in probe.files if key not in {"expected_logits", "expected_values"}}
                logits, values = _model_outputs(algo.get_policy(), observation)
                np.testing.assert_allclose(logits, probe["expected_logits"], atol=2e-5, rtol=1e-5)
                np.testing.assert_allclose(values, probe["expected_values"], atol=2e-5, rtol=1e-5)
                exact_logits &= np.array_equal(logits, probe["expected_logits"])
                exact_values &= np.array_equal(values, probe["expected_values"])
                maximum_logits_error = max(maximum_logits_error, float(np.max(np.abs(logits - probe["expected_logits"]))))
                maximum_value_error = max(maximum_value_error, float(np.max(np.abs(values - probe["expected_values"]))))
        if "parameter_sha256" in report:
            assert _parameter_digest(algo.get_policy().model) == report["parameter_sha256"]
        sample = Sampler(env, algo).episode(seed=123)
        assert Chem.MolFromSmiles(sample.smiles) is not None
        assert sample.smiles == report["sample"]["smiles"]
        assert sample.score == report["sample"]["score"]
        report["separate_process_restore"] = {
            "exact_logits": exact_logits, "exact_state_values": exact_values, "same_seed_sample": True,
            "numerical_logits_and_values": True,
            "exact_weights": True if "parameter_sha256" in report else None,
            "maximum_logits_error": maximum_logits_error, "maximum_value_error": maximum_value_error,
            "probed_phases": len(probe_paths),
            "sample": sample.to_dict(), "loader": restored_by,
        }
        write_json(directory / "validation.json", report)
        print(json.dumps({"case": report["case"], "restored_in_separate_process": True}), flush=True)
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments_validation"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Reuse fully validated cases with identical validation settings")
    parser.add_argument("--worker-case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--restore", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    available = torch.cuda.is_available()
    if args.require_cuda or args.device == "cuda":
        if not available:
            raise RuntimeError("CUDA is required but unavailable; no GPU result will be claimed.")
        args.device = "cuda"
    elif args.device == "auto":
        args.device = "cuda" if available else "cpu"
    if min(args.iterations, args.batch_size, args.minibatch_size, args.epochs) < 1:
        parser.error("iterations, batch sizes, and epochs must be positive")
    if args.minibatch_size > args.batch_size:
        parser.error("minibatch size exceeds batch size")
    if args.restore:
        _restore_case(args)
        return
    if args.worker_case:
        _validate_case(args)
        return
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = {"device": args.device, "cuda_available": available,
              "gpu": torch.cuda.get_device_name(0) if available else None,
              "torch_cuda": torch.version.cuda, "settings": {**vars(args), "output": str(directory)}, "cases": {}}
    write_json(directory / "validation_summary.json", report)
    for case in args.cases:
        case_directory = directory / case
        existing_path = case_directory / "validation.json"
        if args.resume and existing_path.is_file():
            previous = json.loads(existing_path.read_text())
            expected = {"seed": args.seed, "iterations": args.iterations,
                        "batch_size": args.batch_size, "minibatch_size": args.minibatch_size, "epochs": args.epochs}
            # Earlier revisions stored these settings in their resolved config.
            if "validation_settings" not in previous and case != "old_e2e":
                saved, _ = load_saved_config(previous["checkpoint"])
                previous["validation_settings"] = {
                    "seed": saved.training.seed, "iterations": saved.training.iterations,
                    "batch_size": saved.training.train_batch_size,
                    "minibatch_size": saved.training.minibatch_size, "epochs": saved.training.num_epochs,
                }
            if previous.get("separate_process_restore") and previous.get("validation_settings") == expected and previous["device"].startswith(args.device):
                report["cases"][case] = previous
                write_json(directory / "validation_summary.json", report)
                print(f"Reused completed {case} validation", flush=True)
                continue
            raise ValueError(f"Existing incomplete or mismatched validation cannot be resumed: {case_directory}")
        command = [sys.executable, "-m", "scripts.validate_experiment", "--device", args.device,
                   "--output", str(case_directory), "--iterations", str(args.iterations),
                   "--batch-size", str(args.batch_size), "--minibatch-size", str(args.minibatch_size),
                   "--epochs", str(args.epochs), "--seed", str(args.seed)]
        subprocess.run([*command, "--worker-case", case], check=True)
        subprocess.run([*command, "--restore"], check=True)
        report["cases"][case] = json.loads((case_directory / "validation.json").read_text())
        write_json(directory / "validation_summary.json", report)
    print(f"Validated {len(report['cases'])} cases on {args.device}; evidence: {directory}", flush=True)


if __name__ == "__main__":
    main()
