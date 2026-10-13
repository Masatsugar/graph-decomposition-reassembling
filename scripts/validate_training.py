"""Reproducible CUDA correctness, throughput, and real PPO training checks.

Run from the repository root: python -m scripts.validate_training --output tmp/gpu.json
No online experiment tracker is used. This is a short validation run, not a
claim of convergence or a paper benchmark.
"""

import argparse
import copy
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
import torch.nn.functional as F
from guacamol.standard_benchmarks import qed_benchmark
from rdkit import RDLogger

from moldr.chemutils import get_mol
from moldr.config import get_default_config_v1, get_generative_config_v1
from moldr.e2e.config import get_e2e_config
from moldr.e2e.env import MolEnvE2E
from moldr.e2e.model import PointerPolicyNet
from moldr.generative.proposer import VAEProposer
from moldr.generative.vae import train_fragment_vae
from moldr.generative.vocab import FragmentVocab


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def original_forward(net, base, phase, options, mask):
    context = torch.cat((base, phase), -1)
    ctx = net.ctx_enc(context)
    opt = net.opt_enc(options)
    pairs = torch.cat((ctx[:, None].expand_as(opt), opt), -1)
    logits = net.scorer(pairs).squeeze(-1) + torch.log(mask).clamp(min=-1e9)
    return logits, net.vf(context).squeeze(-1)


def pointer_benchmark(n_options, repeats):
    results = {}
    # Full GuacaMol table, default hidden width and PPO minibatch size.
    for device in ("cpu", "cuda"):
        torch.manual_seed(19)
        model = PointerPolicyNet(hidden=256).to(device)
        args = (
            torch.randn(128, 300, device=device),
            torch.tensor([[1.0, 0.0]], device=device).expand(128, -1),
            torch.randn(128, n_options, 300, device=device),
            torch.ones(128, n_options, device=device),
        )
        args[-1][:, -4:] = 0
        with torch.no_grad():
            actual = model(*args)
            expected = original_forward(model, *args)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-5)

        def run(forward, model=model, args=args, device=device):
            model.zero_grad(set_to_none=True)
            logits, value = forward(model, *args)
            loss = F.cross_entropy(
                logits, torch.zeros(128, dtype=torch.long, device=device)
            )
            (loss + value.square().mean()).backward()

        for label, forward in (
            ("original", original_forward),
            ("fixed", lambda net, *a: net(*a)),
        ):
            for _ in range(3):
                run(forward)
            sync(device)
            if device == "cuda":
                model.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(repeats):
                run(forward)
            sync(device)
            results[f"{device}_{label}"] = {
                "forward_backward_ms": (time.perf_counter() - start) * 1000 / repeats,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
                if device == "cuda"
                else None,
            }
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                logits, value = model(*args)
                loss = (
                    F.cross_entropy(
                        logits, torch.zeros(128, dtype=torch.long, device=device)
                    )
                    + value.float().square().mean()
                )
            model.zero_grad(set_to_none=True)
            loss.backward()
            assert torch.isfinite(loss) and all(
                torch.isfinite(p.grad).all() for p in model.parameters()
            )
            assert (torch.softmax(logits, -1)[:, -4:] == 0).all()
            results["cuda_fp16_finite_backward"] = True
        del model, args, actual, expected
    return results


def reconstruction(model, vocab, device):
    with torch.no_grad():
        desc = torch.tensor(vocab.descriptors, dtype=torch.float32, device=device)
        mu, _ = model.encode(desc)
        logits = model.decode(mu, torch.zeros(len(vocab), 300, device=device))
        ids = torch.arange(len(vocab), device=device)
        return {
            "cross_entropy_at_mean": F.cross_entropy(logits, ids).item(),
            "accuracy_at_mean": (logits.argmax(-1) == ids).float().mean().item(),
        }


def vae_benchmark(vocab, epochs, directory):
    results = {}
    for device in ("cpu", "cuda"):
        initial = train_fragment_vae(
            vocab, hidden_dim=256, epochs=0, seed=23, device=device
        )
        before = reconstruction(initial, vocab, device)
        del initial
        sync(device)
        start = time.perf_counter()
        model = train_fragment_vae(
            vocab,
            hidden_dim=256,
            epochs=epochs,
            batch_size=256,
            beta=0.01,
            seed=23,
            device=device,
        )
        sync(device)
        elapsed = time.perf_counter() - start
        after = reconstruction(model, vocab, device)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert after["cross_entropy_at_mean"] < before["cross_entropy_at_mean"]
        results[device] = {
            "seconds": elapsed,
            "fragment_examples_per_second": epochs * len(vocab) / elapsed,
            "before": before,
            "after": after,
        }
        if device == "cuda":
            proposer = VAEProposer(model, vocab, n_blocks=64)
            blocks = proposer.propose()
            assert len(blocks) == 64 and all(get_mol(s) is not None for s in blocks)
            proposer.save(directory)
            restored = VAEProposer.from_spec({"dir": str(directory)})
            np.testing.assert_allclose(
                proposer._logits(None), restored._logits(None), rtol=1e-4, atol=1e-4
            )
            results["cuda_propose_and_cpu_reload"] = True
    return results


def evaluate(algo, config, count=12):
    env = MolEnvE2E(config.env_config)
    scores = {"random": [], "trained": []}
    for mode, mode_scores in scores.items():
        for seed in range(count):
            obs, _ = env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            for _ in range(env.max_episode_steps):
                action = (
                    int(rng.choice(np.flatnonzero(obs["action_mask"])))
                    if mode == "random"
                    else algo.compute_single_action(obs, explore=False)
                )
                obs, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    break
            assert get_mol(env.base_smiles) is not None
            mode_scores.append(float(env.scoring_function(env.base_smiles)))
    return {
        k: {
            "mean_final_qed": float(np.mean(v)),
            "std": float(np.std(v)),
            "episodes": count,
        }
        for k, v in scores.items()
    }


def ppo_run(mode, device, blocks, proposer_dir, output, iterations):
    objective = qed_benchmark()
    common = {
        "model_path": "models/model_300dim.pkl",
        "num_workers": 2,
        "num_gpus": int(device == "cuda"),
        "score_threshold": 1.1,
        "length": 20,
    }
    if mode == "legacy":
        config = get_default_config_v1(objective, blocks, **common)
    elif mode == "generative_legacy":
        config = get_generative_config_v1(
            objective, {"dir": str(proposer_dir)}, **common
        )
    else:
        config = get_e2e_config(
            objective,
            blocks,
            **common,
            option_pool_size=32,
            max_attach=32,
            hidden=128,
            max_episode_steps=40,
            proposer={"dir": str(proposer_dir)} if mode == "generative_e2e" else None,
        )
    config.training(train_batch_size=512, minibatch_size=128, num_epochs=3)
    config.debugging(seed=31, log_level="ERROR")
    config.reporting(min_sample_timesteps_per_iteration=512, min_time_s_per_iteration=0)
    config.env_runners(rollout_fragment_length=128)
    algo = config.build()
    try:
        policy = algo.get_policy()
        actual_device = str(next(policy.model.parameters()).device)
        assert actual_device.startswith(device)
        initial = {
            k: v.detach().cpu().clone() for k, v in policy.model.state_dict().items()
        }
        records = []
        for i in range(iterations):
            sync(device)
            start = time.perf_counter()
            result = algo.train()
            sync(device)
            elapsed = time.perf_counter() - start
            stats = result["info"]["learner"]["default_policy"]["learner_stats"]
            for k in ("total_loss", "policy_loss", "vf_loss", "entropy", "kl"):
                assert np.isfinite(stats[k]), (mode, k, stats)
            assert all(torch.isfinite(p).all() for p in policy.model.parameters())
            record = {
                "iteration": i + 1,
                "seconds": elapsed,
                "sampled_steps": result["num_env_steps_sampled_this_iter"],
                "timers": result.get("timers", {}),
                "learner_stats": stats,
            }
            records.append(record)
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "device": actual_device,
                        "iteration": i + 1,
                        "seconds": elapsed,
                        "loss": stats["total_loss"],
                    }
                ),
                flush=True,
            )
        change = sum(
            (v.detach().cpu() - initial[k]).abs().sum().item()
            for k, v in policy.model.state_dict().items()
        )
        assert change > 0
        run = {
            "model_device": actual_device,
            "weight_absolute_change": change,
            "iterations": records,
        }
        if mode == "e2e" and device == "cuda":
            run["evaluation"] = evaluate(algo, config)
            checkpoint = algo.save(str(output / "ppo_checkpoint"))
            weights = copy.deepcopy(policy.get_weights())
            algo.restore(checkpoint.checkpoint.path)
            for k, v in algo.get_policy().get_weights().items():
                np.testing.assert_array_equal(v, weights[k])
            run["checkpoint_restore_equal"] = True
        return run
    finally:
        algo.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tmp/gpu_validation.json"))
    parser.add_argument("--vae-epochs", type=int, default=50)
    parser.add_argument("--ppo-iterations", type=int, default=3)
    parser.add_argument("--pointer-repeats", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A visible CUDA GPU is required; no CPU fallback for this validation"
        )
    torch.set_num_threads(1)
    RDLogger.DisableLog("rdApp.*")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    directory = args.output.parent.resolve()
    vocab = FragmentVocab.from_smiles_list(
        pd.read_csv("data/building_blocks/guacamol/guacamol_s10000_l2.csv", index_col=0)
        .iloc[:, 0]
        .tolist()
    )
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch_cuda": torch.version.cuda,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "ray", "rdkit", "numpy", "scipy", "pyarrow", "guacamol")
        },
        "vocab_size": len(vocab),
        "settings": vars(args) | {"output": str(args.output)},
    }

    def save():
        args.output.write_text(
            json.dumps(
                report,
                indent=2,
                default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
            )
            + "\n"
        )

    report["pointer"] = pointer_benchmark(len(vocab), args.pointer_repeats)
    print("Pointer correctness and throughput complete", flush=True)
    save()
    report["vae"] = vae_benchmark(vocab, args.vae_epochs, directory / "proposer")
    print("VAE CPU/GPU training and serialization complete", flush=True)
    save()
    report["ppo"] = {}
    ray.init(
        num_cpus=4,
        num_gpus=1,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=512 * 2**20,
        runtime_env={
            "env_vars": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        },
    )
    try:
        for mode, device, iterations in (
            ("e2e", "cpu", args.ppo_iterations),
            ("e2e", "cuda", args.ppo_iterations),
            ("generative_e2e", "cuda", args.ppo_iterations),
            ("legacy", "cuda", 1),
            ("generative_legacy", "cuda", 1),
        ):
            report["ppo"][f"{mode}_{device}"] = ppo_run(
                mode,
                device,
                list(vocab.smiles),
                directory / "proposer",
                directory,
                iterations,
            )
            save()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
