"""Count chemistry work independently from score requests and calculations."""

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class MoleculeMetrics(DefaultCallbacks):
    def on_episode_step(self, *, base_env, env_index, episode, **kwargs):
        env = base_env.get_sub_environments()[env_index]
        info = episode.last_info_for() or {}
        counts = getattr(env, "_experiment_counts", None)
        if counts is None:
            counts = env._experiment_counts = {"updates": 0, "valid_molecules": 0, "episodes": 0, "quality_sum": 0.0, "final_smiles": []}
        if info.get("phase") == "select_attach" or ("prev_action" in info and info.get("gen_smiles")):
            counts["updates"] += 1
            counts["valid_molecules"] += 1

    def on_episode_end(self, *, base_env, env_index, episode, **kwargs):
        env = base_env.get_sub_environments()[env_index]
        counts = getattr(env, "_experiment_counts", None)
        if counts is None:
            return
        counts["episodes"] += 1
        counts["quality_sum"] += float(getattr(env, "prev_score", getattr(env, "prev_reward", 0.0)))
        counts["final_smiles"].append(env.base_smiles)
        # Bound diagnostic samples while retaining exact throughput counts.
        counts["final_smiles"] = counts["final_smiles"][-512:]


def _worker_metrics(worker):
    def collect(env):
        result = dict(getattr(env, "_experiment_counts", {}))
        if hasattr(env, "scorer"):
            result.update(env.scorer.stats)
        return result

    return worker.foreach_env(collect)


def collect_metrics(algo):
    results = algo.env_runner_group.foreach_worker(_worker_metrics)
    total = {"updates": 0, "valid_molecules": 0, "episodes": 0, "quality_sum": 0.0, "requests": 0, "computations": 0, "hits": 0}
    smiles = []
    for worker in results:
        for env in worker:
            for key in total:
                total[key] += env.get(key, 0)
            smiles.extend(env.get("final_smiles", []))
    total["unique_final_molecules"] = len(set(smiles))
    total["sampled_final_molecules"] = len(smiles)
    return total


def metric_delta(before, after, seconds):
    result = {key: after[key] - before.get(key, 0) for key in ("updates", "valid_molecules", "episodes", "quality_sum", "requests", "computations", "hits")}
    result["molecule_updates_per_second"] = result["updates"] / seconds
    result["valid_molecules_per_second"] = result["valid_molecules"] / seconds
    result["mean_final_score"] = result["quality_sum"] / result["episodes"] if result["episodes"] else None
    result["sampled_unique_fraction"] = after["unique_final_molecules"] / after["sampled_final_molecules"] if after["sampled_final_molecules"] else None
    return result
