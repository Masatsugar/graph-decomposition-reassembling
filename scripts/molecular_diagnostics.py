"""Optional evaluation capture; chemical analysis runs only when writing.

Wrap a bound sampler method, or patch ``Sampler.episode`` with the unbound
form. Capture keeps scalar counters, chosen molecules and policy entropy from
the existing action forward. It never scores or generates an extra candidate.
``write`` performs RDKit analysis and optional, separately counted cap audits.
"""

from __future__ import annotations

import inspect
import json
import math
import time
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import numpy as np


@contextmanager
def _replace_method(owner, name, replacement):
    """Restore the exact previous instance/class lookup after evaluation."""
    namespace = getattr(owner, "__dict__", {})
    existed, previous = name in namespace, namespace.get(name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        if existed:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)


def _supports_keyword(function, key):
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return key in parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


def _number(value):
    return None if value is None else float(value)


def _phase(observation):
    if isinstance(observation, Mapping) and "phase" in observation:
        phase = np.asarray(observation["phase"]).reshape(-1)
        if len(phase) == 2:
            return "select_attach" if phase[1] > 0.5 else "select_bb"
    return "legacy"


def _policy_entropy(observation, output):
    if (
        not isinstance(output, tuple)
        or len(output) < 3
        or not isinstance(output[2], Mapping)
    ):
        return None
    logits = output[2].get("action_dist_inputs")
    if logits is None:
        return None
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if isinstance(observation, Mapping) and "action_mask" in observation:
        mask = np.asarray(observation["action_mask"]).reshape(-1) > 0
        if len(mask) != len(logits):
            return None
        logits = logits[mask]
    count = len(logits)
    if not count or not np.isfinite(logits).all():
        return None
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    positive = probabilities > 0
    entropy = float(-(probabilities[positive] * np.log(probabilities[positive])).sum())
    return {
        "phase": _phase(observation),
        "valid_actions": count,
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(count) if count > 1 else None,
        "single_action": count == 1,
        "maximum_probability": float(probabilities.max()),
    }


class MolecularDiagnostics:
    """Collect one checkpoint's evaluation without changing its decisions.

    ``records`` remains a lightweight JSON-serializable trajectory summary.
    Entropy describes softmax(action_dist_inputs), including during greedy
    evaluation; it is not the entropy of deterministic argmax outputs. Explicit
    full-fetch value-preview calls are excluded. Unsupported policies report
    unavailable entropy instead of running another forward.
    """

    def __init__(self, env=None, *, cap_audit_limit=0):
        if (
            isinstance(cap_audit_limit, bool)
            or not isinstance(cap_audit_limit, int)
            or cap_audit_limit < 0
        ):
            raise ValueError("cap_audit_limit must be a nonnegative integer")
        self.env = env
        self.cap_audit_limit = cap_audit_limit
        self.records = []
        self.metadata = None
        self._audit_pairs = []

    def _initialize(self, env):
        if self.metadata is not None:
            if self.env is not env:
                raise ValueError(
                    "Use one MolecularDiagnostics instance per environment/checkpoint"
                )
            return
        self.env = env
        scorer = getattr(env, "scorer", None)
        objective = getattr(scorer, "objective", None)
        generator = getattr(env, "candidate_generator", None)
        rows = list(getattr(env, "building_blocks", ()))
        self.metadata = {
            "environment_class": type(env).__name__,
            "atom_limit": getattr(env, "length", None),
            "score_threshold": getattr(env, "threshold", None),
            "max_episode_steps": getattr(env, "max_episode_steps", None),
            "max_attach": getattr(env, "max_attach", None),
            "initial_smiles": getattr(env, "_base_smiles", None),
            "objective": getattr(
                objective, "name", getattr(objective, "__name__", "unknown")
            ),
            "objective_is_builtin": getattr(scorer, "_builtin", None),
            "reward_mode": getattr(env, "reward_mode", None),
            "catalog_source": "proposer"
            if getattr(env, "_proposer", None) is not None
            else "fixed",
            "fragment_rows": rows,
            "catalog_smiles": list(getattr(env, "catalog_smiles", rows)),
            "chemistry": dict(getattr(generator, "config", {})),
        }

    def capture(self, original_episode_method):
        """Return a wrapper accepting bound and unbound Sampler.episode calls."""
        bound_sampler = getattr(original_episode_method, "__self__", None)

        @wraps(original_episode_method)
        def wrapped(*args, **kwargs):
            sampler = bound_sampler or (args[0] if args else None)
            env = getattr(sampler, "env", None)
            if env is None:
                env = self.env
            if env is None:
                raise ValueError("capture requires a sampler.env or constructor env")
            self._initialize(env)
            generation_calls, entropy_records = [], []
            generator = getattr(env, "candidate_generator", None)
            algo = getattr(sampler, "algo", None)
            seed = kwargs.get("seed")
            random = bool(kwargs.get("random", False))
            original_generate = getattr(generator, "generate", None)
            original_compute = getattr(algo, "compute_single_action", None)

            def generate(*values, **options):
                result = original_generate(*values, **options)
                base = values[0] if values else options.get("base_mol")
                fragment = values[1] if len(values) > 1 else options.get("fragment_mol")
                generation_calls.append(
                    {
                        "base_atoms": base.GetNumAtoms() if base is not None else None,
                        "fragment_atoms": fragment.GetNumAtoms()
                        if fragment is not None
                        else None,
                        "candidates": len(result),
                        "stats": dict(getattr(generator, "stats", {})),
                    }
                )
                return result

            @wraps(original_compute)
            def compute(*values, **options):
                # value_lookahead asks for explicit full_fetch previews. They
                # are not actual policy decisions and must not inflate entropy.
                requested_full_fetch = bool(options.get("full_fetch", False))
                if requested_full_fetch:
                    return original_compute(*values, **options)
                observation = values[0] if values else options.get("observation")
                if observation is None and options.get("input_dict") is not None:
                    observation = options["input_dict"].get("obs")
                output = original_compute(*values, **dict(options, full_fetch=True))
                entropy = _policy_entropy(observation, output)
                if entropy is not None:
                    entropy_records.append(entropy)
                # RLlib Algorithm's full_fetch flag only changes its return
                # shape; Policy.compute_single_action performs the same call.
                if isinstance(output, tuple) and len(output) >= 3:
                    return output if output[1] else output[0]
                return output

            from contextlib import ExitStack

            with ExitStack() as stack:
                stats_available = callable(original_generate)
                if stats_available:
                    stack.enter_context(
                        _replace_method(generator, "generate", generate)
                    )
                entropy_available = (
                    not random
                    and callable(original_compute)
                    and _supports_keyword(original_compute, "full_fetch")
                )
                if entropy_available:
                    stack.enter_context(
                        _replace_method(algo, "compute_single_action", compute)
                    )
                result = original_episode_method(*args, **kwargs)
            self.records.append(
                self._record(
                    result,
                    seed,
                    random,
                    sampler,
                    generation_calls,
                    entropy_records,
                    stats_available,
                    entropy_available,
                )
            )
            return result

        return wrapped

    def _record(
        self,
        result,
        seed,
        random,
        sampler,
        generations,
        entropy,
        stats_available,
        entropy_available,
    ):
        events, generation_index, selected_fragment = [], 0, None
        infos = result.infos
        rows = self.metadata["fragment_rows"]
        for index, info in enumerate(infos):
            phase = info.get("phase", "legacy")
            generation = None
            if phase == "select_bb" or (phase == "legacy" and "prev_action" in info):
                if generation_index < len(generations):
                    generation = generations[generation_index]
                generation_index += 1
                identifier = info.get("chosen_bb", info.get("prev_action"))
                identifier = int(identifier) if identifier is not None else None
                fragment_smiles = info.get("chosen_bb_smiles")
                if (
                    fragment_smiles is None
                    and identifier is not None
                    and 0 <= identifier < len(rows)
                    and self.metadata["catalog_source"] == "fixed"
                ):
                    fragment_smiles = rows[identifier]
                generated = info.get(
                    "n_candidates_total",
                    generation["candidates"]
                    if generation
                    else len(info.get("gen_smiles", ())),
                )
                retained = info.get("n_candidates", len(info.get("gen_smiles", ())))
                selected_fragment = {
                    "step": index + 1,
                    "phase": phase,
                    "fragment_id": identifier,
                    "fragment_smiles": fragment_smiles,
                    "prev_smiles": info.get("prev_smiles"),
                    "generated": int(generated),
                    "retained": int(retained),
                    "capped": generated > retained,
                    "base_atoms": generation["base_atoms"] if generation else None,
                    "fragment_atoms": generation["fragment_atoms"]
                    if generation
                    else None,
                    "candidate_stats": generation["stats"] if generation else None,
                }
                events.append(selected_fragment)
                if (
                    generated > retained
                    and len(self._audit_pairs) < self.cap_audit_limit
                ):
                    pair = (info.get("prev_smiles"), fragment_smiles)
                    if all(pair) and pair not in self._audit_pairs:
                        self._audit_pairs.append(pair)
            if phase == "select_attach" or (phase == "legacy" and "score" in info):
                chosen = info.get("chosen_attach")
                selected = info.get("final_smiles")
                if selected is None and chosen is not None:
                    selected = info.get("gen_smiles", ())[int(chosen)]
                if selected is None:
                    selected = (
                        infos[index + 1].get("prev_smiles")
                        if index + 1 < len(infos)
                        else result.smiles
                    )
                events.append(
                    {
                        "step": index + 1,
                        "phase": "select_attach"
                        if phase != "legacy"
                        else "legacy_update",
                        "chosen_attach": int(chosen) if chosen is not None else None,
                        "prev_smiles": info.get("prev_smiles"),
                        "selected_candidate_smiles": selected,
                        "score": _number(info.get("score")),
                        "improvement": _number(info.get("improvement")),
                        "previous_score": _number(info.get("prev_score")),
                        "fragment_id": selected_fragment["fragment_id"]
                        if selected_fragment
                        else None,
                        "fragment_smiles": selected_fragment["fragment_smiles"]
                        if selected_fragment
                        else None,
                        "no_candidates": selected_fragment is not None
                        and selected_fragment["retained"] == 0,
                    }
                )
        updates = [
            event
            for event in events
            if event["phase"] in {"select_attach", "legacy_update"}
            and not event["no_candidates"]
        ]
        first = updates[0] if updates else None
        initial_score = first.get("previous_score") if first else float(result.score)
        if initial_score is None and first and first.get("improvement") is not None:
            initial_score = first["score"] - first["improvement"]
        initial_smiles = next(
            (event["prev_smiles"] for event in events if event.get("prev_smiles")),
            self.metadata.get("initial_smiles") or result.smiles,
        )
        visited = (
            [(initial_smiles, initial_score, 0)] if initial_score is not None else []
        )
        visited.extend(
            (event["selected_candidate_smiles"], event["score"], event["step"])
            for event in updates
            if event["score"] is not None
        )
        best = max(visited, key=lambda entry: entry[1]) if visited else None
        return {
            "seed": int(seed) if seed is not None else None,
            "smiles": result.smiles,
            "score": float(result.score),
            "total_reward": float(result.total_reward),
            "steps": int(result.steps),
            "terminated": bool(result.terminated),
            "truncated": bool(result.truncated),
            "end_phase": infos[-1].get("phase", "legacy") if infos else None,
            "scoring": {key: int(value) for key, value in result.scoring.items()},
            "random": random,
            "explore": bool(getattr(sampler, "explore", False)),
            "selector": getattr(sampler, "selector", "unknown"),
            "events": events,
            "candidate_stats_available": stats_available,
            "entropy_available": entropy_available,
            "policy_entropy": entropy,
            "initial_smiles": initial_smiles,
            "initial_score": initial_score,
            "best_visited_score": best[1] if best else None,
            "best_visited_smiles": best[0] if best else None,
            "best_visited_step": best[2] if best else None,
            "final_gap_from_best_visited": max(0.0, best[1] - float(result.score))
            if best
            else None,
        }

    def write(self, output_directory):
        """Run offline chemistry once per unique molecule and write a JSON report."""
        from rdkit import Chem
        from rdkit.Chem import QED, rdMolDescriptors
        from rdkit.Chem.Scaffolds import MurckoScaffold

        start = time.perf_counter()
        metadata = self.metadata or {}
        parsed = {}

        def molecule(smiles):
            if not smiles:
                return None
            if smiles not in parsed:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    for atom in mol.GetAtoms():
                        atom.SetAtomMapNum(0)
                parsed[smiles] = mol
            return parsed[smiles]

        def canonical(smiles):
            mol = molecule(smiles)
            return Chem.MolToSmiles(mol) if mol is not None else None

        catalog = [canonical(smiles) for smiles in metadata.get("catalog_smiles", ())]
        catalog_valid = {smiles for smiles in catalog if smiles is not None}
        row_molecules = [
            molecule(smiles) for smiles in metadata.get("fragment_rows", ())
        ]
        largest = max(
            (
                mol.GetNumAtoms()
                for mol in [
                    *row_molecules,
                    *(molecule(smiles) for smiles in catalog_valid),
                ]
                if mol is not None
            ),
            default=0,
        )
        counts = Counter(canonical(record["smiles"]) for record in self.records)
        valid_samples = sum(
            count for smiles, count in counts.items() if smiles is not None
        )
        molecule_records = []
        for smiles, count in counts.items():
            if smiles is None:
                continue
            mol = molecule(smiles)
            properties = QED.properties(mol)
            desirability = {
                name: float(QED.ads(value, QED.adsParameters[name]))
                for name, value in properties._asdict().items()
            }
            weights = QED.WEIGHT_MEAN._asdict()
            weight_total = sum(weights.values())
            penalties = {
                name: -weights[name] * math.log(value) / weight_total
                for name, value in desirability.items()
            }
            scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
            molecule_records.append(
                {
                    "canonical_smiles": smiles,
                    "sample_count": count,
                    "atom_count": mol.GetNumAtoms(),
                    "heavy_atom_count": mol.GetNumHeavyAtoms(),
                    "qed": float(QED.qed(mol, qedProperties=properties)),
                    "qed_properties": {
                        key: float(value) for key, value in properties._asdict().items()
                    },
                    "qed_desirability": desirability,
                    "qed_weighted_negative_log_contribution": penalties,
                    "formal_charge": sum(
                        atom.GetFormalCharge() for atom in mol.GetAtoms()
                    ),
                    "radical_electrons": sum(
                        atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()
                    ),
                    "connected_components": len(Chem.GetMolFrags(mol)),
                    "ring_count": rdMolDescriptors.CalcNumRings(mol),
                    "murcko_scaffold": scaffold or None,
                }
            )
        episodes, used_rows, used_fragments, kinds = [], Counter(), Counter(), Counter()
        reasons, caps, totals, retained_totals, raw_stats = (
            Counter(),
            0,
            0,
            0,
            Counter(),
        )
        entropy_by_phase = {}
        for record in self.records:
            episode = dict(record, events=[dict(event) for event in record["events"]])
            selected = [
                event
                for event in episode["events"]
                if event["phase"] in {"select_attach", "legacy_update"}
                and not event.get("no_candidates")
            ]
            for event in episode["events"]:
                if event["phase"] in {"select_bb", "legacy"} and "generated" in event:
                    if event["fragment_id"] is not None:
                        used_rows[event["fragment_id"]] += 1
                    chemical = canonical(event["fragment_smiles"])
                    if chemical is not None:
                        used_fragments[chemical] += 1
                    totals += event["generated"]
                    retained_totals += event["retained"]
                    caps += event["capped"]
                    raw_stats.update(event.get("candidate_stats") or {})
                if event["phase"] in {
                    "select_attach",
                    "legacy_update",
                } and not event.get("no_candidates"):
                    before, fragment, after = (
                        molecule(event[key])
                        for key in (
                            "prev_smiles",
                            "fragment_smiles",
                            "selected_candidate_smiles",
                        )
                    )
                    if all(mol is not None for mol in (before, fragment, after)):
                        delta = after.GetNumAtoms() - before.GetNumAtoms()
                        event["atom_delta"] = delta
                        merged = (
                            before.GetNumAtoms()
                            + fragment.GetNumAtoms()
                            - after.GetNumAtoms()
                        )
                        event["merge_kind"] = {1: "node", 2: "edge"}.get(
                            merged, "unknown"
                        )
                        kinds[event["merge_kind"]] += 1
            first = selected[0] if selected else None
            initial_score = record.get(
                "initial_score",
                first.get("previous_score") if first else record["score"],
            )
            if initial_score is None and first and first.get("improvement") is not None:
                initial_score = first["score"] - first["improvement"]
            scores = ([initial_score] if initial_score is not None else []) + [
                event["score"] for event in selected if event["score"] is not None
            ]
            negative = [
                i for i in range(1, len(scores)) if scores[i] < scores[i - 1] - 1e-12
            ]
            episode["trajectory"] = {
                "initial_score": initial_score,
                "molecular_updates": len(selected),
                "negative_transitions": len(negative),
                "recovered_negative_transitions": sum(
                    max(scores[i:]) >= scores[i - 1] - 1e-12 for i in negative
                ),
                "best_observed_score": max(scores) if scores else None,
                "final_score_drop_from_best": max(scores) - record["score"]
                if scores
                else None,
            }
            final = molecule(record["smiles"])
            atoms = final.GetNumAtoms() if final is not None else None
            limit = metadata.get("atom_limit")
            no_attachment = bool(
                episode["events"] and episode["events"][-1].get("retained") == 0
            )
            if record["truncated"]:
                horizon = metadata.get("max_episode_steps")
                reason = (
                    "time_limit"
                    if horizon is not None and record["steps"] >= horizon
                    else "sampler_limit"
                )
            elif record["end_phase"] == "no_candidates":
                reason = "empty_action_mask"
            elif no_attachment or (
                episode["events"] and episode["events"][-1].get("no_candidates")
            ):
                reason = "no_attachment_candidates"
            elif (
                record["terminated"]
                and atoms is not None
                and limit is not None
                and atoms > limit
            ):
                reason = "atom_limit"
            elif (
                record["terminated"]
                and metadata.get("score_threshold") is not None
                and record["score"] >= metadata["score_threshold"]
            ):
                reason = "score_threshold"
            else:
                reason = "other_terminal" if record["terminated"] else "unfinished"
            episode.update(
                end_reason=reason,
                canonical_smiles=canonical(record["smiles"]),
                final_atom_count=atoms,
                atom_limit_excess=max(0, atoms - limit)
                if atoms is not None and limit is not None
                else None,
                end_conditions={
                    "atom_limit_exceeded": atoms is not None
                    and limit is not None
                    and atoms > limit,
                    "score_threshold_met": metadata.get("score_threshold") is not None
                    and record["score"] >= metadata["score_threshold"],
                    "no_attachment_candidates": reason == "no_attachment_candidates",
                    "empty_action_mask": reason == "empty_action_mask",
                },
            )
            reasons[reason] += 1
            for entropy in record["policy_entropy"]:
                entropy_by_phase.setdefault(entropy["phase"], []).append(entropy)
            episodes.append(episode)
        bb_choices = sum(
            "generated" in event for episode in episodes for event in episode["events"]
        )
        fragment_choices = sum(used_fragments.values())
        fragment_entropy = (
            -sum(
                (count / fragment_choices) * math.log(count / fragment_choices)
                for count in used_fragments.values()
            )
            if fragment_choices
            else None
        )
        scaffold_counts = Counter()
        for record in molecule_records:
            scaffold_counts[record["murcko_scaffold"]] += record["sample_count"]
        cyclic = sum(
            count for scaffold, count in scaffold_counts.items() if scaffold is not None
        )
        generated_counts = [
            event["generated"]
            for episode in episodes
            for event in episode["events"]
            if "generated" in event
        ]
        initial = molecule(metadata.get("initial_smiles"))
        prior_limit = max(
            metadata.get("atom_limit") or 0,
            initial.GetNumAtoms() if initial is not None else 0,
        )
        summary = {
            "episode_count": len(episodes),
            "valid_episodes": valid_samples,
            "validity": valid_samples / len(episodes) if episodes else None,
            "unique_molecules": len(molecule_records),
            "unique_fraction_of_valid": len(molecule_records) / valid_samples
            if valid_samples
            else None,
            "top1_fraction_of_valid": max(
                (record["sample_count"] for record in molecule_records), default=0
            )
            / valid_samples
            if valid_samples
            else None,
            "top5_fraction_of_valid": sum(
                sorted(
                    (record["sample_count"] for record in molecule_records),
                    reverse=True,
                )[:5]
            )
            / valid_samples
            if valid_samples
            else None,
            "end_reasons": dict(reasons),
            "selected_merge_kinds": dict(kinds),
            "molecular_updates": sum(
                episode["trajectory"]["molecular_updates"] for episode in episodes
            ),
            "episodes_with_negative_transitions": sum(
                episode["trajectory"]["negative_transitions"] > 0
                for episode in episodes
            ),
            "negative_transitions": sum(
                episode["trajectory"]["negative_transitions"] for episode in episodes
            ),
            "recovered_negative_transitions": sum(
                episode["trajectory"]["recovered_negative_transitions"]
                for episode in episodes
            ),
            "score_threshold_attained_episodes": sum(
                episode["end_conditions"]["score_threshold_met"] for episode in episodes
            ),
            "fragment_choices": bb_choices,
            "capped_choices": caps,
            "capped_choice_fraction": caps / bb_choices if bb_choices else None,
            "generated_candidates": totals,
            "retained_candidates": retained_totals,
            "discarded_candidate_fraction": (totals - retained_totals) / totals
            if totals
            else None,
            "generated_candidates_per_choice": _distribution(generated_counts),
            "raw_candidate_stats": dict(raw_stats),
            "catalog_row_count": len(metadata.get("fragment_rows", ())),
            "catalog_unique_canonical_count": len(catalog_valid),
            "used_fragment_rows": len(used_rows),
            "fragment_row_coverage": len(used_rows)
            / len(metadata.get("fragment_rows", ()))
            if metadata.get("fragment_rows")
            and metadata.get("catalog_source") == "fixed"
            else None,
            "used_canonical_fragments": len(used_fragments),
            "canonical_fragment_coverage": len(set(used_fragments) & catalog_valid)
            / len(catalog_valid)
            if catalog_valid
            else None,
            "fragment_choice_entropy": fragment_entropy,
            "fragment_row_frequency": dict(used_rows),
            "canonical_fragment_frequency": dict(used_fragments),
            "nonempty_murcko_scaffolds": sum(
                scaffold is not None for scaffold in scaffold_counts
            ),
            "acyclic_fraction_of_valid": scaffold_counts[None] / valid_samples
            if valid_samples
            else None,
            "top1_scaffold_fraction_of_cyclic": max(
                (
                    count
                    for scaffold, count in scaffold_counts.items()
                    if scaffold is not None
                ),
                default=0,
            )
            / cyclic
            if cyclic
            else None,
            "final_atom_counts": _distribution(
                [
                    episode["final_atom_count"]
                    for episode in episodes
                    if episode["final_atom_count"] is not None
                ]
            ),
            "atom_limit_excess": _distribution(
                [
                    episode["atom_limit_excess"]
                    for episode in episodes
                    if episode["atom_limit_excess"] is not None
                ]
            ),
            "catalog_max_fragment_atoms": largest,
            "node_result_atom_upper_bound": prior_limit + largest - 1
            if largest
            else None,
            "edge_result_atom_upper_bound": prior_limit + largest - 2
            if largest
            else None,
            "policy_entropy_by_phase": {
                phase: _entropy_summary(values)
                for phase, values in entropy_by_phase.items()
            },
            "episodes_with_entropy": sum(
                bool(episode["policy_entropy"]) for episode in episodes
            ),
            "best_visited_scores": _distribution(
                [
                    episode["best_visited_score"]
                    for episode in episodes
                    if episode.get("best_visited_score") is not None
                ]
            ),
            "final_gap_from_best_visited": _distribution(
                [
                    episode["final_gap_from_best_visited"]
                    for episode in episodes
                    if episode.get("final_gap_from_best_visited") is not None
                ]
            ),
            "episodes_final_below_best_visited": sum(
                episode.get("final_gap_from_best_visited", 0) > 1e-12
                for episode in episodes
            ),
            "final_qed": _distribution(
                [
                    record["qed"]
                    for record in molecule_records
                    for _ in range(record["sample_count"])
                ]
            ),
            "final_qed_properties": {
                name: _distribution(
                    [
                        record["qed_properties"][name]
                        for record in molecule_records
                        for _ in range(record["sample_count"])
                    ]
                )
                for name in QED.QEDproperties._fields
            },
            "final_molecules_present_in_fragment_catalog": sum(
                record["sample_count"]
                for record in molecule_records
                if record["canonical_smiles"] in catalog_valid
            ),
        }
        audits = self._cap_audits()
        output = {
            "schema_version": 1,
            "metadata": metadata,
            "summary": summary,
            "episodes": episodes,
            "molecules": molecule_records,
            "cap_audits": audits,
            "offline_analysis_seconds": time.perf_counter() - start,
            "notes": [
                "Capture does not generate or score extra candidates; RDKit analysis and optional audits run only in write().",
                "Policy entropy is entropy of the masked softmax logits, not of greedy argmax outputs; K=1 has normalized_entropy=null.",
                "Explicit full_fetch lookahead value previews are excluded from policy-decision entropy.",
                "QED is bounded by 0 and 1; a threshold above 1 disables score-threshold termination for the standard QED objective.",
                "Atom limits terminate after a selected merge exceeds the limit; they do not prefilter oversize candidates.",
                "Capped-choice denominator is all selected building blocks, including choices with zero candidates.",
                "Raw candidate invalid/duplicate counts use CandidateGenerator's processing-stage counts and residual duplicate definition.",
                "Fragment catalog size is not the number of reachable assembled molecules; catalog-relative novelty is not training-set novelty.",
                "Fragment-row coverage is reported only for a fixed table; proposer row IDs can denote different fragments after refresh.",
                "Deterministic full-catalog greedy evaluation often repeats trajectories across environment seeds; repetition alone does not establish policy mode collapse.",
                "Empty Murcko scaffolds (acyclic molecules) are counted separately from cyclic scaffold concentration.",
                "QED contributions are weighted negative log desirabilities; their sum equals -log(QED).",
                "Best-visited scores include the initial molecule and actually selected intermediates only; final scores remain the primary evaluation outcome.",
                "Offline cap audits compare immediate QED only and do not establish lost future return or a global optimum.",
            ],
        }
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "molecular_diagnostics.json"
        path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        return path

    def _cap_audits(self):
        if not self._audit_pairs:
            return {
                "enabled": self.cap_audit_limit > 0,
                "score_computations": 0,
                "pairs": [],
            }
        from rdkit.Chem import QED

        from moldr.chemistry import CandidateGenerator
        from moldr.chemutils import get_mol

        if self.metadata.get("chemistry", {}).get("mol_transfer") == "direct":
            return {
                "enabled": True,
                "score_computations": 0,
                "pairs": [],
                "skip_reason": "SMILES-only capture does not preserve direct-Mol atom ordering for exact replay",
            }
        records, computations = [], 0
        generator = CandidateGenerator(self.metadata.get("chemistry"))
        for base, fragment in self._audit_pairs:
            start = time.perf_counter()
            candidates = generator.generate(get_mol(base), get_mol(fragment))
            scores = [float(QED.qed(candidate.mol)) for candidate in candidates]
            retained = self.metadata.get("max_attach") or len(scores)
            before, after = scores[:retained], scores[retained:]
            computations += len(scores)
            records.append(
                {
                    "base_smiles": base,
                    "fragment_smiles": fragment,
                    "total": len(scores),
                    "retained": len(before),
                    "discarded": len(after),
                    "retained_best_immediate_qed": max(before) if before else None,
                    "discarded_best_immediate_qed": max(after) if after else None,
                    "discarded_best_immediate_qed_advantage": max(
                        0.0, max(after) - max(before)
                    )
                    if before and after
                    else None,
                    "offline_score_computations": len(scores),
                    "seconds": time.perf_counter() - start,
                }
            )
        return {
            "enabled": True,
            "objective": "RDKit QED",
            "score_computations": computations,
            "normal_environment_score_counters_affected": False,
            "pairs": records,
        }


def _distribution(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _entropy_summary(values):
    normalized = [
        value["normalized_entropy"]
        for value in values
        if value["normalized_entropy"] is not None
    ]
    return {
        "decisions": len(values),
        "single_action_fraction": sum(value["single_action"] for value in values)
        / len(values),
        "entropy": _distribution([value["entropy"] for value in values]),
        "normalized_entropy": _distribution(normalized),
        "valid_actions": _distribution([value["valid_actions"] for value in values]),
        "maximum_probability": _distribution(
            [value["maximum_probability"] for value in values]
        ),
    }
