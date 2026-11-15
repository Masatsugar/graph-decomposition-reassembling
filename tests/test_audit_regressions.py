"""Regression tests for codebase audit fixes and speedups."""

import json
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from rdkit import Chem

import moldr.sascore as sascore
from moldr.chemutils import canonical_smiles, get_mol, get_smiles
from moldr.cli import explicit_values, parser, run_evaluation
from moldr.core.moltree import MolTree
from moldr.env import MolEnvValueMax
from moldr.experiment import ExperimentConfig
from moldr.factory import (
    _load_fixed_file,
    load_fragments,
    make_environment,
    resolve_config,
)
from moldr.gspan.utils import (
    create_gspan_dataset,
    create_junction_tree_dataset,
    save_graph,
)
from moldr.metrics import MoleculeMetrics
from moldr.mol2vec.features import MolSentence, mol2alt_sentence, sentences2vec
from moldr.mol2vec.model import Mol2Vec
from moldr.provenance import (
    load_saved_config,
    prepare_restored_config,
    save_run,
    sha256_file,
)
from moldr.scoring import ScoringService


def test_mol2vec_default_radius_1_and_sentences2vec_bit_exact():
    model = Mol2Vec(model_path="models/model_300dim.pkl")
    assert model.radius == 1

    mols = [
        Chem.MolFromSmiles(s) for s in ("C", "CCO", "c1ccccc1", "CC1CCC2=CC=CC=C2O1")
    ]
    embeddings = np.asarray(model.fit_transform(mols), dtype=np.float32)
    unk = np.asarray(model.model.wv["UNK"], dtype=np.float32)
    wv = model.model.wv
    for smiles in ("CCO", "c1ccccc1", "CC1CCC2=CC=CC=C2O1"):
        assert [
            t
            for t in mol2alt_sentence(Chem.MolFromSmiles(smiles), 1)
            if t not in wv.key_to_index
        ] == []
    assert (
        len(
            [
                t
                for t in mol2alt_sentence(Chem.MolFromSmiles("c1ccccc1"), 2)
                if t not in wv.key_to_index
            ]
        )
        == 6
    )
    assert not np.allclose(embeddings[1], unk)
    assert not np.allclose(embeddings[2], unk)
    assert not np.allclose(embeddings[1], embeddings[2])

    # Verify key_to_index fast lookup is bit-for-bit identical to set(index_to_key).
    sentences = [MolSentence(mol2alt_sentence(m, 1)) for m in mols]
    vocab_set = set(wv.index_to_key)
    ref_vecs = np.array(
        [
            sum(wv[word] if word in vocab_set else wv["UNK"] for word in sentence)
            for sentence in sentences
        ]
    )
    fast_vecs = sentences2vec(sentences, model.model, unseen="UNK")
    assert np.array_equal(fast_vecs, ref_vecs)

    bad_cfg = ExperimentConfig()
    bad_cfg.representation.radius = 2
    bad_cfg.fragments.smiles = ["C", "CC"]
    with pytest.raises(ValueError, match="radius"):
        resolve_config(bad_cfg)


def test_scoring_qed_sa_mol_and_smiles_parity_and_cache_key():
    assert ExperimentConfig().scoring.cache_size == 4096

    smiles_svc = ScoringService("qed-sa", input_mode="smiles", cache_size=16)
    mol_svc = ScoringService("qed-sa", input_mode="mol", cache_size=16)
    samples = [
        "C",
        "c1ccccc1",
        "c1ccc2ccccc2c1",
        "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1",
    ]
    for smi in samples:
        expected = smiles_svc.score(smi)
        assert mol_svc.score(get_mol(smi)) == pytest.approx(expected)
        assert mol_svc.score(Chem.MolFromSmiles(smi)) == pytest.approx(expected)

    # Built-in Mol/SMILES/kekule representations for the same molecule share one cache entry.
    shared = ScoringService("qed-sa", input_mode="mol", cache_size=16)
    s1 = shared.score(get_mol("c1ccccc1"))
    s2 = shared.score("c1ccccc1")
    s3 = shared.score(Chem.MolFromSmiles("c1ccccc1"))
    assert s1 == pytest.approx(s2)
    assert s2 == pytest.approx(s3)
    assert shared.stats == {"requests": 3, "computations": 1, "hits": 2}


def test_legacy_env_mol_mode_preserves_full_greedy_candidates():
    cfg = ExperimentConfig()
    cfg.environment.mode = "legacy"
    cfg.environment.base_smiles = "c1ccccc1"
    cfg.environment.max_attach = 1
    cfg.environment.atom_limit = 30
    cfg.scoring.objective = "qed-sa"
    cfg.scoring.input_mode = "mol"
    cfg.fragments.smiles = ["CCO", "c1ccccc1"]

    env = make_environment(cfg)
    assert isinstance(env, MolEnvValueMax)
    obs, _ = env.reset(seed=0)
    unk = np.asarray(env.mol2vec.model.wv["UNK"], dtype=np.float32)
    assert not np.allclose(obs, unk)
    expected_reset = ScoringService("qed-sa").score("c1ccccc1")
    assert env.prev_reward == pytest.approx(expected_reset)

    mols, smiles = env._reassemble(0)
    assert len(mols) == len(smiles) == 2 > cfg.environment.max_attach

    _, _, _, _, info = env.step(0)
    assert info["score"] == pytest.approx(
        ScoringService("qed-sa").score(env.base_smiles)
    )
    env.close()

    bad_cfg = ExperimentConfig()
    bad_cfg.environment.max_attach = 0
    with pytest.raises(ValueError, match="environment.max_attach"):
        bad_cfg.validate()


def test_named_run_checkpoint_restore_hardlinks_and_out_of_repo_cwd(
    tmp_path, monkeypatch
):
    cfg = ExperimentConfig()
    cfg.output.directory = str(tmp_path / "runs")
    cfg.output.run_name = "myrun"
    cfg.fragments.smiles = ["C", "CC", "CCO"]
    resolved, blocks, catalog = resolve_config(cfg)
    run_dir = tmp_path / "runs" / "myrun"
    save_run(run_dir, resolved, blocks=blocks, catalog=catalog)
    ckpt = run_dir / "checkpoint_000001"
    ckpt.mkdir()
    (ckpt / "rllib_checkpoint.json").write_text("{}")

    loaded, _ = load_saved_config(run_dir)
    assert loaded.output.run_name is None

    reloaded_same = load_saved_config(
        run_dir, cli_values={"output.checkpoint": str(run_dir)}
    )[0]
    reloaded_same.output.run_name = "myrun"
    prepared = prepare_restored_config(reloaded_same)
    assert prepared.output.run_name is None

    eval_cfg = prepare_restored_config(
        load_saved_config(run_dir, cli_values={"output.checkpoint": str(run_dir)})[0]
    )
    eval_cfg.output.run_name = "eval1"
    eval_dir = save_run(
        tmp_path / "runs" / "eval1", eval_cfg, blocks=blocks, catalog=catalog
    )
    assert (
        os.stat(eval_dir / "assets/mol2vec/model_300dim.pkl").st_ino
        == os.stat(run_dir / "assets/mol2vec/model_300dim.pkl").st_ino
    )

    outside_cfg = ExperimentConfig()
    outside_cfg.fragments.smiles = ["C", "CC", "CCO"]
    outside_cfg.representation.model_path = str(
        Path("models/model_300dim.pkl").resolve()
    )
    outside_resolved, _, _ = resolve_config(outside_cfg)

    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    monkeypatch.setattr(sascore, "_fscores", None)
    assert sascore.calculateScore(Chem.MolFromSmiles("CCO")) > 0

    outside_run = tmp_path / "runs" / "outside_run"
    save_run(outside_run, outside_resolved, blocks=blocks, catalog=catalog)
    manifest = pd.read_json(outside_run / "manifest.json", typ="series")
    assert len(manifest["source_sha256"]) > 0
    assert manifest["revision"] is None or len(manifest["revision"]) > 0


def test_molecule_metrics_rejects_invalid_smiles():
    callbacks = MoleculeMetrics()
    env = SimpleNamespace(base_smiles="CCO")
    base_env = SimpleNamespace(get_sub_environments=lambda: [env])

    bad_episode = SimpleNamespace(
        last_info_for=lambda: {
            "phase": "select_attach",
            "final_smiles": "not_a_valid_smiles((",
        }
    )
    callbacks.on_episode_step(base_env=base_env, env_index=0, episode=bad_episode)
    assert env._experiment_counts["updates"] == 1
    assert env._experiment_counts["valid_molecules"] == 0

    good_episode = SimpleNamespace(
        last_info_for=lambda: {"phase": "select_attach", "final_smiles": "CCO"}
    )
    callbacks.on_episode_step(base_env=base_env, env_index=0, episode=good_episode)
    assert env._experiment_counts["updates"] == 2
    assert env._experiment_counts["valid_molecules"] == 1


def test_core_env_imports_cleanly():
    import moldr.core.env as core_env

    assert hasattr(core_env, "MolEnv")


def test_cli_zinc_jt_fallback_fixed_action_indices_and_vae_conditioning(
    monkeypatch,
):
    parsed = explicit_values(
        parser().parse_args(["train", "--dataset", "ZINC", "--jt"])
    )
    expected_candidate = "data/building_blocks/zinc/gspan_jt.datavocab.csv"
    assert parsed["fragments.path"] == (
        expected_candidate
        if Path(expected_candidate).is_file()
        else "data/zinc/vocab.csv"
    )
    assert Path(parsed["fragments.path"]).is_file()

    cfg = ExperimentConfig()
    cfg.fragments.source = "fixed"
    cfg.fragments.path = "data/zinc/vocab.csv"
    raw_df = pd.read_csv(cfg.fragments.path, index_col=0)
    blocks, _ = load_fragments(cfg)
    assert blocks == raw_df.iloc[:, 0].tolist()

    import moldr.generative.proposer as proposer_mod

    seen_calls = []

    class FakeProposer:
        context_trained = True
        n_blocks = 2
        vocab = SimpleNamespace(smiles=["C", "CC"])
        vae = SimpleNamespace(context_dim=300)

        def propose(self, ctx):
            seen_calls.append(("propose", ctx))
            return ["C", "CC"]

        def propose_library(self, ctx):
            seen_calls.append(("propose_library", ctx))
            return ["C", "CC"]

    monkeypatch.setattr(
        proposer_mod.VAEProposer, "from_spec", lambda spec: FakeProposer()
    )

    vae_cfg = ExperimentConfig()
    vae_cfg.fragments.source = "vae"
    vae_cfg.fragments.proposer_dir = "unused"
    vae_cfg.environment.base_smiles = "c1ccccc1"
    vae_cfg.fragments.recond_every = "episode"
    load_fragments(vae_cfg)
    assert seen_calls[-1][0] == "propose"
    expected_vec = np.asarray(
        Mol2Vec(model_path=vae_cfg.representation.model_path, radius=1).fit_transform(
            [get_mol("c1ccccc1")]
        ),
        dtype=np.float32,
    ).reshape(-1)
    assert np.allclose(seen_calls[-1][1], expected_vec)

    vae_cfg.fragments.recond_every = "never"
    load_fragments(vae_cfg)
    assert seen_calls[-1] == ("propose_library", None)


def test_run_evaluation_forces_num_workers_zero(tmp_path, monkeypatch):
    import moldr.cli as cli_mod
    import moldr.factory as factory_mod
    import moldr.sampling as sampling_mod

    cfg = ExperimentConfig()
    cfg.output.directory = str(tmp_path / "runs")
    cfg.output.run_name = "trained"
    cfg.training.num_workers = 4
    cfg.fragments.smiles = ["C", "CC", "CCO"]
    resolved, blocks, catalog = resolve_config(cfg)
    run_dir = tmp_path / "runs" / "trained"
    save_run(run_dir, resolved, blocks=blocks, catalog=catalog)
    ckpt = run_dir / "checkpoint_000001"
    ckpt.mkdir()
    (ckpt / "rllib_checkpoint.json").write_text("{}")

    captured_workers = []

    class FakeAlgo:
        def restore(self, path):
            self.restored = path

        def stop(self):
            pass

    monkeypatch.setattr(cli_mod, "_init_ray", lambda c: None)
    monkeypatch.setattr(
        cli_mod,
        "make_ppo_config",
        lambda c: (
            captured_workers.append(c.training.num_workers),
            SimpleNamespace(build=lambda: FakeAlgo()),
        )[1],
    )

    class FakeSampler:
        def __init__(self, env, algo, **kwargs):
            pass

        def episode(self, seed=0, random=False):
            return SimpleNamespace(
                smiles="CCO",
                score=0.5,
                total_reward=0.5,
                steps=1,
                terminated=True,
                truncated=False,
                scoring={"requests": 1, "computations": 1, "hits": 0},
                infos=[],
            )

    monkeypatch.setattr(sampling_mod, "Sampler", FakeSampler)

    eval_in = load_saved_config(run_dir)[0]
    eval_in.evaluation.random = False
    eval_in.evaluation.episodes = 1
    eval_in.output.run_name = "eval_out"
    out_dir = run_evaluation(eval_in, checkpoint=run_dir)
    assert captured_workers == [0]
    saved_resolved = (out_dir / "resolved_config.yaml").read_text()
    assert "num_workers: 4" in saved_resolved


def test_create_gspan_and_jt_dataset_matches_moltree_reference(tmp_path):
    smiles = [
        "c1ccccc1",
        "CC(=O)Nc1ccc(O)cc1",
        "c1ccc2ccccc2c1",
        "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1",
    ]
    mols = [get_mol(s) for s in smiles]

    jt_tuples, jt_vocab = create_junction_tree_dataset(mols)

    ref_tuples = []
    ref_vocab = defaultdict()
    cnt = 0
    for idx, mol in enumerate(mols):
        mt = MolTree(get_smiles(mol))
        for node in mt.nodes:
            if node.smiles not in ref_vocab:
                ref_vocab[node.smiles] = cnt
                cnt += 1
        ref_tuples.append(("t", "#", str(idx)))
        for i, node in enumerate(mt.nodes):
            ref_tuples.append(("v", i, ref_vocab[node.smiles]))
        for left, right in mt.edges:
            ref_tuples.append(("e", left, right, 1))
    ref_tuples.append(("t", "#", "-1"))

    assert jt_tuples == ref_tuples
    assert dict(jt_vocab) == dict(ref_vocab)

    raw_tuples = create_gspan_dataset(mols[:2])
    out = tmp_path / "raw.data"
    save_graph(raw_tuples, out)
    lines = out.read_text().splitlines()
    assert lines[0] == "t # 0 "
    assert lines[-1] == "t # -1 "


def test_fixed_file_stat_cache_and_pre_radius_checkpoint_guard(tmp_path):
    frag_file = tmp_path / "frags.smi"
    frag_file.write_text("CC\nCO\n")

    cfg1 = ExperimentConfig()
    cfg1.fragments.path = str(frag_file)
    _load_fixed_file.cache_clear()
    before = _load_fixed_file.cache_info()
    _, blocks1, _ = resolve_config(cfg1)
    after_first = _load_fixed_file.cache_info()
    assert blocks1 == ["CC", "CO"]
    assert after_first.misses == before.misses + 1

    cfg2 = ExperimentConfig()
    cfg2.fragments.path = str(frag_file)
    _, blocks2, _ = resolve_config(cfg2)
    after_second = _load_fixed_file.cache_info()
    assert blocks2 == ["CC", "CO"]
    assert after_second.hits > after_first.hits

    frag_file.write_text("CC\nCO\nCCN\n")
    cfg3 = ExperimentConfig()
    cfg3.fragments.path = str(frag_file)
    resolved3, blocks3, catalog3 = resolve_config(cfg3)
    after_third = _load_fixed_file.cache_info()
    assert blocks3 == ["CC", "CO", "CCN"]
    assert after_third.misses > after_second.misses

    run_dir = tmp_path / "runs" / "pre_radius"
    save_run(run_dir, resolved3, blocks=blocks3, catalog=catalog3)
    resolved_path = run_dir / "resolved_config.yaml"
    raw = yaml.safe_load(resolved_path.read_text())
    raw["representation"].pop("radius")
    resolved_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["resolved_config_sha256"] = sha256_file(resolved_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="predates explicit Mol2Vec radius tracking"):
        load_saved_config(run_dir)
