# Python だけで試す MOLDR チュートリアル

小さなサンプルで動作を確認し、目的に合わせたスコアを設定して、学習済みモデルから分子を生成するまでの手順です。
Jupyter は不要です。入口は [examples/tutorial.py](../examples/tutorial.py)、
独自スコアの例は [examples/tutorial_objective.py](../examples/tutorial_objective.py) にあります。

このチュートリアルは共通 factory / CLI と E2E v2 を含む開発版に対応しています。
2026-11-07 時点の `main` にはまだ含まれていないため、スクリプトと同じ版のソース一式を使ってください。
README の Installation を済ませ、以下のコマンドはリポジトリ直下で実行します。

## 1. 学習せずに 1 エピソード動かす

```bash
uv run python examples/tutorial.py demo
```

付属の小さな building block 表と固定 Mol2Vec モデルを使い、CPU で分子を生成します。
Ray の起動や PPO 学習は行いません。各段階の有効な選択肢から一様ランダムに選ぶため、
この結果は学習済みモデルの性能を表しません。

E2E は「A: building block を選ぶ → B: 再結合候補を選ぶ」を繰り返します。
最大 8 decisions は通常 4 回の再結合に相当します。
デモは段階、行動後の SMILES、分子スコア、報酬を表示し、出力先を最後に示します。
終了理由は保存した `trajectory.csv` と `molecules.json` で確認できます。
既定の出力先は `outputs/tutorial/` の新しい run ディレクトリです。

| ファイル | 内容 |
| --- | --- |
| `objective_preview.csv` | 学習前に確かめるサンプル分子とスコア |
| `trajectory.csv` | 各 decision の分子・報酬・終了フラグ |
| `molecules.json` | 生成結果 |
| `tutorial_settings.json` | チュートリアルで指定した条件 |
| `resolved_config.yaml` | 実際に使った設定 |
| `manifest.json`, `fragment_catalog.json`, `assets/` | 復元に必要な構成とアセット |

各コマンドの引数は `--help` で確認できます。

```bash
uv run python examples/tutorial.py --help
uv run python examples/tutorial.py train --help
uv run python examples/tutorial.py sample --help
```

別のディレクトリから実行する場合は、このプロジェクトの `.venv/bin/python` と
`examples/tutorial.py` を絶対パスで指定できます。`--fragments`、`--checkpoint`、`--output` の
相対パスは、コマンドを起動したディレクトリから解決します。

## 2. 分子スコアを目的に合わせる

最初に決めるのは、分子 (m) を評価する **スコア (S(m))** です。
この例では値が大きいほど良い分子として扱います。
既定の独自スコアは QED と「目標の重原子数への近さ」の加重平均です。

```text
size_score(m) = exp(-0.5 * ((heavy_atoms(m) - target_atoms) / size_width)^2)
S(m) = qed_weight * QED(m) + (1 - qed_weight) * size_score(m)
```

| 引数 | 既定値 | 意味 |
| --- | --- | --- |
| `--target-atoms` | `8` | 目標とする重原子数（水素以外の原子数） |
| `--size-width` | `4` | 目標からのずれをどの程度許容するか。小さいほど厳しい |
| `--qed-weight` | `0.5` | QED の重み。0〜1 で指定し、残りをサイズに配分する |

例えば、重原子数 10 付近を重視する場合は次のようにします。

```bash
uv run python examples/tutorial.py demo \
  --objective custom --target-atoms 10 --size-width 2 --qed-weight 0.3
```

サンプル分子の採点結果と `objective_preview.csv` を見て、意図した分子が高得点になるか確認してください。
このスコアは決定的で、常に有限の 0〜1 の値を返します。サイズと QED の組み合わせは説明用の例です。
実際の目的に適した指標と重みは、研究側で検討する必要があります。

QED だけを使う場合は、組み込み objective を選べます。

```bash
uv run python examples/tutorial.py demo --objective QED
```

独自の指標へ変えるときは `tutorial_objective.py` 内の採点式を編集します。
SMILES を受け取って値を返す関数を `moldr.scoring.register_objective` で登録し、
登録名を `scoring.objective` に設定するのが基本の接続方法です。
この例は RDKit Mol を直接採点する関数も登録しています。
`NaN` や無限大を返さず、「大きいほど良い」という向きをそろえてください。
同じ分子に同じ値を返す関数として `deterministic=True` を指定しています。
独自の式で値域を変更する場合は、スクリプト内の `environment.score_threshold=1.1` も
目的に合う終了条件へ見直してください。大きな値を返す指標は、報酬のスケールにも影響します。

## 3. スコアと報酬を区別する

PPO が受け取る **報酬 (r)** は、分子スコアから環境が計算します。
スコア関数を登録することと、各遷移の報酬の与え方を選ぶことは別の設定です。

| 場面 | `--reward improvement` | `--reward sparse` |
| --- | --- | --- |
| A: building block の選択（候補あり） | `0` | `0` |
| B: 再結合後、通常の遷移 | `S(new) - S(previous)` | `0` |
| B: 環境の終了条件を満たす | 上記に `final_weight * S(new)` を加える | `final_weight * S(new)` |
| A: 再結合候補がなく終了 | `final_weight * S(previous)` | `final_weight * S(previous)` |
| decisions の上限だけに到達 | 完了報酬は加えない | 完了報酬は加えない |

例えば `S(previous)=0.4`, `S(new)=0.6`, `final_weight=1` のとき、通常の improvement 報酬は `0.2` です。
その遷移で真に終了すれば `0.2 + 0.6 = 0.8`、時間制限による打ち切りだけなら `0.2` のままです。
生成結果の `score` と `reward` が違っていても、この定義に沿った挙動です。

```bash
uv run python examples/tutorial.py demo --reward improvement --final-weight 1
uv run python examples/tutorial.py demo --reward sparse --final-weight 0.5
```

`terminated` はスコア閾値・原子数超過・候補なしによる終了、`truncated` は decisions の上限による打ち切りです。
チュートリアルでは `score_threshold=1.1` として、0〜1 のスコアで閾値終了させない設定にしています。
原子数は `atom_count > atom_limit` で終了を判定するため、最終分子が上限を超える場合があります。
`sparse` で打ち切りだけのエピソードになると報酬はゼロです。まずは既定の `improvement` で動作を確認してください。
`sparse` と `final_weight=0` を組み合わせると、終了時も含めて報酬が常にゼロになります。
スコアを学習に反映するため、`sparse` の学習では完了報酬の重みを正にしてください。

## 4. 短い CPU 学習を実行する

デモで確認した条件を `train` に渡します。

```bash
uv run python examples/tutorial.py train --iterations 1 \
  --objective custom --target-atoms 10 --size-width 2 --qed-weight 0.3
```

CPU、GPU 数 0、rollout worker 数 0、batch 64、minibatch 32、更新 1 epoch、policy hidden 幅 32 の設定です。
worker 数が 0 でも、学習にはローカル Ray の起動が必要です。
1 iteration は学習・保存・復元を確かめるための短い実行で、分子最適化の性能を評価するには不十分です。

終了時に表示された run ディレクトリを控えてください。例えば `outputs/tutorial/train_..._seed7` です。
設定とアセットに加え、次のファイルが保存されます。

| ファイル | 内容 |
| --- | --- |
| `measurements.json` | 各 iteration の損失、時間など |
| `summary.json` | 学習回数、重みの変化など |
| `checkpoints/iteration_000001/` | 1 iteration 終了時の学習済みモデル |
| `last_checkpoint.json` | 最後に保存した checkpoint の場所 |
| `logs/` | 学習ログ |

独自 objective の関数とパラメータは `assets/objective.pkl` に保存されます。
サイズの目標や重みを関数に閉じ込めているため、後でソースを編集しても、保存済み run は学習時の採点定義を復元します。
RDKit などのライブラリは同じ実行環境で利用する必要があります。

## 5. 別プロセスでモデルを復元して使う

新しいターミナルから、学習時に表示された run のパスを指定します。

```bash
uv run python examples/tutorial.py sample \
  --checkpoint outputs/tutorial/train_実際の日時_seed7 --episodes 3
```

`train_実際の日時_seed7` は実際のディレクトリ名へ置き換えてください。
`sample` は保存された設定、fragment 表、Mol2Vec、独自 objective、policy の重みを読み込みます。
独自 objective の再登録や、ネットワーク構造の手動再作成は不要です。
同じ評価条件で、学習済み policy による生成を実行できます。

出力先に `molecules.json` と `molecules.csv`（SMILES・スコア・報酬・終了理由など）、
`evaluation.json`（平均スコア、有効分子率、多様性など）が保存されます。
3 エピソードは接続確認用です。研究で比較する場合は、評価回数・seed・探索条件をそろえてランダム選択とも比較してください。

checkpoint のサブディレクトリだけでなく、**run ディレクトリ全体**を保持・移動してください。
復元時は保存済みアセットを使うため、元の入力ファイルの場所が変わっても利用できます。
学習後に fragment 表や観測次元を差し替えることはできません。
旧 legacy / E2E v1 の checkpoint は、この E2E v2 の復元経路と互換ではありません。

## 6. 自分の分子・building block に置き換える

既定の 13 building blocks は、[quickstart.ipynb](../examples/quickstart.ipynb) の小さなサンプルから採掘したものです。
`--fragments` は**再結合に使う building block の表**を受け取ります。
元の分子データを自動で分解する引数ではありません。分解から試す場合は
[README の Decomposition](../README.md#decomposition) の Python コードを使えます。
構造を図で確認したい場合は Notebook の gSpan の例も利用できます。

例えば `my_fragments.smi` を次の内容で用意します。

```text
CC
CO
CN
C1CCCCC1
```

```bash
uv run python examples/tutorial.py demo \
  --fragments my_fragments.smi --base-smiles CCO --atom-limit 16 --max-steps 12
```

CSV を使う場合は、先頭列がインデックス、その次の列が SMILES の形式にします。

```csv
,smiles
0,CC
1,CO
2,CN
```

少数の分子で読み込みと採点を確かめてから、データを増やしてください。
明示的な原子・結合の立体化学は現在の再結合実装では扱えず、環境が入力を拒否します。
同じ条件で `train` を実行し、その新しい run を `sample --checkpoint` に渡すと、自分の building block を使った生成につながります。

より詳しい設定は [experiment_configuration.md](experiment_configuration.md)、
既存の長い学習と評価の結果は [experiment_learning500.md](experiment_learning500.md) を参照してください。
