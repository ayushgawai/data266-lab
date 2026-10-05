# DATA266 Lab 1 — Sneha

**In the team repo, this `sneha/` folder is the project root for Sneha's runs:** `cd sneha`
first, then every command below works as written. Ayush's work is in `../ayush/`, and the
shared brief and evaluation script are in `../team/`. Plans are in `plans/`; the current one is
`plans/DATA266_Lab1_Plan_Sneha_v3.html`. On the lab machine the same folder was called
`data266-lab1`.

LLM pretraining (Task 1), sentiment classification (Task 2), and CycleGAN style
transfer (Task 3). Inside this folder, Sneha's work is in the `sneha/` folder of each task,
next to the shared data, the evaluation code in `common/`, and `reproducibility/`.

| Task | Folder | Status |
|---|---|---|
| 1. Character-level GPT on TinyStories | `task1_llm/` | Sneha: one notebook, tested; awaiting the real training run |
| 2. Yelp Polarity sentiment | `task2_sentiment/` | Sneha: one notebook, tested; awaiting the real training run |
| 3. CycleGAN Monet ↔ Photo | `task3_gan/` | Sneha: one notebook, tested; awaiting the real training run |

## Smoke test (one command)

From the repo root, after the setup below:

```bash
python smoke_test.py
```

This runs Sneha's whole Task 1 notebook end to end in smoke mode: a tiny GPT on 160
sequences from the 22 MB TinyStories validation file, downloaded on first use. It takes
about a minute on a laptop CPU, prints `SMOKE TEST PASSED`, and exits non-zero if any cell
fails. `python smoke_test.py --task 2`, `--task 3` or `--task all` smoke-test the other
notebooks. Smoke outputs go to `task*/sneha/_smoke/` (git-ignored) and never touch a real run.

## Datasets (Google Drive)

The datasets are too large for GitHub. Zipped copies, readable by anyone with the link:

| Dataset | Used by | Drive link | Where to unzip |
|---|---|---|---|
| TinyStories V2 (`TinyStoriesV2-GPT4-train.txt`, `-valid.txt`) | Task 1 | _add link_ | `task1_llm/data/` |
| Yelp Polarity (`yelp_review_polarity_csv`: `train.csv`, `test.csv`, `readme.txt`) | Task 2 | _add link_ | `task2_sentiment/data/raw/` |
| Monet / photo (`dataset.zip` from the course, `monet_jpg/` + `photo_jpg/`) | Task 3 | _add link_ | `task3_gan/data/dataset.zip` (the notebook extracts it) |

The notebooks also download the TinyStories and Yelp files from their original sources if
they are missing, and record each file's SHA-256 in the logs and manifests.

## Setup

### On the SJSU GPU lab machine (Docker)

The lab rules forbid installing software on the machine itself, and USB drives. Work inside
the provided PyTorch Docker image, in a folder named after you on the Desktop
(Docker User Guide; HPC Lab Rules 6–7).

1. Log in with the account from your reservation e-mail (`.\<username>`), then start
   **Docker Desktop** (Accept, then "Continue without signing in"). In File Explorer, open
   `This PC > OS (C:) > DockerImages` and double-click the PyTorch image to load it.
2. Create `Desktop\<your name>`. Download this repo's zip from Google Drive (the browser
   is the only allowed transfer) and extract it there, giving `Desktop\<your name>\data266-lab1`.
3. In **Command Prompt** (not PowerShell), using the image name and tag that Docker Desktop shows:
   ```bat
   docker run --gpus all -p 8888:8888 -v "%USERPROFILE%\Desktop\<your name>":/app <IMAGE_NAME>:<TAG>
   ```
   Open the `http://127.0.0.1:8888/...` link it prints. Inside Jupyter, the repo is `/app/data266-lab1`.
4. In a Jupyter **Terminal** (New > Terminal), install the extra packages. The image's own
   torch stays untouched:
   ```bash
   cd /app/data266-lab1
   pip install -r requirements-docker.txt
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   python smoke_test.py
   ```
   The check must print `True` and the GPU name. The image's torch version may differ from
   the pinned 2.4.1. Pre-flight then prints a `warn` line instead of failing, and every
   manifest records the versions actually used.
5. Open each task notebook, set `SMOKE = True` and *Run All*, then `SMOKE = False` and
   *Restart & Run All*. Save each notebook with its outputs.
6. **Before you leave the machine:** in the terminal, run `python pack_results.py`, upload the
   `sneha_results_*.zip` it writes to Google Drive, and check that it opens. Only then delete
   `Desktop\<your name>` and lock the machine (Windows + L), as the guide asks. Sessions are
   wiped after the reservation. If a run is unfinished, add `--with-resume` so that it can continue
   from the last epoch on another machine (`RESUME = True`).

A reservation runs until noon on its last day. You can leave a long run (Task 3) training
and collect results 12:00–12:30 on the end date. Check in for every day of the reservation.

### Your own computer

Python 3.11 (torch 2.4.1 has no wheels for 3.13). From the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
pip install -r requirements.txt              # on Windows install the CUDA torch first, see below
python smoke_test.py
```

On Windows with an NVIDIA GPU, run
`pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121`
**before** `requirements.txt`. Plain `pip` installs a CPU-only torch there, and pip then treats
`torch==2.4.1` as already satisfied.

**GPU:** `device: cuda` in a config means "a GPU or stop with an error", so a CPU-only
torch can never silently turn a GPU run into hours on the CPU. torch 2.4.1 does not support
RTX 50-series cards (5090, sm_120); the code refuses to start on a GPU that its torch build
has no kernels for. For comparable speed metrics (tokens/s, images/s), both members train
each task on the same GPU model.

## Task 1 — Sneha

Everything is in one notebook: **`task1_llm/sneha/src/task1_llm_sneha.ipynb`**. It
reads its settings from `task1_llm/sneha/config.yaml`, then runs the data download, the model written from scratch,
the unit tests, a pre-flight check, training, generation, evaluation and results.

Data: exactly **100K training / 10K validation fixed-length sequences**. A seeded
contiguous stretch of TinyStories is chunked into disjoint 129-character sequences
(128-character context + 1 shifted target); the first 100K train and the next 10K
validate. That is 12.9M training characters and 1,563 steps per epoch at batch 64.

```bash
jupyter notebook task1_llm/sneha/src/task1_llm_sneha.ipynb      # with the venv activated
```

1. **Smoke check:** set `SMOKE = True` in the first cell, then *Run All*. Tiny model,
   160 sequences from the 22 MB validation file, about a minute. Outputs go to
   `task1_llm/sneha/_smoke/` (git-ignored).
2. **Real run:** set `SMOKE = False`, then *Kernel → Restart & Run All*. The tests and the
   pre-flight check run first and stop the notebook if anything is wrong. Pre-flight
   must print `PREFLIGHT PASSED` and show the RTX 4090.
3. **Save the notebook with its outputs.** It is the submitted Task 1 code.

The notebook downloads the TinyStories file it needs (~2.2 GB for the real run). If
that cannot reach Hugging Face, download `TinyStoriesV2-GPT4-train.txt` and
`TinyStoriesV2-GPT4-valid.txt` from
<https://huggingface.co/datasets/roneneldan/TinyStories/tree/main> into
`task1_llm/data/`. If a GPU session ends mid-run, set `RESUME = True` and *Run All*:
training continues from `checkpoints/gpt_last.pt`, which is saved after every epoch.

Before you leave the lab machine, copy back `task1_llm/sneha/` (the saved notebook,
`checkpoints/`, `outputs/`, `metrics_report.csv`), `reproducibility/raw_logs/sneha/`
and `reproducibility/manifests/`.

| Output | Where |
|---|---|
| Every Task 1 metric, one row | `task1_llm/sneha/metrics_report.csv` |
| Loss curves, per-step LR / grad-norm plots | `task1_llm/sneha/outputs/*.png` |
| Generated samples | `task1_llm/sneha/outputs/generations.txt` |
| Best checkpoint (settings + vocab inside) | `task1_llm/sneha/checkpoints/gpt_best.pt` |
| Architecture and hyperparameter justification | `task1_llm/sneha/results.md` |
| Three failure cases | `task1_llm/sneha/failure_analysis.md` |
| Settings snapshot and list of raw logs | `task1_llm/sneha/config_used.yaml`, `logs.md` |
| Manifest (versions, GPU, data + checkpoint hashes, checkpoint → metrics mapping) | `reproducibility/manifests/sneha_task1.json` |

## Task 2 — Sneha

Everything is in one notebook: **`task2_sentiment/sneha/src/task2_sentiment_sneha.ipynb`**.
It contains the settings, the Yelp Polarity download (about 160 MB), the shared split,
preprocessing with its analysis plots, three models trained from scratch (`meanpool`
baseline, `bigru`, `charcnn`), the stopword/stemming ablation, every Task 2 metric,
McNemar tests against the baseline, and the 20-error review candidates.

Run it exactly like Task 1: `SMOKE = True` and *Run All* first (a few thousand reviews,
one epoch), then `SMOKE = False` and *Restart & Run All*, then save it with its outputs.
`RESUME = True` reuses models already trained with the same settings.

**Shared with Ayush:** `task2_sentiment/data/splits/{train,val,test}_idx.npy` and
`split_info.json`. The first run creates them (stratified 100K / 20K from the official
training file, seed 42; the test split is the full official 38K). Every later run loads
them and never redraws them. Commit them, and Ayush must load the same files: the team
accuracy comparison and the McNemar test are only valid on identical test rows.

If the download fails, put `train.csv`, `test.csv` and `readme.txt` from
`yelp_review_polarity_csv.tgz` into `task2_sentiment/data/raw/yelp_review_polarity_csv/`.

| Output | Where |
|---|---|
| Every metric, three rows (one per model) | `task2_sentiment/sneha/metrics_report.csv` |
| Stopword/stemming ablation (validation macro-F1) | `task2_sentiment/sneha/ablation_report.csv` |
| Length / class distribution / class balance plots | `task2_sentiment/sneha/outputs/*.png` |
| Training curve, confusion matrix, reliability diagram per model | `task2_sentiment/sneha/outputs/<model>_*.png` |
| Test probabilities per model | `task2_sentiment/sneha/outputs/<model>_test_probs.npy` |
| 20-error review candidates | `task2_sentiment/sneha/outputs/error_review_candidates.md` |
| Vocabularies, cleaning summary, examples | `task2_sentiment/sneha/data_processed/` |
| Checkpoints | `task2_sentiment/sneha/checkpoints/<model>_best.pt` |
| Write-ups | `task2_sentiment/sneha/results.md`, `failure_analysis.md` |
| Manifest | `reproducibility/manifests/sneha_task2.json` |

## Task 3 — Sneha

Everything is in one notebook: **`task3_gan/sneha/src/task3_gan_sneha.ipynb`**. It
covers data and the shared image lists, a CycleGAN written from scratch (U-Net-256
generators, 70×70 PatchGAN discriminators, LSGAN, cycle and identity losses, image pool),
EMA generator weights and DiffAugment, a run length fitted to the lab session, resume,
per-direction checkpoint selection on a validation set, the Kaggle submission scored by the
**provided** evaluation script, every Task 3 metric, and the blinded human-audit helpers.

Before the first run, copy the course's `Part3/dataset.zip` to `task3_gan/data/dataset.zip`.
The notebook extracts it. Then run as for Tasks 1 and 2: `SMOKE = True` and *Run All*,
then `SMOKE = False` and *Restart & Run All*. `RESUME = True` continues after an
interrupted session.

**Run length:** before the real run, set `train_until` in `task3_gan/sneha/config.yaml` to
when training must end, in lab time **with its offset** (`"2026-10-02T09:00:00-07:00"`;
the Docker clock runs on UTC). With `epochs: auto`, pre-flight prints how many epochs fit.
Training re-plans from its measured speed while the learning rate is still constant, and
always stops by `train_until` after validating, so the submission and metrics cells that
follow finish well before noon. The final schedule is logged and saved; setting `epochs`
and `decay_start_epoch` to those numbers reproduces the run. Leave the machine locked
(Windows + L), not signed out, overnight: signing out stops Docker and the run.

**What happened, and Google Colab:** the lab run of 1 October (`config.yaml`, 400 epochs) stopped
after epoch 73 with no error message: the process was ended on the lab machine.
`checkpoints/cyclegan_last.pt` holds epoch 70, so the run continues on Colab. Section 0 of the
notebook does the setup: it unpacks the project from Drive and backs up to Drive every 20 minutes.
1. **Step A:** `config_colab_submit.yaml`. No training: the submission and metrics from the
   generators saved in the lab.
2. **Step B:** `config_colab_resume.yaml`. It resumes at epoch 70 and decays the learning rate
   linearly to 0 over as many epochs as fit before `train_until` (at most `max_epochs`).

The logs record the GPU of every run (hardware disclosure).

**Shared with Ayush:** `task3_gan/data/{ref_photos_300,eval_photos_300,val_photos_300,photo_subset_2000,photo_train_all_6138}.txt`
and `lists_info.json`. Made once, then only loaded. `ref_photos_300` is exactly the set the
provided script scores Monet → photo against (the first 300 photos by filename), so it is
never trained on. Training uses `photo_train_all_6138` (every photo outside the three
held-out lists; `train_photos: subset` switches back to the 2,000).

**Kaggle submission:**
1. After the real run, `task3_gan/sneha/submission.csv` is the file the provided script
   wrote. `task3_gan/eval/Part3_Evaluation_Script.ipynb` runs unchanged except its `BASE`,
   `GEN_A2B` and `GEN_B2A` path lines, and the log records the replaced lines.
2. Upload that file unchanged, under the team name `PairProgramming_Team_##`.
3. Record the public and private score and the rank in `task3_gan/sneha/results.md`.

Integrity, from the brief: the CSV must come from this model's own inference on the fixed
lists. No hand-picked, edited or copied images, and no typed-in numbers. Anything else
scores zero for Task 3.

| Output | Where |
|---|---|
| Kaggle file (written by the provided script) | `task3_gan/sneha/submission.csv` |
| Every Task 3 metric, both directions | `task3_gan/sneha/metrics_report.csv` |
| Generated images (300 per direction) | `task3_gan/sneha/outputs/pred_A2B/`, `pred_B2A/` |
| Loss curves, per-iteration losses, sample grids | `task3_gan/sneha/outputs/` |
| Generators (each the best of its direction by validation FID; final) | `task3_gan/sneha/checkpoints/` (~435 MB each, git-ignored: share via a drive link) |
| Human audit (team) | `task3_gan/audit/` |
| Write-ups | `task3_gan/sneha/results.md`, `failure_analysis.md` |
| Manifest | `reproducibility/manifests/sneha_task3.json` |

## Reproducibility rules

- **One config file per member and task** holds every hyperparameter and path:
  `task*/<member>/config.yaml`. The notebooks read it and hard-code nothing. The SHA-256
  of the resolved settings is the first line of every raw log, and each run also writes a
  snapshot to `config_used.yaml`.
- **Raw logs are append-only evidence.** They are opened in exclusive-create mode, so a
  run can never overwrite an existing log, and they are never edited afterwards.
- **Manifests** record package versions, GPU, seed, git commit, the checkpoint's
  SHA-256, and which metrics rows each checkpoint produced.
- **Large files:** the raw data is downloaded by the code and is git-ignored. Resume
  checkpoints (`*_last.pt`) are git-ignored. Files over 100 MB go to Git LFS or external
  storage, and their links are recorded in the manifest.
- Seeds are fixed from the config (`seed: 42`). cuDNN runs deterministically in Tasks 1 and 2; Task 3 trades bitwise determinism for speed (`cudnn_benchmark`).

## Layout

```
smoke_test.py         the one-command smoke test
pack_results.py       zips every result for the Google Drive backup
requirements-docker.txt  extra packages for the lab's PyTorch Docker image
common/               shared helper code and its tests (Sneha's notebooks are self-contained and do not import it)
task1_llm/data/       TinyStories files land here (+ .sha256 of each), git-ignored
task*/<member>/       config.yaml, src/<notebook>.ipynb, checkpoints/, outputs/, metrics_report.csv, results.md, failure_analysis.md, ...
task2_sentiment/      data/splits/ (shared split indices, committed), data/raw/ (git-ignored), <member>/
task3_gan/            data/ (dataset.zip + extracted images, git-ignored; shared lists, committed), eval/ (provided script), audit/, <member>/
reproducibility/      manifests/, raw_logs/<member>/
report/               DATA266_Lab1_Report_Team_[N].pdf
```

## References

1. A. Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017.
2. R. Eldan and Y. Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" arXiv:2305.07759, 2023.
3. J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks," *ICCV*, 2017.
