# Running the L-CDAIDS gateway-hardware benchmark on your Raspberry Pi

This measures real CPU, memory, inference latency, and retraining latency
for L-CDAIDS on your Pi 4, to directly answer the reviewers' request for
gateway-class hardware validation (not just workstation numbers).

## 1. Get the files onto the Pi

Copy these onto the Pi, into the **same folder**:

- `pi_benchmark_lcdaids.py` (in this bundle)
- `final_sample_200000.csv` — copy this from your own machine at
  `G:\IoT_IDS_Research\data\processed\final_sample_200000.csv` (it's the
  same 200k-record sample the paper's main drift tables already use, ~40 MB)

Any of these work: a USB stick, `scp` if SSH is enabled on the Pi, or
copying via a shared network folder. If you're not sure how to get a file
onto the Pi, the easiest is usually a USB stick: copy both files onto it
from your Windows machine, plug it into the Pi, then copy them from the
stick into a folder like `~/lcdaids_bench/`.

## 2. Open a terminal on the Pi and check Python

```
python3 --version
```

Raspberry Pi OS ships Python 3.9 or newer — that's fine, no need to install
a different version.

## 3. Install the Python packages

```
cd ~/lcdaids_bench        # or wherever you put the two files
python3 -m pip install --upgrade pip
python3 -m pip install pandas numpy scikit-learn psutil river
```

Raspberry Pi OS's `pip` is already configured to pull prebuilt ARM wheels
from piwheels.org for most packages, so this should just download binaries
— it should **not** need to compile anything, and should take a few minutes
at most.

**If the `river` install hangs or tries to compile for a long time**, you
can skip it — just run:

```
python3 -m pip install pandas numpy scikit-learn psutil
```

The benchmark script detects when `river` isn't installed and automatically
falls back to a simpler drift detector, so it still runs and still produces
valid timing/memory numbers; only the exact retrain-trigger points may shift
slightly, which doesn't affect the CPU/memory/latency numbers we need.

If `pip` for some reason tries to build `scikit-learn` or `numpy` from
source (you'll see it says "Building wheel for numpy..." and it runs for a
very long time), stop it (Ctrl+C) and instead run:

```
sudo apt update
sudo apt install python3-numpy python3-pandas python3-sklearn
```

then retry the pip install for the remaining packages — this uses the
system's own prebuilt packages instead.

## 4. Run the benchmark

Run each seed one at a time (keeps each run short, and if the Pi needs a
break or restart partway through, you don't lose earlier seeds):

```
python3 pi_benchmark_lcdaids.py --seed 42
python3 pi_benchmark_lcdaids.py --seed 43
python3 pi_benchmark_lcdaids.py --seed 44
python3 pi_benchmark_lcdaids.py --aggregate
```

Each `--seed` run takes roughly a minute or two on a Pi 4 (it fits two
100-tree random forests on 160,000 rows, then runs a 120,000-row stream with
on-the-fly retraining) — expect it to be noticeably slower than a laptop,
which is exactly the point.

You'll see live output like:

```
  fit (Top-15, 160,000 rows): 23.6s, peak RSS 478 MB, mean CPU 50%
  inference (Top-15 static, 120,000 rows): 1.6s total, 13.1 us/instance
  drift-triggered retraining: 1 retrain(s), mean 2.3s each, peak RSS 616 MB
```

(Those exact numbers are from testing the script, not from a Pi — your
Pi's numbers, especially the timings, will likely be quite different,
probably slower on CPU-bound steps. That's expected and is exactly the
real-hardware data we need.)

## 5. Send the results back

After `--aggregate` finishes, a `pi_results/` folder will have been created
next to the script, containing:

- `pi_summary_per_seed.csv` — the headline numbers (fit time, peak memory,
  inference latency, retraining latency, CPU%) per seed
- `pi_phasewise_accuracy.csv` — accuracy/F1/FPR per phase, to sanity-check
  that the Pi reproduces the same detection quality as the workstation
- `pi_result_seed42.json`, `_seed43.json`, `_seed44.json` — full detail,
  including every individual retrain event's timing/memory

Send me all of these (the CSVs are small; the JSONs are a few KB each) and
I'll fold the real numbers into the paper's cost table and deployment
discussion.

## Second benchmark: the exact `tab:cost` / Table 5-6 protocol

You already have `pi_benchmark_lcdaids.py`'s results (those matched Table
`tab:fifteen`'s configuration). There's a second script in this bundle,
`pi_benchmark_tabcost.py`, that instead reproduces the **exact** protocol
behind Table `tab:cost` and the main Tables 5/6 (fixed 80:20 split, 50-tree
`class_weight="balanced"` forests) -- confirmed by extracting the logic
directly from the recovered original notebook, not reconstructed from the
paper's prose. This lets the Pi numbers go directly into `tab:cost` next to
the ARF comparison.

It uses the same `final_sample_200000.csv` you already copied over -- no new
data file needed. Run it the same way:

```
python3 pi_benchmark_tabcost.py --seed 42
python3 pi_benchmark_tabcost.py --seed 43
python3 pi_benchmark_tabcost.py --seed 44
python3 pi_benchmark_tabcost.py --aggregate
```

This one is noticeably faster per seed than the first script (50 trees
instead of 100, and no max-depth cap changes little here) -- expect roughly
15-20s for the fit step rather than ~80s, based on testing. It also expects
occasional retrains with bigger phase-3 swings between seeds than the first
script, because this is the exact configuration the paper's own Table 6
documents as more seed-sensitive (that instability is a real, already-
reported property of this configuration, not a bug in the script).

Results land in a separate `pi_results_tabcost/` folder next to the first
one's `pi_results/`, so nothing gets overwritten. Send both
`pi_tabcost_summary_per_seed.csv` and `pi_tabcost_phasewise_accuracy.csv`
back (plus the JSONs if convenient), the same way as before.

## Optional: the ARF baseline comparison

The paper's Table (`tab:cost`) also compares against the Adaptive Random
Forest (ARF) baseline. That comparison is **not** included in this bundle
because ARF is a pure-Python, per-instance learner that is already
noted in the paper as "far slower than batch RF" even on a workstation —
on a Pi 4 it could plausibly take hours per seed. If you'd like that
comparison too, say so and I'll prepare a separate, resumable/checkpointed
version you can run in the background over however long it takes, rather
than bundling it with the main benchmark above.
