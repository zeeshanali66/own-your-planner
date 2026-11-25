# Own your planner: can a small offline model teach itself to plan from its own verified successes?

A home robot must bring an object to a room in a randomly generated house. Some doors are locked, so the short route
may be blocked. Every plan is executed by a simulator that checks doors, locks, picking and placing. The question:
can a small model (Qwen2.5-1.5B-Instruct, runs offline on one 8 GB GPU) improve itself by **expert iteration**:
sample several plans per training house, keep only the ones the simulator verifies, fine-tune on them, repeat?

**Short answer: not from zero.** The 1.5B model solved **0 of 600** training houses in **4,800 samples**, so the
loop never got a single example to learn from. The failure is specific and worth knowing (see below).

## Design
- **Houses**: 6-8 rooms with a random spanning tree of doors plus 1-3 extra doors (so loops and detours exist).
  In ~60% of houses a door on the shortest route is locked and a detour exists. Every optimal plan has at least 4
  actions and is checked by the simulator.
- **Training set**: 600 houses (200 small ones with 4-5 rooms as a curriculum, 400 with 6-8 rooms).
- **Test sets** (no house appears in training): `test` 100 unseen houses; `test_big` 100 houses with 10-12 rooms;
  `test_names` 100 houses that use only room names never seen in training.
- **Systems**: `base` (zero-shot); `fewshot` (2 worked examples in the prompt); `self_r1` (expert iteration round
  1: 8 samples per training house at temperature 0.8, keep verified successes, LoRA fine-tune); `ref` Gemma 4 12B
  zero-shot through Ollama as a larger-model reference.
- **Side check**: 100 arithmetic word problems, to see whether fine-tuning damages other skills.
- Statistics: 95% bootstrap CIs; paired sign-flip permutation tests against `base` with Holm correction.

## Results

| system | test (unseen houses) | test_big (10-12 rooms) | test_names (unseen room names) | locked-door houses |
|---|---|---|---|---|
| base (Qwen2.5-1.5B) | 0.00 [0.00, 0.00] | 0.00 | 0.00 | 0.00 |
| fewshot | 0.04 [0.01, 0.08] | 0.00 | 0.00 | 0.00 |
| Gemma 4 12B, zero-shot | **0.78 [0.69, 0.86]** | not run | not run | 0.72 |

- Gemma 4 vs base: +0.78, Holm p = 0.0004. Few-shot vs base: +0.01, Holm p = 0.13 (no reliable gain).
- **Self-training round 1: 0/600 training houses had even one verified success among 8 samples.** Expert iteration
  has nothing to train on, so rounds 2-3 could not start.

### Why the small model fails
1. **It does not navigate at all.** In 90 of 100 test plans, the base model's first action is "go to" the room it is
   already standing in, followed by picking the object up there. Even when that no-op move is forgiven (a lenient
   re-scoring), success stays at **0/100** on every split, because the object is not in that room ("bad pick" in
   about 90% of plans). The model treats the task as "pick, place" and ignores the house.
2. **Examples change the form, not the skill.** With two worked examples the model starts moving (illegal first moves
   drop from 96 to 66 of 100), but it walks through doors that do not exist (52% of plans under lenient scoring).
   It does not copy objects from the examples (only 4 of 95 parsed plans pick the wrong object, none from the
   examples); it copies the *shape* of a plan without the path-finding behind it.
3. **Bigger is not just better, it is a different regime.** Gemma 4 12B plans correctly 78% of the time zero-shot;
   its remaining errors are mostly locked doors (11%).

### What this means
Expert iteration (STaR-style self-training) needs a non-zero starting success rate: sampling more cannot find what the
model never produces. For a small offline model on a structured task, the first step has to come from outside, for
example a small set of verified demonstrations, before self-improvement can multiply it. The repository includes that
next experiment (`train seed`: LoRA on 50 optimal plans, then `ssample`/`train sself_rN` rounds, and an `oracle` arm
trained on all 600 optimal plans). The code is tested on CPU, but **those arms were not run** for this report.

## Reproduce
```
python own_study.py tasks           # houses, splits, checks
python own_study.py ref             # Gemma 4 12B / Qwen 3.5 9B via Ollama
python own_study.py eval base       # also: eval fewshot
python own_study.py sample 1        # expert iteration round 1
python own_study.py results         # writes results/RESULTS.md
# not run yet: train seed; eval seed; ssample 1; train sself_r1; train oracle; eval oracle
```
Hardware: RTX 3060 Ti (8 GB). `watchdog.py` stops a run if the GPU reaches 84 C.

## Limitations
One small model family (Qwen2.5-1.5B); temperature-0.8 sampling only; the Gemma 4 reference was run on `test` only
(an Ollama crash stopped the other splits); the seeded and oracle arms are not run. The arithmetic side check scored
5/100 for both base and few-shot (the model rarely answers with just the number), so it is not informative yet.

## Files
- `own_study.py`: houses, simulator, prompts, evaluation, sampling, LoRA training, results.
- `data/`: training and test houses, sanity questions. `results/`: predictions and `RESULTS.md`.
- `docs/RESEARCH_LOG.md`: decisions, problems, and what changed along the way.

License: MIT.
