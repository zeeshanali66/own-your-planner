# Research log: own your planner

### 1. Why
Earlier studies in this series used large local models as planners. A practical question for robots and offline
deployments is whether a *small* model that fits on one consumer GPU can improve itself from its own successes, with a
simulator as the only teacher. Expert iteration / STaR-style self-training does this for reasoning tasks; here the
verifier is exact (the simulator), so every kept example is guaranteed correct.

### 2. Design choices
- Locked doors on the shortest route in ~60% of houses, so a plan cannot succeed by guessing the obvious route.
- Three test splits (unseen houses, bigger houses, unseen room names) to separate learning the skill from memorising
  houses or names.
- A curriculum of 200 small houses (4-5 rooms) in the training set, to give self-training an easier start.
- An arithmetic side check to detect damage to other skills after fine-tuning.

### 3. What happened
- Base: 0/100 on every split; few-shot: 4/100 on `test`. Self-training round 1: 8 samples x 600 houses = 4,800 plans,
  **zero verified successes**, including the 200 small houses. Rounds 2-3 and their evaluations failed because there
  was no adapter to load (the pipeline continued past the empty round; the steps simply errored).
- The oracle arm (LoRA on optimal plans) ran out of GPU memory at batch size 8. Fixed afterwards: batch 2 with
  gradient accumulation 4 (same effective batch), tested on CPU, not yet run on GPU.
- Ollama's Gemma 4 12B reference finished `test` (78/100). The other splits failed when Ollama's llama-server crashed
  on load (CUDA "shared object initialization failed", exit 0xc0000409), most likely because system RAM was low while
  another job ran. The client now retries and then stops, instead of recording an empty answer as a failed plan. The
  finished `test` file was checked: no empty answers.

### 4. Checking the failure analysis
- First reading of a few raw outputs suggested the few-shot model copied objects from the examples. Counting showed
  this was wrong: 4 of 95 parsed plans pick the wrong object and none of those comes from the examples. The README
  reports the counted result.
- 90 of 100 base plans start with "go" to the robot's current room, which the simulator counts as an illegal move.
  To make sure the strict simulator was not the whole story, all plans were re-scored with such no-op moves removed:
  base stays at 0/100 on all three splits (the next action picks the object in the wrong room), few-shot at 4/1/3,
  Gemma 4 79/100. The conclusion does not depend on the strict rule.

### 5. Next
- Seeded expert iteration: LoRA on 50 verified plans, then self-training rounds; compare with the oracle trained on
  all 600 plans. This asks how much self-training multiplies a small seed, which is the realistic setting ("a few
  logged successes").
- A larger small model (3-4B) to find where zero-shot success becomes non-zero.
- A sanity check whose answer format the model follows, so damage from fine-tuning can be measured.
