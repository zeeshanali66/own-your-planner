"""Own your planner: can a small offline model improve itself from its own simulator-verified successes?

Inspired by YC's Winter 2027 request "Own Your Intelligence" (turn production failures into training improvements)
and "AI for the Air-Gapped Economy" (everything runs offline on one 8 GB GPU).

Task: a home robot must bring an object to a room in a randomly generated house (6-8 rooms; some doors locked, so
the short route may be blocked). A simulator checks every plan: legal moves, picking, placing, goal reached.
  base        - Qwen2.5-1.5B-Instruct, zero-shot
  fewshot     - the same model with 2 worked examples in the prompt (the "prompt fix")
  self_r1/r2  - expert iteration: sample 8 plans per training task, keep only simulator-verified successes, fine-tune
                (LoRA), repeat with the improved model
  oracle      - LoRA on optimal plans from a classical search planner (the "teacher" upper reference)
  ref 12B     - Gemma 4 12B and Qwen 3.5 9B, zero-shot via Ollama
Test on UNSEEN houses (test, 6-8 rooms) and LARGER houses (test_big, 10-12 rooms).
Commands: tasks | ref | eval <tag> | sample <round> | train <tag> | results
"""
import json, os, random, re, sys, time, urllib.request
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES, ADP = (os.path.join(HERE, d) for d in ("data", "results", "adapters"))
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
NAMES = ["kitchen", "hallway", "living room", "bedroom", "bathroom", "garden", "study", "garage", "laundry room",
         "dining room", "office", "guest room", "pantry", "balcony"]
NEW_NAMES = ["attic", "workshop", "nursery", "porch", "basement", "library", "gym", "studio", "cellar", "sunroom"]
OBJECTS = ["cup", "book", "phone", "towel", "remote", "keys", "umbrella", "charger"]
K_SAMPLES, K_HARD = 8, 16
N_SEED = 50          # seeded variant: optimal plans for 50 random training houses ("a few verified logs")


def seed_ids():
    ids = [h["id"] for h in jl(os.path.join(DATA, "train.jsonl"))]
    return sorted(random.Random("seed-set").sample(ids, N_SEED))
SPLITS = ["test", "test_big", "test_names"]


# ---------------- environment ----------------
def adj(doors, locked):
    lk = {frozenset(d) for d in locked}
    a = {}
    for x, y in doors:
        a.setdefault(x, set()); a.setdefault(y, set())
        if frozenset((x, y)) not in lk:
            a[x].add(y); a[y].add(x)
    return a


def bfs(a, s, t):
    prev, q = {s: None}, deque([s])
    while q:
        u = q.popleft()
        for v in sorted(a.get(u, ())):
            if v not in prev:
                prev[v] = u; q.append(v)
    if t not in prev:
        return None
    p, u = [], t
    while u is not None:
        p.append(u); u = prev[u]
    return p[::-1]


def optimal(h):
    a = adj(h["doors"], h["locked"])
    p1 = bfs(a, h["robot"], h["objects"][h["target"]]); p2 = bfs(a, p1[-1], h["goal"])
    return [{"action": "go", "room": r} for r in p1[1:]] + [{"action": "pick", "object": h["target"]}] + \
           [{"action": "go", "room": r} for r in p2[1:]] + [{"action": "place", "object": h["target"]}]


def simulate(h, plan):
    if not isinstance(plan, list):
        return False, "no plan"
    a = adj(h["doors"], h["locked"])
    room, held, objs = h["robot"], None, dict(h["objects"])
    for i, s in enumerate(plan):
        if not isinstance(s, dict):
            return False, f"step {i}: not an action"
        act = s.get("action")
        if act == "go":
            if s.get("room") not in a.get(room, ()):
                locked = any(frozenset((room, s.get("room"))) == frozenset(d) for d in h["locked"])
                return False, "locked door" if locked else "no door"
            room = s.get("room")
        elif act == "pick":
            if held or objs.get(s.get("object")) != room:
                return False, "bad pick"
            held = s.get("object"); objs[held] = "robot"
        elif act == "place":
            if not held:
                return False, "bad place"
            objs[held] = room; held = None
        else:
            return False, "unknown action"
    ok = objs.get(h["target"]) == h["goal"]
    return ok, "ok" if ok else "goal not reached"


def make_house(seed, lo, hi, names=NAMES):
    rng = random.Random(f"house-{seed}")
    while True:
        rooms = rng.sample(names, rng.randint(lo, hi))
        doors = [(rooms[i], rng.choice(rooms[:i])) for i in range(1, len(rooms))]   # random spanning tree
        for _ in range(rng.randint(1, 3)):                                          # a few extra doors (loops)
            x, y = rng.sample(rooms, 2)
            if not any(frozenset((x, y)) == frozenset(d) for d in doors):
                doors.append((x, y))
        objs = {o: rng.choice(rooms) for o in rng.sample(OBJECTS, 5)}
        tgt = rng.choice(list(objs))
        robot = rng.choice([r for r in rooms if r != objs[tgt]])
        goal = rng.choice([r for r in rooms if r != objs[tgt]])
        h = {"id": seed, "rooms": rooms, "doors": [list(d) for d in doors], "locked": [], "objects": objs,
             "target": tgt, "goal": goal, "robot": robot}
        # lock a door on the free shortest route if a detour exists (in ~60% of houses): forces search, not guessing
        if rng.random() < 0.6:
            free = optimal(h)
            route = [robot] + [s["room"] for s in free if s["action"] == "go"]
            cands = list(zip(route, route[1:])); rng.shuffle(cands)
            for x, y in cands:
                h2 = dict(h, locked=[[x, y]])
                a = adj(h2["doors"], h2["locked"])
                if bfs(a, robot, objs[tgt]) and bfs(a, objs[tgt], goal):
                    h = h2; break
        if len(optimal(h)) >= 4:
            return h


def describe(h):
    doors = "; ".join(f"{x} - {y}" for x, y in h["doors"])
    lk = "; ".join(f"{x} - {y}" for x, y in h["locked"]) or "none"
    objs = "; ".join(f"{o}: {r}" for o, r in h["objects"].items())
    return (f"House doors (each connects two rooms): {doors}.\nLocked doors (cannot be used): {lk}.\n"
            f"Robot: in the {h['robot']}, hands empty.\nObjects: {objs}.\nTask: bring the {h['target']} to the {h['goal']}.")


SYS = ("You are the planner of a home robot. Actions: {\"action\": \"go\", \"room\": R} moves to an adjacent room "
       "through an open door; {\"action\": \"pick\", \"object\": O} picks up an object in the current room; "
       "{\"action\": \"place\", \"object\": O} puts the held object down in the current room. Reply only with JSON: "
       "{\"plan\": [actions...]}")


def user_msg(h, fewshot=None):
    ex = ""
    if fewshot:
        ex = "Examples of correct plans:\n" + "\n\n".join(
            describe(e) + "\nPlan: " + json.dumps({"plan": optimal(e)}) for e in fewshot) + "\n\nNow your task.\n"
    return ex + describe(h) + "\nWrite the full plan."


def parse(text):
    i = text.find("{")
    if i < 0:
        return None
    try:
        j, _ = json.JSONDecoder().raw_decode(text[i:])
        return j.get("plan") if isinstance(j, dict) else None
    except Exception:
        return None


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def wjl(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_tasks():
    # curriculum: 200 small houses (4-5 rooms) to get self-training started, then 400 normal ones (6-8 rooms)
    train = [make_house(s, 4, 5) for s in range(50_000, 50_200)] + [make_house(s, 6, 8) for s in range(400)]
    test = [make_house(s, 6, 8) for s in range(10_000, 10_100)]
    big = [make_house(s, 10, 12) for s in range(20_000, 20_100)]
    names = [make_house(s, 6, 8, names=NEW_NAMES) for s in range(30_000, 30_100)]   # rooms never seen in training
    for name, rows in (("train", train), ("test", test), ("test_big", big), ("test_names", names)):
        for h in rows:
            assert simulate(h, optimal(h))[0]
        wjl(os.path.join(DATA, f"{name}.jsonl"), rows)
    tr = {json.dumps([sorted(map(sorted, h["doors"])), h["objects"]]) for h in train}
    assert not any(json.dumps([sorted(map(sorted, h["doors"])), h["objects"]]) in tr for h in test + big + names)
    assert not any(r in NAMES for h in names for r in h["rooms"])
    # general-skill check: 100 arithmetic word problems (does fine-tuning break other abilities?)
    rng = random.Random("sanity")
    san = []
    for i in range(100):
        a, b, c = rng.randint(10, 90), rng.randint(2, 9), rng.randint(5, 60)
        san.append({"id": i, "q": f"A shop has {a} apples. It sells {b} boxes of 3 apples each and then receives {c} "
                                   "more apples. How many apples does it have now? Answer with just the number.",
                    "a": a - 3 * b + c})
    wjl(os.path.join(DATA, "sanity.jsonl"), san)
    print(f"tasks ok: train {len(train)} (200 small + 400 normal), test/test_big/test_names 100 each, sanity 100;",
          "optimal plans verified; no test house in training; test_names uses only unseen room names;",
          f"locked-door share in train {sum(bool(h['locked']) for h in train) / len(train):.2f}")


# ---------------- reference 12B models via Ollama ----------------
def ollama(model, msgs):
    extra = {"think": False} if model.startswith(("qwen3", "gemma4")) else {}
    body = json.dumps({"model": model, "stream": False, "format": "json", **extra, "messages": msgs,
                       "options": {"temperature": 0, "seed": 42, "num_predict": 800, "num_ctx": 4096}}).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())["message"]["content"]
        except Exception as e:
            print("retry", e, flush=True); time.sleep(60)
    raise RuntimeError("Ollama failed 3 times; stopping instead of scoring an empty answer")


def cmd_ref():
    for m in ["gemma4:12b", "qwen3.5:9b"]:
        for split in SPLITS:
            path = os.path.join(RES, f"pred_ref-{m.replace(':', '_')}_{split}.jsonl")
            if os.path.exists(path):
                continue
            rows = []
            for h in jl(os.path.join(DATA, f"{split}.jsonl")):
                t0 = time.time()
                raw = ollama(m, [{"role": "system", "content": SYS}, {"role": "user", "content": user_msg(h)}])
                ok, err = simulate(h, parse(raw))
                rows.append({"id": h["id"], "ok": ok, "err": err, "locked": bool(h["locked"]),
                             "sec": round(time.time() - t0, 2), "raw": raw[:800]})
            wjl(path, rows); print(m, split, sum(r["ok"] for r in rows), flush=True)
        urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/generate",
                                                      data=json.dumps({"model": m, "keep_alive": 0}).encode()))


# ---------------- small model (HF) ----------------
def load(adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": os.environ.get("OWN_DEVICE", "cuda:0")})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, os.path.join(ADP, adapter))
    model.eval()
    return tok, model


def chat_ids(tok, h, fewshot=None):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user_msg(h, fewshot)}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def generate(tok, model, prompts, sample=False, k=1, bs=8):
    import torch
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=400, do_sample=sample, temperature=0.8 if sample else None,
                               top_p=0.95 if sample else None, top_k=None, num_return_sequences=k,
                               pad_token_id=tok.pad_token_id)
        txt = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        outs += [txt[j * k:(j + 1) * k] for j in range(len(txt) // k)]
    return outs


def cmd_eval(tag):
    adapter = None if tag in ("base", "fewshot") else tag
    tok, model = load(adapter)
    shots = jl(os.path.join(DATA, "train.jsonl"))[:2] if tag == "fewshot" else None
    for split in SPLITS:
        hs = jl(os.path.join(DATA, f"{split}.jsonl"))
        t0 = time.time()
        outs = generate(tok, model, [chat_ids(tok, h, shots) for h in hs])
        sec = (time.time() - t0) / len(hs)
        rows = []
        for h, o in zip(hs, outs):
            ok, err = simulate(h, parse(o[0]))
            rows.append({"id": h["id"], "ok": ok, "err": err, "locked": bool(h["locked"]), "sec": round(sec, 3),
                         "raw": o[0][:800]})
        wjl(os.path.join(RES, f"pred_{tag}_{split}.jsonl"), rows)
        print(tag, split, sum(r["ok"] for r in rows), "/", len(rows), flush=True)
    san = jl(os.path.join(DATA, "sanity.jsonl"))
    prompts = [tok.apply_chat_template([{"role": "user", "content": q["q"]}], add_generation_prompt=True, tokenize=False)
               for q in san]
    outs = generate(tok, model, prompts, bs=16)
    rows = []
    for q, o in zip(san, outs):
        nums = re.findall(r"-?[0-9]+", o[0])
        rows.append({"id": q["id"], "ok": bool(nums) and int(nums[-1]) == q["a"], "raw": o[0][:200]})
    wjl(os.path.join(RES, f"sanity_{tag}.jsonl"), rows)
    print(tag, "sanity", sum(r["ok"] for r in rows), "/ 100", flush=True)


def cmd_sample(rnd, seeded=False):
    """Expert iteration round: sample K plans per training task with the current model, keep verified successes.
    seeded=True: the chain starts from the 'seed' adapter (trained on N_SEED optimal plans) and the seed plans are
    kept in the training set."""
    rnd = int(rnd)
    pre, prev_tag = ("s", "sself_r") if seeded else ("r", "self_r")
    tok, model = load(f"{prev_tag}{rnd - 1}" if rnd > 1 else ("seed" if seeded else None))
    hs = jl(os.path.join(DATA, "train.jsonl"))
    if rnd > 1:
        keep = {r["id"]: r for r in jl(os.path.join(DATA, f"sft_{pre}{rnd - 1}.jsonl"))}
    elif seeded:
        byid = {h["id"]: h for h in hs}
        keep = {i: {"id": i, "plan": optimal(byid[i])} for i in seed_ids()}
    else:
        keep = {}
    # every task gets K_SAMPLES tries; from round 2 on, tasks never solved so far get K_HARD tries instead
    # ("spend more effort on the failures")
    hard = [h for h in hs if h["id"] not in keep] if rnd > 1 else []
    easy = [h for h in hs if h["id"] in keep] if rnd > 1 else hs
    outs = generate(tok, model, [chat_ids(tok, h) for h in easy], sample=True, k=K_SAMPLES, bs=4)
    if hard:
        outs += generate(tok, model, [chat_ids(tok, h) for h in hard], sample=True, k=K_HARD, bs=2)
    hs = easy + hard
    solved_now = 0
    for h, os_ in zip(hs, outs):
        good = [p for p in (parse(o) for o in os_) if simulate(h, p)[0]]
        if good:
            solved_now += 1
            best = min(good, key=len)
            if h["id"] not in keep or len(best) < len(keep[h["id"]]["plan"]):
                keep[h["id"]] = {"id": h["id"], "plan": best}
    wjl(os.path.join(DATA, f"sft_{pre}{rnd}.jsonl"), list(keep.values()))
    json.dump({"round": rnd, "seeded": seeded, "tasks_with_a_success": solved_now, "training_examples": len(keep),
               "hard_tasks_retried": len(hard), "train_tasks": len(hs)},
              open(os.path.join(RES, f"sample_{pre}{rnd}.json"), "w"))
    print(f"round {rnd}: {solved_now}/{len(hs)} tasks had a verified success; {len(keep)} training examples", flush=True)


def cmd_train(tag):
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0); random.seed(0)
    hs = {h["id"]: h for h in jl(os.path.join(DATA, "train.jsonl"))}
    if tag == "oracle":
        data = [{"id": i, "plan": optimal(h)} for i, h in hs.items()]
    elif tag == "seed":
        data = [{"id": i, "plan": optimal(hs[i])} for i in seed_ids()]
    elif tag.startswith("sself_r"):
        data = jl(os.path.join(DATA, f"sft_s{tag[len('sself_r'):]}.jsonl"))
    else:
        data = jl(os.path.join(DATA, f"sft_{tag.split('_')[1]}.jsonl"))
    assert data, f"no training examples for {tag}"
    tok, model = load()
    tok.padding_side = "right"
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    ex = []
    for d in data:
        p = tok(chat_ids(tok, hs[d["id"]]), add_special_tokens=False)["input_ids"]
        a = tok(json.dumps({"plan": d["plan"]}) + tok.eos_token, add_special_tokens=False)["input_ids"]
        ex.append((p + a, [-100] * len(p) + a))
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=2e-4)
    model.train(); losses = []
    # small datasets (the 50-plan seed) get more epochs so the number of updates is comparable
    EPOCHS, BS, ACC = (6 if len(ex) < 100 else 2), 2, 4      # effective batch 8
    for ep in range(EPOCHS):
        random.shuffle(ex)
        for b in range(0, len(ex), BS * ACC):
            for k in range(b, min(b + BS * ACC, len(ex)), BS):
                batch = ex[k:k + BS]; L = max(len(x) for x, _ in batch)
                ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in batch], device=model.device)
                lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch], device=model.device)
                att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in batch], device=model.device)
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / ACC
                loss.backward(); losses.append(round(float(loss.detach()) * ACC, 4))
            opt.step(); opt.zero_grad()
        print(tag, "epoch", ep, "loss", losses[-1], flush=True)
    model.save_pretrained(os.path.join(ADP, tag))
    json.dump({"tag": tag, "examples": len(ex), "epochs": EPOCHS, "batch": BS * ACC, "lr": 2e-4, "r": 16, "losses": losses},
              open(os.path.join(ADP, tag, "train_log.json"), "w"))


def cmd_results():
    import glob
    import numpy as np, pandas as pd
    rows = []
    for p in glob.glob(os.path.join(RES, "pred_*.jsonl")):
        name = os.path.basename(p)[5:-6]
        split = next(sp for sp in ("test_big", "test_names", "test") if name.endswith(sp))
        tag = name[: -len(split) - 1]
        for r in jl(p):
            rows.append({"system": tag, "split": split, "sec": r.get("sec", float("nan")),
                         **{k: r[k] for k in ("id", "ok", "err", "locked")}})
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)

    def ci(x):
        x = np.asarray(x, float); bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
        return f"{x.mean():.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]"

    order = ["base", "fewshot", "self_r1", "self_r2", "self_r3", "seed", "sself_r1", "sself_r2", "sself_r3", "oracle",
             "ref-qwen3.5_9b", "ref-gemma4_12b"]
    out = ["# Own your planner: results", "", "Success = the plan runs in the simulator and the object ends in the goal "
           "room. 95% bootstrap CIs. test = 100 unseen houses (6-8 rooms); test_big = 100 larger houses (10-12 rooms).", "",
           "| system | test (unseen houses) | test_big (10-12 rooms) | test_names (unseen room names) | "
           "locked-door houses (test) | sec per plan | arithmetic sanity |", "|---|---|---|---|---|---|---|"]
    for s in [o for o in order if o in set(df.system)]:
        g = df[df.system == s]
        san = jl(os.path.join(RES, f"sanity_{s}.jsonl"))
        out.append(f"| {s} | {ci(g[g.split == 'test'].ok)} | {ci(g[g.split == 'test_big'].ok)} | "
                   f"{ci(g[g.split == 'test_names'].ok)} | {g[(g.split == 'test') & g.locked].ok.mean():.2f} | "
                   f"{g.sec.mean():.2f} | {(sum(r['ok'] for r in san) / 100) if san else '-'} |")
    for p in sorted(glob.glob(os.path.join(RES, "sample_r*.json"))) + sorted(glob.glob(os.path.join(RES, "sample_s*.json"))):
        j = json.load(open(p))
        out.append(f"\n{'Seeded self-training' if j.get('seeded') else 'Self-training from zero'}, round {j['round']}: {j['tasks_with_a_success']}/{j.get('train_tasks', 400)} training tasks had at least one "
                   f"verified success among {K_SAMPLES} samples; {j['training_examples']} training examples.")
    out += ["", "## Why plans fail (share of all plans)", "", "```",
            df.groupby(["system", "split"]).err.value_counts(normalize=True).unstack(fill_value=0).round(2).to_string(), "```"]
    # paired tests vs base on test
    out += ["", "## Paired tests vs base (same houses; sign-flip permutation; Holm)", ""]
    base = df[(df.system == "base")].set_index(["split", "id"]).ok.astype(float)
    tests = []
    for s in [o for o in order if o in set(df.system) and o != "base"]:
        x = df[df.system == s].set_index(["split", "id"]).ok.astype(float)
        d = (x - base.reindex(x.index)).dropna().values
        obs = d.mean(); p = (sum(abs((d * rng.choice([-1, 1], len(d))).mean()) >= abs(obs) for _ in range(5000)) + 1) / 5001
        tests.append([s, obs, p])
    srt = sorted(range(len(tests)), key=lambda i: tests[i][2]); run = 0
    for r, i in enumerate(srt):
        run = max(run, min(1, tests[i][2] * (len(tests) - r))); tests[i].append(run)
    for s, obs, p, ph in tests:
        out.append(f"- {s} minus base: {obs:+.2f} (p = {p:.4f}, Holm p = {ph:.4f})")
    os.makedirs(RES, exist_ok=True)
    open(os.path.join(RES, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


def cmd_smoke():
    """CPU smoke test of every code path on 2 houses (no GPU needed)."""
    os.environ["OWN_DEVICE"] = "cpu"
    import torch
    tok, model = load()
    hs = jl(os.path.join(DATA, "test.jsonl"))[:2]
    o = generate(tok, model, [chat_ids(tok, h) for h in hs], sample=True, k=2, bs=2)
    print("sampled", len(o), "x", len(o[0]), "| parsed:", [parse(x) is not None for x in o[0]], "|", o[0][0][:200])
    from peft import LoraConfig, get_peft_model
    tok.padding_side = "right"
    m = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]))
    p = tok(chat_ids(tok, hs[0]), add_special_tokens=False)["input_ids"]
    a = tok(json.dumps({"plan": optimal(hs[0])}) + tok.eos_token, add_special_tokens=False)["input_ids"]
    loss = m(input_ids=torch.tensor([p + a]), labels=torch.tensor([[-100] * len(p) + a])).loss
    loss.backward(); print("train step ok, loss", round(float(loss), 3))


if __name__ == "__main__":
    c = sys.argv[1]
    {"tasks": cmd_tasks, "ref": cmd_ref, "results": cmd_results, "smoke": cmd_smoke}.get(c, lambda: None)()
    if c == "eval": cmd_eval(sys.argv[2])
    if c == "sample": cmd_sample(sys.argv[2])
    if c == "ssample": cmd_sample(sys.argv[2], seeded=True)
    if c == "train": cmd_train(sys.argv[2])
