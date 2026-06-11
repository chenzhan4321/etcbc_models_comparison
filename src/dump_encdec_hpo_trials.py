#!/usr/bin/env python3
"""把 ssi_morphology 下 encoder-decoder HPO 的所有 study DB 里全部 trial 导成一张 CSV +
一份可读 summary,供审稿人透明性核查。"""
import csv
import glob
import os

import optuna

OUT = "models_1_revision_export"
os.makedirs(OUT, exist_ok=True)
rows = []
for db in sorted(glob.glob("hpo_results/*.db")):
    storage = "sqlite:///" + db
    try:
        names = optuna.study.get_all_study_names(storage)
    except Exception as e:  # noqa: BLE001
        print("skip", db, e)
        continue
    for name in names:
        s = optuna.load_study(study_name=name, storage=storage)
        for t in s.trials:
            r = {"db": os.path.basename(db), "study": name, "trial": t.number,
                 "state": t.state.name, "objective_value": t.value}
            r.update(t.params)
            rows.append(r)

keys = ["db", "study", "trial", "state", "objective_value"]
for r in rows:
    for k in r:
        if k not in keys:
            keys.append(k)

csv_path = os.path.join(OUT, "all_encdec_hpo_trials.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# 可读 summary:按 objective 排序(注意上游 objective = 验证 word/char accuracy,越大越好)
done = [r for r in rows if r["state"] == "COMPLETE" and r["objective_value"] is not None]
done.sort(key=lambda r: r["objective_value"], reverse=True)
with open(os.path.join(OUT, "all_encdec_hpo_trials_summary.txt"), "w", encoding="utf-8") as f:
    f.write(f"# Encoder-Decoder HPO — all trials across {len(set(r['db'] for r in rows))} study DBs\n")
    f.write(f"# total trials = {len(rows)}; completed = {len(done)}\n")
    f.write("# objective = upstream validation accuracy (higher = better); CER reported separately in eval_*.txt\n\n")
    f.write("rank  obj_value   trial  state     params\n")
    for i, r in enumerate(done, 1):
        p = {k: r[k] for k in r if k not in ("db", "study", "trial", "state", "objective_value")}
        f.write(f"{i:>3}  {r['objective_value']:.5f}  t{r['trial']:<4} {r['state']:<9} {p}\n")
print(f"wrote {csv_path}; total trials={len(rows)}, completed={len(done)}")
if done:
    print(f"best objective={done[0]['objective_value']:.5f} params={ {k:done[0][k] for k in done[0] if k not in ('db','study','trial','state','objective_value')} }")
