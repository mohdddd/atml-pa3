"""Task 2 summary tables and figures (python -m pa3.task2.aggregate).

Outputs: results/task2/summary/*.{md,csv,json} and figures/task2/*.png
  learning_curves_<variant>.png   training loss, dev caption loss, answer NLL (correct / complementary / no image)
                                  per suite, image-swap penalty dNLL, dev accuracy; stage boundaries marked
  final_accuracy.png              final-test accuracy per suite and image condition (95% pair-bootstrap CIs)
  qualitative_<suite>.png         aligned complementary pairs (selection rule in qualitative.md)
  label_audit_grounding_test.png  annotation-derived final-test pairs with their labels, no model output
                                  (made during preparation, before any training)
"""
from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import numpy as np

from pa3.common.paths import FIGURES, results_dir
from pa3.common.runlog import read_json, read_jsonl, write_json
from pa3.task2 import config as C

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COND_COLOR = {"correct": SLOTS[0], "complementary": SLOTS[1], "mismatched": SLOTS[3], "no_image": SLOTS[2]}
COND_LABEL = {"correct": "correct image", "complementary": "complementary swap", "mismatched": "mismatched image",
              "no_image": "no image"}
SUITE_LABEL = {"grounding": "Annotation-derived grounding", "vqa": "Natural VQAv2"}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "lines.linewidth": 2.0, "figure.dpi": 150, "savefig.bbox": "tight"})
    return plt


def fig_dir() -> Path:
    d = FIGURES / C.results_task()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fmt(m: dict, pct: bool = True) -> str:
    k = 100 if pct else 1
    return f"{k * m['mean']:.1f} [{k * m['ci95'][0]:.1f}, {k * m['ci95'][1]:.1f}]" if pct else \
        f"{m['mean']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]"


def _write_table(path: Path, header: list[str], rows: list[list]) -> str:
    with open(path.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    md = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    md += "".join("| " + " | ".join(str(x) for x in r) + " |\n" for r in rows)
    return md


# ----------------------------------------------------------------------------- learning curves
def learning_curves(variant: str) -> None:
    out = results_dir(C.results_task(), variant)
    if not (out / "train_log.jsonl").exists():
        return
    plt = _style()
    cfg = C.config(variant)
    bounds = C.stage_bounds(cfg)
    tr = read_jsonl(out / "train_log.jsonl")
    dev = read_jsonl(out / "dev_log.jsonl") if (out / "dev_log.jsonl").exists() else []
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.6))
    ax = ax.ravel()
    u = np.array([r["update"] for r in tr])
    w = 25 if len(tr) > 200 else 1

    def smooth(v):
        v = np.array(v, dtype=float)
        if w == 1:
            return v
        out_ = np.full_like(v, np.nan)
        for i in range(len(v)):
            s = v[max(0, i - w + 1):i + 1]
            s = s[~np.isnan(s)]
            out_[i] = s.mean() if len(s) else np.nan
        return out_

    ax[0].plot(u, smooth([r["loss"] for r in tr]), color=INK, lw=1.6, label="all examples")
    for c, col in zip(["caption", "presence", "counting", "vqa"], SLOTS):
        v = [r.get(f"loss_{c}", np.nan) for r in tr]
        if np.isfinite(np.array(v, dtype=float)).any():
            ax[0].plot(u, smooth(v), color=col, lw=1.2, label=c)
    ax[0].set(title=f"Training loss ({w}-update mean)", xlabel="update", ylabel="per-example mean CE")
    ax[0].legend(fontsize=7)
    if dev:
        du = [r["update"] for r in dev]
        ax[1].plot(du, [r["caption_loss"] for r in dev], "-o", ms=4, color=SLOTS[0], label="correct image")
        ax[1].plot(du, [r["caption_loss_noimage"] for r in dev], "-o", ms=4, color=SLOTS[2], label="no image")
        ax[1].set(title="Dev caption reference loss", xlabel="update", ylabel="CE (incl. end-of-turn)")
        ax[1].legend(fontsize=7)
        for i, suite in enumerate(C.SUITES):
            a = ax[2 + i]
            for cond, key in (("correct", "nll_correct"), ("complementary", "nll_complementary"),
                              ("no_image", "nll_no_image")):
                a.plot(du, [r[f"{suite}_{key}"] for r in dev], "-o", ms=4, color=COND_COLOR[cond],
                       label=COND_LABEL[cond])
            a.set(title=f"Dev answer NLL - {SUITE_LABEL[suite]}", xlabel="update", ylabel="mean token NLL")
            a.legend(fontsize=7)
        for suite, col in zip(C.SUITES, SLOTS[:2]):
            ax[4].plot(du, [r[f"{suite}_delta_nll"] for r in dev], "-o", ms=4, color=col, label=SUITE_LABEL[suite])
            ax[5].plot(du, [100 * r[f"{suite}_acc_correct"] for r in dev], "-o", ms=4, color=col,
                       label=SUITE_LABEL[suite])
        ax[4].axhline(0, color=INK2, lw=0.8)
        ax[4].set(title="Image-swap penalty dNLL = NLL(partner) - NLL(correct)", xlabel="update", ylabel="nats")
        ax[5].set(title="Dev accuracy, correct image (greedy)", xlabel="update", ylabel="%")
        ax[4].legend(fontsize=7)
        ax[5].legend(fontsize=7)
    for a in ax:
        for b, name in zip(bounds[:-1], ["end stage 1", "end stage 2"]):
            a.axvline(b, color=INK2, lw=0.8, ls="--")
    ax[0].text(bounds[0], ax[0].get_ylim()[1], " S1 | S2", va="top", fontsize=7, color=INK2)
    ax[0].text(bounds[1], ax[0].get_ylim()[1], " S2 | S3", va="top", fontsize=7, color=INK2)
    fig.suptitle(f"Task 2 learning curves ({variant}: {cfg['n_visual_tokens']} visual tokens); dashed = stage boundaries",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir() / f"learning_curves_{variant}.png")
    plt.close(fig)


def dev_table(variant: str) -> str:
    out = results_dir(C.results_task(), variant)
    if not (out / "dev_log.jsonl").exists():
        return ""
    dev = {r["update"]: r for r in read_jsonl(out / "dev_log.jsonl")}
    cfg = C.config(variant)
    pts = [p for p in [0, *C.stage_bounds(cfg)] if p in dev]
    rows = []
    for p in pts:
        r = dev[p]
        rows.append([p, f"{r['caption_loss']:.3f}", f"{r['caption_loss_noimage']:.3f}"]
                    + [f"{r[f'{s}_{k}']:.3f}" for s in C.SUITES for k in ("nll_correct", "nll_complementary",
                                                                             "nll_no_image", "delta_nll")]
                    + [f"{100 * r[f'{s}_acc_correct']:.1f}" for s in C.SUITES])
    hdr = ["update", "cap loss", "cap loss (no img)"] + [f"{s[:1].upper()} {k}" for s in C.SUITES for k in
                                                         ("NLL corr", "NLL comp", "NLL no-img", "dNLL")] + \
          ["G dev acc %", "V dev acc %"]
    return _write_table(summary_dir() / f"dev_stage_endpoints_{variant}", hdr, rows)


# ----------------------------------------------------------------------------- final results
def summary_dir() -> Path:
    return results_dir(C.results_task(), "summary")


def final_tables(variants: list[str]) -> str:
    md = ""
    for v in variants:
        p = results_dir(C.results_task(), v, "eval") / "metrics.json"
        if not p.exists():
            continue
        m = read_json(p)
        for suite in C.SUITES:
            s = m[suite]
            types = [t for t in s if t not in ("all", "swap_outcomes")]
            hdr = (["condition", "accuracy % [95% CI]"] + [f"{t} acc %" for t in types]
                   + ["pair acc % [CI]", "identical %", "empty %", "ref NLL [CI]", "ms / answer"])
            rows = []
            for c in ("correct", "complementary", "mismatched", "no_image"):
                a = s["all"][c]
                rows.append([COND_LABEL[c], _fmt(a["accuracy"])]
                            + [f"{100 * s[t][c]['accuracy']['mean']:.1f}" for t in types]
                            + [_fmt(a["pair_accuracy"]), f"{100 * a['identical_answer_rate']['mean']:.1f}",
                               f"{100 * a['empty_answer_rate']['mean']:.1f}", _fmt(a["mean_reference_nll"], pct=False),
                               f"{1000 * a['generation_seconds_per_answer']['mean']:.1f}"])
            md += f"\n### {v} - {SUITE_LABEL[suite]} ({s['all']['n_pairs']} pairs / {s['all']['n_questions']} questions)\n\n"
            md += _write_table(summary_dir() / f"final_{v}_{suite}", hdr, rows)
            # correct-image advantages
            hdr = ["comparison", "subset", "accuracy adv. (pp) [95% CI]", "pair-acc adv. (pp) [CI]",
                   "NLL adv. (nats) [CI]"]
            rows = []
            for key in ("correct_minus_complementary", "correct_minus_mismatched", "correct_minus_no_image"):
                for t in ["all"] + types:
                    d = s[t][key]
                    rows.append([key.replace("_minus_", " vs "), t, _fmt(d["accuracy_advantage"]),
                                 _fmt(d["pair_accuracy_advantage"]), _fmt(d["nll_advantage(b_minus_a)"], pct=False)])
            md += f"\nCorrect-image advantages ({v}, {suite}; NLL advantage = NLL(other) - NLL(correct)):\n\n"
            md += _write_table(summary_dir() / f"advantages_{v}_{suite}", hdr, rows)
            so = s["swap_outcomes"]
            md += (f"\nComplementary swap outcomes ({v}, {suite}): follows partner answer "
                   f"{100 * so['follows_partner']:.1f}%, keeps original answer {100 * so['keeps_original_answer']:.1f}%, "
                   f"other answer {100 * so['other_answer']:.1f}%; unchanged from the correct-image prediction "
                   f"{100 * so['unchanged_from_correct_image_prediction']:.1f}%; equal to the partner's correct-image "
                   f"prediction (same input, sanity check) {100 * so['equals_partner_correct_image_prediction']:.1f}%.\n")
    return md


def final_figure(variants: list[str]) -> None:
    ms = {v: read_json(results_dir(C.results_task(), v, "eval") / "metrics.json") for v in variants
          if (results_dir(C.results_task(), v, "eval") / "metrics.json").exists()}
    if not ms:
        return
    plt = _style()
    conds = ["correct", "complementary", "mismatched", "no_image"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    width = 0.8 / len(ms)
    for ax, suite in zip(axes, C.SUITES):
        for k, (v, m) in enumerate(ms.items()):
            vals = [100 * m[suite]["all"][c]["accuracy"]["mean"] for c in conds]
            lo = [100 * m[suite]["all"][c]["accuracy"]["ci95"][0] for c in conds]
            hi = [100 * m[suite]["all"][c]["accuracy"]["ci95"][1] for c in conds]
            x = np.arange(len(conds)) + (k - (len(ms) - 1) / 2) * width
            ax.bar(x, vals, width * 0.92, color=SLOTS[k], label=f"{v} ({'196' if v == 't196' else '49'} visual tokens)")
            ax.errorbar(x, vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none", ecolor=INK, lw=1, capsize=2)
            for xi, val in zip(x, vals):
                ax.text(xi, val + 1, f"{val:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(np.arange(len(conds)), [COND_LABEL[c] for c in conds])
        ax.set(title=SUITE_LABEL[suite], ylabel="accuracy % (95% pair-bootstrap CI)", ylim=(0, 105))
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir() / "final_accuracy.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- resources / compression
def resources_table(variants: list[str]) -> str:
    rows = []
    fc = results_dir(C.results_task()) / "feature_cache.json"
    if fc.exists():
        f = read_json(fc)
        rows.append(["SigLIP feature caching (once, shared)", "-", "frozen encoder", f"{f['images']} images",
                     f"{f['seconds_total'] / 60:.1f} min", f"{1000 / f['images_per_second']:.1f} ms/img",
                     f"{f['peak_gpu_alloc_gb']:.2f}", f"cache {f['cache_size_gb']:.2f} GB"])
    for v in variants:
        p = results_dir(C.results_task(), v) / "resources.json"
        if not p.exists():
            continue
        r = read_json(p)
        cfg = C.config(v)
        for st in cfg["stages"]:
            n = st["name"]
            tp = r["trainable_params_by_stage"].get(n, {})
            pk = r["peak_gpu_by_stage"].get(n, {})
            sec = r["train_seconds_by_stage"].get(n)
            rows.append([v, n, f"{tp.get('total', 0):,} (conn {tp.get('connector', 0):,} + LoRA {tp.get('lora', 0):,})",
                         f"{st['updates']} upd", f"{sec / 3600:.2f} h" if sec else "-",
                         f"{sec / st['updates']:.2f} s/upd" if sec else "-",
                         f"{pk.get('alloc_gb', 0):.2f}", ""])
        e = results_dir(C.results_task(), v, "eval") / "metrics.json"
        if e.exists():
            m = read_json(e)
            rows.append([v, "final-test generation", "-", f"{m['n_generated_answers']} answers",
                         f"{m['eval_seconds'] / 60:.1f} min (incl. NLL)",
                         f"{1000 * m['generation_seconds_total'] / m['n_generated_answers']:.1f} ms/answer",
                         f"{m['peak_gpu_alloc_gb']:.2f}", "batch 8, greedy, 16 tokens"])
        rows.append([v, "dev evaluations", "-", "-", f"{r['dev_eval_seconds'] / 60:.1f} min", "-", "-", ""])
    hdr = ["run", "stage", "trainable params", "work", "time", "rate", "peak GPU alloc (GB)", "note"]
    return _write_table(summary_dir() / "resources", hdr, rows)


def compression_table() -> str:
    ps = {v: results_dir(C.results_task(), v, "eval") / "metrics.json" for v in C.VARIANTS}
    if not all(p.exists() for p in ps.values()):
        return ""
    m = {v: read_json(p) for v, p in ps.items()}
    r = {v: read_json(results_dir(C.results_task(), v) / "resources.json") for v in C.VARIANTS}
    rows = []
    for suite in C.SUITES:
        for c in ("correct", "complementary", "no_image"):
            rows.append([f"{suite} accuracy % ({COND_LABEL[c]})"] + [_fmt(m[v][suite]["all"][c]["accuracy"]) for v in C.VARIANTS])
        rows.append([f"{suite} pair accuracy % (correct)"] + [_fmt(m[v][suite]["all"]["correct"]["pair_accuracy"]) for v in C.VARIANTS])
        rows.append([f"{suite} accuracy adv. vs complementary (pp)"] +
                    [_fmt(m[v][suite]["all"]["correct_minus_complementary"]["accuracy_advantage"]) for v in C.VARIANTS])
        rows.append([f"{suite} NLL adv. vs complementary (nats)"] +
                    [_fmt(m[v][suite]["all"]["correct_minus_complementary"]["nll_advantage(b_minus_a)"], pct=False)
                     for v in C.VARIANTS])
    rows.append(["training time (h)"] + [f"{r[v]['train_hours_total']:.2f}" for v in C.VARIANTS])
    rows.append(["peak GPU alloc, stage 2 (GB)"] + [f"{r[v]['peak_gpu_by_stage'].get('instruct', {}).get('alloc_gb', 0):.2f}"
                                                   for v in C.VARIANTS])
    rows.append(["generation ms / answer"] + [f"{1000 * m[v]['generation_seconds_total'] / m[v]['n_generated_answers']:.1f}"
                                              for v in C.VARIANTS])
    return _write_table(summary_dir() / "compression_196_vs_49", ["metric", "196 tokens (t196)", "49 tokens (t49)"], rows)


# ----------------------------------------------------------------------------- qualitative
QUAL_RULE = ("For each suite, final-test pairs are sorted by pair ID and classified from the correct-image "
             "predictions of the main (196-token) model: (A) grounded - both questions exactly correct, so each "
             "complementary-swap prediction equals the partner's answer; (B) image-insensitive - identical "
             "predictions for both images; (C) image-sensitive but wrong - different predictions, at least one "
             "wrong. The first two pairs of each class are shown (fewer if a class is smaller).")


def _short(s: str, n: int = 36) -> str:
    """One-line display form of a raw prediction (figures only; scoring uses the raw text)."""
    s = s.replace("\n", " \u23ce ")
    return s if len(s) <= n else s[:n - 3] + "..."


def select_qualitative(recs: list[dict], suite: str) -> list[tuple[str, str]]:
    corr = {}
    for r in recs:
        if r["suite"] == suite and r["condition"] == "correct":
            corr.setdefault(r["pair_id"], []).append(r)
    cls = {"A": [], "B": [], "C": []}
    for pid in sorted(corr):
        a, b = sorted(corr[pid], key=lambda r: r["pos"])
        if a["exact"] and b["exact"]:
            cls["A"].append(pid)
        elif a["pred_norm"] == b["pred_norm"]:
            cls["B"].append(pid)
        else:
            cls["C"].append(pid)
    return [(k, p) for k in "ABC" for p in cls[k][:2]]


def qualitative(variant: str = "t196") -> str:
    p = results_dir(C.results_task(), variant, "eval") / "test_examples.jsonl.gz"
    if not p.exists():
        return ""
    recs = read_jsonl(p)
    from pa3.task2.data import image_path

    md = f"\n{QUAL_RULE}\n"
    names = {"A": "grounded", "B": "image-insensitive", "C": "image-sensitive, wrong"}
    for suite in C.SUITES:
        sel = select_qualitative(recs, suite)
        by = {}
        for r in recs:
            if r["suite"] == suite:
                by.setdefault((r["pair_id"], r["pos"]), {})[r["condition"]] = r
        md += f"\n#### {SUITE_LABEL[suite]}\n\n| class | question | image | reference (canonical) | correct | swap | mismatched | no image |\n|---|---|---|---|---|---|---|---|\n"
        for k, pid in sel:
            for pos in (0, 1):
                d = by[(pid, pos)]
                c = d["correct"]
                md += (f"| {names[k]} | {c['question']} | {Path(c['image']).name} | {c['reference']} ({c['canonical']}) | "
                       + " | ".join(repr(_short(d[x]["pred"], 60)) for x in ("correct", "complementary", "mismatched", "no_image"))
                       + " |\n")
        if not all(image_path(by[(pid, 0)]["correct"]["image"]).exists() for _, pid in sel):
            continue
        plt = _style()
        from PIL import Image

        n = len(sel)
        fig, axes = plt.subplots(n, 3, figsize=(10, 2.6 * n), gridspec_kw={"width_ratios": [1, 1, 1.6]})
        axes = np.atleast_2d(axes)
        for row, (k, pid) in enumerate(sel):
            for pos in (0, 1):
                d = by[(pid, pos)]
                ax = axes[row, pos]
                im = Image.open(image_path(d["correct"]["image"])).convert("RGB")
                im.thumbnail((448, 448))
                ax.imshow(im)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                ax.set_title(f"image {'AB'[pos]}: ref '{d['correct']['reference']}'", fontsize=8)
            t = axes[row, 2]
            t.axis("off")
            a, b = by[(pid, 0)], by[(pid, 1)]
            q = textwrap.fill(a["correct"]["question"], 48)
            txt = (f"[{names[k]}] {pid}\nQ: {q}\n\n"
                   f"pred on A: '{_short(a['correct']['pred'])}'   pred on B: '{_short(b['correct']['pred'])}'\n"
                   f"A's question, mismatched img: '{_short(a['mismatched']['pred'])}'\n"
                   f"B's question, mismatched img: '{_short(b['mismatched']['pred'])}'\n"
                   f"no image: '{_short(a['no_image']['pred'])}'\n"
                   f"ref NLL A: correct {a['correct']['nll']:.2f} / swap {a['complementary']['nll']:.2f}\n"
                   f"ref NLL B: correct {b['correct']['nll']:.2f} / swap {b['complementary']['nll']:.2f}")
            t.text(0, 0.95, txt, va="top", fontsize=8, family="monospace", color=INK)
        fig.suptitle(f"{SUITE_LABEL[suite]}: complementary pairs, {variant} final model "
                     f"(swap prediction for A = prediction on B)", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir() / f"qualitative_{suite}.png")
        plt.close(fig)
    (summary_dir() / "qualitative.md").write_text(md)
    return md


def label_audit(n_pairs: int = 16) -> None:
    """Annotation-derived final-test pairs with their generated labels (no model output), for inspecting
    COCO label quality independently of predictions. First n/2 presence and n/2 counting pairs."""
    from pa3.task2.data import image_path, load_splits

    s = load_splits()
    pairs = [p for p in s["grounding"]["test"] if p["type"] == "presence"][:n_pairs // 2] + \
            [p for p in s["grounding"]["test"] if p["type"] == "counting"][:n_pairs // 2]
    if not all(image_path(it["image"]).exists() for p in pairs for it in p["items"]):
        return
    plt = _style()
    from PIL import Image

    cols = 4
    rows = int(np.ceil(len(pairs) * 2 / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.9 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    k = 0
    for p in pairs:
        for it in p["items"]:
            ax = axes.ravel()[k]
            im = Image.open(image_path(it["image"])).convert("RGB")
            im.thumbnail((400, 400))
            ax.imshow(im)
            ax.set_title(textwrap.fill(f"{p['pair_id'][-14:]}: {p['question']} -> {it['answer']}", 40), fontsize=7)
            k += 1
    fig.suptitle("Label audit: annotation-derived final-test pairs and their COCO-derived answers (no model output)",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir() / "label_audit_grounding_test.png")
    plt.close(fig)


def main():
    variants = [v for v in C.VARIANTS if (results_dir(C.results_task(), v) / "train_log.jsonl").exists()]
    md = "# Task 2 summary\n"
    for v in variants:
        learning_curves(v)
        t = dev_table(v)
        if t:
            md += f"\n## Development metrics at stage endpoints ({v})\n\n{t}"
    t = final_tables(variants)
    if t:
        md += f"\n## Final test\n{t}"
    final_figure(variants)
    md += f"\n## Resources\n\n{resources_table(variants)}"
    t = compression_table()
    if t:
        md += f"\n## Optional visual-token compression (196 vs 49)\n\n{t}"
    md += f"\n## Qualitative examples\n{qualitative('t196')}"
    cal = {v: read_json(results_dir(C.results_task(), v) / "calibration.json") for v in variants
           if (results_dir(C.results_task(), v) / "calibration.json").exists()}
    if cal:
        md += "\n## Connector calibration\n\n" + "".join(
            f"- {v}: {c['rule']}; text {c['text_embedding_mean_norm']:.4f}, visual before "
            f"{c['visual_token_mean_norm_before']:.4f} -> scale {c['scale']:.4f} (after {c['visual_token_mean_norm_after']:.4f})\n"
            for v, c in cal.items())
    (summary_dir() / "summary.md").write_text(md)
    print(md)
    try:
        from pa3.common.gitsync import push

        push([summary_dir(), fig_dir()], "task2: summary tables and figures")
    except Exception as e:
        print(f"[git] {e}")


if __name__ == "__main__":
    main()
