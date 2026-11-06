#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh đồ thị tri thức của bài báo, để dùng làm cơ sở hiệu chỉnh nội dung.

    python3 src/build_knowledge_graph.py

Đồ thị nối: khoảng trống nghiên cứu → dữ liệu → thí nghiệm → kết quả →
bảng/hình → phát biểu trong bài → hạn chế. Nhờ đó thấy được mỗi phát biểu dựa
trên bằng chứng nào, và ngược lại, dữ liệu nào chưa được dùng.

Phần sinh TỰ ĐỘNG (không gõ tay): tiết mục và tiểu mục của bài, bảng, hình,
mọi tham chiếu chéo, mọi con số trong results/, script nào sinh ra file nào.
Phần KHAI BÁO (cần phán đoán chuyên môn, ghi rõ trong GAPS/EXPERIMENTS dưới đây):
ánh xạ khoảng trống → thí nghiệm.

Đầu ra: KNOWLEDGE_GRAPH.md
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import MODEL_LABELS, RAE, RESULTS, load_cfg, log, scope

MS = RAE / "manuscript" / "RAE_manuscript.md"

# --- Khai báo: khoảng trống nghiên cứu, rút từ Introduction ---
GAPS = [
    ("G1", "Các nghiên cứu trước báo cáo độ chính xác ~99% nhưng tính trên ma trận đã "
           "bỏ background, nên không cho biết bỏ sót bao nhiêu",
     ["E-eval", "E-seed"], "Introduction ¶5"),
    ("G2", "Chưa rõ ba họ kiến trúc phát hiện hành xử thế nào trên ảnh cận cảnh ruộng "
           "mía, khác với ảnh UAV/aerial của phần lớn tài liệu",
     ["E-train", "E-eval", "E-boot"], "Introduction ¶5"),
    ("G3", "Chưa có đánh giá khả năng triển khai trên thiết bị biên gắn với yêu cầu "
           "nông học cụ thể", ["E-speed", "E-complex"], "Introduction ¶5"),
    ("G4", "Kết quả một lần chạy báo cáo tới ba chữ số thập phân không phân biệt được "
           "khác biệt thật với nhiễu giữa các lần huấn luyện",
     ["E-seed", "E-boot"], "Results, Statistics"),
]

# --- Khai báo: thí nghiệm, mỗi cái gắn với script và file kết quả ---
EXPERIMENTS = [
    ("E-train", "Huấn luyện 3 mô hình quy ước hộp, 3 seed mỗi mô hình (9 lượt)",
     "src/train_detectron2.py, src/train_ultralytics.py, run_seeds.sh",
     "runs/<model>[_seed<N>]/"),
    ("E-eval", "Đánh giá trên tập test: ma trận nhầm lẫn hai biến thể + AP theo COCO",
     "src/evaluate.py", "results/<model>[_seed<N>].json"),
    ("E-boot", "Bootstrap ghép cặp 1.000 lần trên ảnh test",
     "src/bootstrap.py", "results/bootstrap.json"),
    ("E-seed", "Tổng hợp phương sai do huấn luyện qua các seed",
     "src/seed_variance.py", "results/seed_variance.json"),
    ("E-complex", "Đo tham số, GFLOPs, VRAM đỉnh",
     "src/complexity.py", "results/complexity.json"),
    ("E-speed", "Đo thông lượng, 10 lượt lặp lấy trung bình",
     "src/edge_benchmark.py", "results/edge_<model>.json"),
    ("E-data", "Thống kê bộ dữ liệu và chuyển đổi nhãn",
     "src/data_prep.py", "results/dataset_stats.json"),
]


def read_ms() -> str:
    if not MS.exists():
        raise SystemExit(f"Chưa có bản thảo: {MS}")
    return MS.read_text(encoding="utf-8")


def parse_sections(t: str) -> list[dict]:
    """Tiết mục và tiểu mục, kèm tham chiếu chéo mà mỗi tiểu mục dùng."""
    out, cur_sec = [], "(đầu bài)"
    for block in t.split("\n\n"):
        b = block.strip()
        if b.startswith("## "):
            cur_sec = b[3:].strip()
            continue
        m = re.match(r"\*\*([A-Z][^.]{3,70})\.\*\*", b)
        if not m:
            continue
        body = b
        out.append({
            "section": cur_sec,
            "sub": m.group(1),
            "tables": sorted({int(x) for x in re.findall(r"Table (\d)", body)}),
            "figures": sorted({int(x) for x in re.findall(r"Figure (\d)", body)}),
            "words": len(body.split()),
            "numbers": sorted({x for x in re.findall(r"\b\d+\.\d{2,3}\b", body)}),
        })
    return out


def parse_artifacts(t: str) -> tuple[list, list]:
    tables = [(int(m.group(1)), m.group(2).strip())
              for m in re.finditer(r"(?m)^\*\*Table (\d)\. ([^*]+)\*\*", t)]
    figs = [(int(m.group(1)), m.group(2).strip())
            for m in re.finditer(r"(?m)^Figure (\d)\. ([^\n]+)", t)]
    return sorted(tables), sorted(figs)


def key_results(cfg: dict) -> list[dict]:
    """Kết quả then chốt, đọc thẳng từ results/ — không gõ tay."""
    import discussion as D
    sv = json.loads((RESULTS / "seed_variance.json").read_text())
    D.use_seed_variance(sv)
    boot = json.loads((RESULTS / "bootstrap.json").read_text())
    comp = json.loads((RESULTS / "complexity.json").read_text())
    from common import clean_fps, throughput_requirement
    t = throughput_requirement(cfg)
    primary, _ = scope(cfg)

    R = []
    for k in primary:
        R.append({"id": f"R-ap50-{k}", "what": f"AP50 {MODEL_LABELS[k]}",
                  "value": f"{D.seed_mean(k,'ap50'):.2f} ± {D.seed_sd(k,'ap50'):.2f}",
                  "from": "E-seed", "shown": "Table 2"})
        R.append({"id": f"R-rec-{k}", "what": f"Recall cỏ (đủ) {MODEL_LABELS[k]}",
                  "value": f"{D.seed_mean(k,'weed_recall_full'):.3f}",
                  "from": "E-seed", "shown": "Table 2"})
        R.append({"id": f"R-f1-{k}", "what": f"F1 (bỏ nền) {MODEL_LABELS[k]}",
                  "value": f"{D.seed_mean(k,'f1'):.3f}", "from": "E-seed",
                  "shown": "Table 3"})
        R.append({"id": f"R-fps-{k}", "what": f"Thông lượng {MODEL_LABELS[k]}",
                  "value": f"{clean_fps(k)[0]:.1f} FPS", "from": "E-speed",
                  "shown": "Table 4"})
        R.append({"id": f"R-cost-{k}", "what": f"Chi phí tính toán {MODEL_LABELS[k]}",
                  "value": f"{comp[k]['params']/1e6:.1f} M, "
                           f"{comp[k]['gflops_median']:.0f} GFLOPs",
                  "from": "E-complex", "shown": "Table 4"})
    # HAI phép so khác nhau cho cùng một cặp mô hình, và chúng cho kết luận khác
    # nhau. Nếu chỉ liệt kê bootstrap thì người hiệu chỉnh sẽ viết "cả ba cặp có
    # ý nghĩa", ngược với kết luận của bài. Phải đưa cả hai, ghi rõ nguồn.
    from scipy import stats as _st
    import numpy as _np
    for pair, d in boot.get("pairwise", {}).items():
        a, b = pair.split("_vs_")
        if a in primary and b in primary and "ap50" in d:
            x = d["ap50"]
            R.append({"id": f"R-boot-{a}-{b}",
                      "what": f"Hiệu AP50 {MODEL_LABELS[a]} − {MODEL_LABELS[b]} "
                              f"(CHỈ phương sai tập test)",
                      "value": f"{x['diff']:+.2f} [{x['ci95'][0]:+.1f}, {x['ci95'][1]:+.1f}] "
                               + ("có ý nghĩa" if x["significant"] else "không"),
                      "from": "E-boot", "shown": "Statistics ¶1"})
    for i, a in enumerate(primary):
        for b in primary[i + 1:]:
            va = [e["ap50"] for e in sv[a]["runs"] if e.get("status") == "ok"]
            vb = [e["ap50"] for e in sv[b]["runs"] if e.get("status") == "ok"]
            _t, pv = _st.ttest_ind(va, vb, equal_var=False)
            R.append({"id": f"R-welch-{a}-{b}",
                      "what": f"Hiệu AP50 {MODEL_LABELS[a]} − {MODEL_LABELS[b]} "
                              f"(GỘP cả phương sai huấn luyện)",
                      "value": f"{_np.mean(va) - _np.mean(vb):+.2f}, P = {pv:.3f} "
                               + ("có ý nghĩa" if pv < 0.05 else "**không**"),
                      "from": "E-seed + E-boot", "shown": "Statistics ¶3"})
    R.append({"id": "R-threshold", "what": "Ngưỡng thông lượng",
              "value": f"{t['fps_required_on_device']:.1f} FPS thiết bị → "
                       f"{t['fps_required_on_workstation']:.0f} FPS máy trạm",
              "from": "cấu hình (đặt trước)", "shown": "Methods, Table 4"})
    return R


def orphan_check(secs, tables, figs, results) -> list[str]:
    """Kiểm chỗ hở: bảng/hình không được nhắc, kết quả không xuất hiện trong bài,
    tiểu mục không dựa trên bảng/hình nào."""
    t = read_ms()
    msgs = []
    ref_t = {int(x) for x in re.findall(r"Table (\d)", t)}
    ref_f = set()
    cap_spans = [m.span() for m in re.finditer(r"(?m)^Figure (\d)\. ", t)]
    for m in re.finditer(r"Figure (\d)", t):
        if not any(a <= m.start() < b for a, b in cap_spans):
            ref_f.add(int(m.group(1)))

    for n, cap in tables:
        if n not in ref_t:
            msgs.append(f"⚠ Bảng {n} có chú thích nhưng không đoạn nào nhắc tới")
    for n, cap in figs:
        if n not in ref_f:
            msgs.append(f"⚠ Hình {n} có chú thích nhưng không đoạn nào nhắc tới")

    body = t
    for r in results:
        v = re.split(r"[ ±\[]", r["value"])[0].rstrip("%,")
        if re.fullmatch(r"\d+\.\d+", v) and v not in body:
            msgs.append(f"⚠ {r['id']} ({r['what']} = {r['value']}) không xuất hiện trong bài")

    for s in secs:
        if s["section"] == "RESULTS AND DISCUSSION" and not s["tables"] and not s["figures"]:
            msgs.append(f"ℹ Tiểu mục '{s['sub']}' ở phần Kết quả không viện dẫn bảng/hình nào")
    return msgs or ["✓ không thấy chỗ hở nào"]


def mermaid(secs, tables, figs) -> str:
    L = ["```mermaid", "graph LR"]
    for gid, txt, exps, _src in GAPS:
        L.append(f'  {gid}["{gid}: {txt[:46]}…"]:::gap')
        for e in exps:
            L.append(f"  {gid} --> {e.replace('-', '_')}")
    for eid, txt, _s, _o in EXPERIMENTS:
        L.append(f'  {eid.replace("-", "_")}["{eid}<br/>{txt[:40]}…"]:::exp')
    L.append('  E_data --> E_train --> E_eval')
    L.append('  E_eval --> E_boot')
    L.append('  E_eval --> E_seed')
    for n, _c in tables:
        L.append(f'  T{n}["Table {n}"]:::art')
    for n, _c in figs:
        L.append(f'  F{n}["Figure {n}"]:::art')
    L += ["  E_seed --> T2", "  E_seed --> T3", "  E_boot --> T2",
          "  E_complex --> T4", "  E_speed --> T4", "  E_eval --> T5",
          "  E_data --> F1", "  E_train --> F2", "  E_eval --> F3", "  E_eval --> F4"]
    L += ["  classDef gap fill:#fde7e7,stroke:#c62828",
          "  classDef exp fill:#e8f1fb,stroke:#1565c0",
          "  classDef art fill:#eef7ee,stroke:#2e7d32", "```"]
    return "\n".join(L)


def build() -> str:
    cfg = load_cfg()
    t = read_ms()
    secs = parse_sections(t)
    tables, figs = parse_artifacts(t)
    results = key_results(cfg)
    stats = json.loads((RESULTS / "dataset_stats.json").read_text())

    L = []
    A = L.append
    A("# Đồ thị tri thức — bản thảo RAE")
    A("")
    A("Sinh tự động từ `manuscript/RAE_manuscript.md` và `results/*.json` bằng "
      "`src/build_knowledge_graph.py`. Chạy lại sau mỗi lần sửa bài để đồ thị không lệch "
      "khỏi nội dung.")
    A("")
    A("Dùng để làm gì: xem mỗi phát biểu trong bài dựa trên bằng chứng nào, dữ liệu nào "
      "chưa được dùng, và chỗ nào trong lập luận còn hở.")
    A("")
    A("## Tổng quan")
    A("")
    A(mermaid(secs, tables, figs))
    A("")

    A("## 1. Khoảng trống nghiên cứu → thí nghiệm")
    A("")
    A("| Mã | Khoảng trống | Nêu ở | Thí nghiệm trả lời |")
    A("|---|---|---|---|")
    for gid, txt, exps, src in GAPS:
        A(f"| **{gid}** | {txt} | {src} | {', '.join(exps)} |")
    A("")

    A("## 2. Thí nghiệm → mã nguồn → đầu ra")
    A("")
    A("| Mã | Thí nghiệm | Script | Đầu ra |")
    A("|---|---|---|---|")
    for eid, txt, script, out in EXPERIMENTS:
        A(f"| **{eid}** | {txt} | `{script}` | `{out}` |")
    A("")

    A("## 3. Dữ liệu")
    A("")
    bb, pg, cb = stats["bbox"], stats["polygon"], stats["_combined"]
    A("| Bộ | Quy ước | Train | Valid | Test | Ảnh gốc | Đối tượng | Đối tượng/ảnh | Cỏ trung vị |")
    A("|---|---|---|---|---|---|---|---|---|")
    for name, d, conv in (("Dataset_Bounding_Box", bb, "hộp"),
                          ("Dataset_Polygon", pg, "đa giác")):
        A(f"| `{name}` | {conv} | {d['train']['images']:,} | {d['valid']['images']:,} | "
          f"{d['test']['images']:,} | {d['_total']['source_images']:,} | "
          f"{d['_total']['instances_total']:,} | {d['test']['instances_per_image']:.1f} | "
          f"{d['test']['weed_area_pct_median']:.2f}% |")
    A("")
    A(f"Hợp hai bộ: **{cb['source_images_union']:,} ảnh gốc**, "
      f"{cb['source_images_shared']:,} ảnh có nhãn ở cả hai định dạng. "
      f"**Phạm vi bài chỉ dùng bộ nhãn hộp** — xem G2 và mục *Scope of the comparison*.")
    A("")

    A("## 4. Kết quả then chốt")
    A("")
    A("Mọi giá trị đọc thẳng từ `results/`, không gõ tay.")
    A("")
    A("> **Đọc kỹ hai nhóm `R-boot-*` và `R-welch-*`.** Cùng một cặp mô hình nhưng hai "
      "phép so cho kết luận khác nhau: bootstrap chỉ tính phương sai do tập test hữu "
      "hạn nên nói cả ba cặp có ý nghĩa; Welch trên giá trị từng seed tính thêm phương "
      "sai do huấn luyện nên chỉ còn một cặp. **Kết luận của bài dựa trên nhóm thứ "
      "hai.** Khi hiệu chỉnh, đừng trích nhóm đầu như thể đó là kết luận cuối.")
    A("")
    A("| Mã | Đại lượng | Giá trị | Từ thí nghiệm | Trình bày ở |")
    A("|---|---|---|---|---|")
    for r in results:
        A(f"| `{r['id']}` | {r['what']} | {r['value']} | {r['from']} | {r['shown']} |")
    A("")

    A("## 5. Bảng và hình")
    A("")
    A("| Mã | Chú thích (rút gọn) | Được nhắc ở tiểu mục |")
    A("|---|---|---|")
    for n, cap in tables:
        users = [s["sub"] for s in secs if n in s["tables"]]
        A(f"| **Table {n}** | {cap[:95]}… | {', '.join(users) or '—'} |")
    for n, cap in figs:
        users = [s["sub"] for s in secs if n in s["figures"]]
        A(f"| **Figure {n}** | {cap[:95]}… | {', '.join(users) or '—'} |")
    A("")

    A("## 6. Cấu trúc bài và bằng chứng từng tiểu mục")
    A("")
    A("| Tiết mục | Tiểu mục | Từ | Bảng | Hình | Số liệu nêu |")
    A("|---|---|---|---|---|---|")
    for s in secs:
        A(f"| {s['section']} | **{s['sub']}** | {s['words']} | "
          f"{', '.join(map(str, s['tables'])) or '—'} | "
          f"{', '.join(map(str, s['figures'])) or '—'} | {len(s['numbers'])} |")
    A("")

    A("## 7. Kiểm chỗ hở")
    A("")
    A("Dùng mục này khi hiệu chỉnh: mỗi dòng ⚠ là một chỗ cần xử lý trước khi nộp.")
    A("")
    for m in orphan_check(secs, tables, figs, results):
        A(f"- {m}")
    A("")

    A("## 8. Mạch lập luận chính")
    A("")
    A("Ba nhánh, mỗi nhánh đi từ khoảng trống tới kết luận:")
    A("")
    A("**Nhánh A — chỉ số quen dùng che mất điều quan trọng** (G1)")
    A("")
    A("> E-eval tính ma trận nhầm lẫn ở *hai* biến thể → R-f1-* cao (≈0,97) nhưng "
      "R-rec-* thấp (0,40–0,50) → Bảng 2 và Bảng 3 đặt cạnh nhau → tiểu mục "
      "*Detection completeness* → kết luận: 50–60% cỏ không được phát hiện.")
    A("")
    A("**Nhánh B — xếp hạng mô hình không vững như tưởng** (G2, G4)")
    A("")
    A("> E-train 9 lượt → E-seed đo phương sai huấn luyện → E-boot đo phương sai tập "
      "test → R-diff-* → tiểu mục *Statistics* → kết luận: chỉ 1/3 cặp có ý nghĩa.")
    A("")
    A("**Nhánh C — nhanh và chính xác kéo ngược nhau** (G3)")
    A("")
    A("> R-threshold (đặt trước từ nông học) + E-speed + E-complex → Bảng 4 → tiểu mục "
      "*Inference cost* → kết luận: chỉ YOLOv10 qua sàng lọc, nhưng nó có recall thấp "
      "nhất → đánh đổi, không phong mô hình tốt nhất.")
    A("")
    A("## 9. Chỗ đồ thị này KHÔNG kiểm được")
    A("")
    A("- **Ánh xạ khoảng trống → thí nghiệm** ở mục 1 là do tôi khai báo, không sinh tự "
      "động. Nếu sửa Introduction thì phải sửa `GAPS` trong script.")
    A("- **Chất lượng lập luận**: đồ thị cho biết một phát biểu *có* bằng chứng, không "
      "cho biết bằng chứng đó *đủ mạnh* hay không.")
    A("- **Tính đúng của con số**: việc đó do `src/check_consistency.py` lo.")
    return "\n".join(L)


def main():
    out = RAE / "KNOWLEDGE_GRAPH.md"
    out.write_text(build(), encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    main()
