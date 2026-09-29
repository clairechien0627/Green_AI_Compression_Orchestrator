#!/usr/bin/env python3
"""
產生 dashboard/pages_custom/comparison.py 第 5 節（Token 成本比較）需要的
兩個根目錄 JSON 檔案：token_usage_estimated_final_results_30.json（離線估計）、
token_usage_real_final_results_30_noretry.json（真實記錄）。

兩份數字都不是重新計算出來的新資料，只是把「原本就已經存在、但分散在別處」的數字
重新包成 comparison.py 要的格式：

  - estimated（final_results_30，5-retry）：讀
    results/runs/30trial/llm_token_usage_reconstructed.json，套用跟
    report_30trial.py §7.2 完全相同的公式（avg_exact + avg_hidden * per_call）算出
    「估計總 token」，確保這裡跟該頁單獨顯示的數字一致。
  - real（final_results_30_noretry）：直接取自 report_30trial_noretry.py
    第 3476-3481 行寫死的常數（那批資料有真實 llm_usage_log.jsonl，這幾個數字
    當時已經算過一次，這裡不重算，只是換個格式存成檔案）。

用法：
  python3 analysis/build_token_usage_comparison_data.py
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODE_ORDER = ["full", "summary", "window", "tool"]


def build_estimated() -> list:
    raw = json.load(open(ROOT / "results" / "runs" / "30trial" / "llm_token_usage_reconstructed.json", encoding="utf-8"))

    by_mode = defaultdict(list)
    for row in raw:
        row["exact_total_calls"] = row["exact_calls"] + row["summary_extra_calls"]
        row["exact_total_tok"] = (row["exact_in"] + row["exact_out"]
                                   + row["summary_extra_in"] + row["summary_extra_out"])
        row["hidden_events"] = row["hidden_dedup_events"] + row["hidden_jsonretry_events"]
        by_mode[row["mode"]].append(row)

    estimated = []
    for mode in MODE_ORDER:
        rows = by_mode.get(mode, [])
        if not rows:
            continue
        n = len(rows)
        avg_calls = sum(r["exact_total_calls"] for r in rows) / n
        avg_exact = sum(r["exact_total_tok"] for r in rows) / n
        avg_hidden = sum(r["hidden_events"] for r in rows) / n
        per_call = avg_exact / avg_calls if avg_calls else 0
        approx_total = avg_exact + avg_hidden * per_call
        estimated.append({
            "mode": mode,
            "config": "final_results_30",
            "avg_calls": round(avg_calls, 1),
            "avg_exact_tokens": round(avg_exact, 0),
            "avg_total_tokens": round(approx_total, 0),
            "avg_hidden_events": round(avg_hidden, 1),
            "data_type": "estimated",
        })
    return estimated


def build_real() -> list:
    # 來源：report_30trial_noretry.py 的 _token_rows 常數（真實 llm_usage_log.jsonl 算出來的數字）
    real_source = [
        {"mode": "full",    "avg_calls": 30.0, "avg_total_tokens": 110268},
        {"mode": "window",  "avg_calls": 30.0, "avg_total_tokens": 71051},
        {"mode": "summary", "avg_calls": 35.0, "avg_total_tokens": 77074},
        {"mode": "tool",    "avg_calls": 53.0, "avg_total_tokens": 160194},
    ]
    return [
        {**r, "config": "final_results_30_noretry", "avg_hidden_events": 0.0, "data_type": "real"}
        for r in real_source
    ]


def main():
    estimated = build_estimated()
    real = build_real()

    est_path = ROOT / "token_usage_estimated_final_results_30.json"
    real_path = ROOT / "token_usage_real_final_results_30_noretry.json"

    with open(est_path, "w", encoding="utf-8") as f:
        json.dump(estimated, f, ensure_ascii=False, indent=2)
    with open(real_path, "w", encoding="utf-8") as f:
        json.dump(real, f, ensure_ascii=False, indent=2)

    print(f"寫入 {est_path}（{len(estimated)} 筆）")
    print(f"寫入 {real_path}（{len(real)} 筆）")


if __name__ == "__main__":
    main()
