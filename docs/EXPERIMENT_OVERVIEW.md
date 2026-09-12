# Green AI Compression Orchestrator — 實驗方法、目標與成果總覽

> 本文彙整目前 `dashboard/` 互動報表與 `final_results*` / `results*` 各批次結果所呈現的實驗全貌，作為對外說明或口頭報告的書面依據。細節數字皆可在 `streamlit run dashboard/app.py` 中即時互動查驗。

---

## 1. 研究目標

在「LLM 壓縮」（量化 / 低秩分解 / 稀疏化）任務中，壓縮設定空間龐大且各參數對 **準確度、延遲、VRAM、碳排放** 四個目標的影響互相牽制，傳統上需要人工試錯或跑統計優化演算法（TPE、遺傳演算法……）才能找到 Pareto 最優解。

本專案要驗證的核心假設是：

> **具備領域知識的 LLM，能不能作為「決策者」，用比傳統黑盒優化演算法更少的試驗次數、更低的失敗率，找到同樣好（甚至更好）的壓縮設定？**

延伸出三個具體研究問題：

1. **LLM-guided 搜尋 vs 統計搜尋**：在相同 trial 預算下，LLM 決策（`Global_Tuner`）與 Optuna TPE / NSGA-II / Random（`Systematic_Tuner`）相比，最終分數、災難性失敗率、收斂速度各如何？
2. **LLM 的「記憶策略」怎麼影響決策品質**：把歷史 trial 餵給 LLM 的方式（全部歷史 / 摘要 / 滑動視窗 / 主動檢索工具）會不會造成明顯差異？
3. **系統穩健性機制值不值得**：像「重複設定安全網」（避免 LLM 建議完全重複的設定而浪費一次 trial）這種工程機制，拿掉之後代價有多大？

---

## 2. 系統架構

![系統架構總覽](../png/01_system_architecture_overview.png)

- **`Global_Tuner/`**：LLM 決策版本。每輪把當前歷史（依記憶策略而定）組成 prompt，交給 LLM 選下一組壓縮設定；`Strategy/executors.py` 負責在獨立子行程中實際執行壓縮＋評測，保證 VRAM 100% 釋放。
- **`Systematic_Tuner/`**：統計優化版本，用 Optuna 的 TPE / NSGA-II / Random sampler 取代 LLM 決策，其餘（搜尋空間、執行器、計分）與 `Global_Tuner` 完全共用，確保兩邊比較公平。
- **`Strategy/`**：兩個 tuner 共用的核心——`schemas.py`（設定的 Pydantic schema）、`executors.py`（8 種壓縮 mode 的實際執行與計分邏輯）、`trial_naming.py`。
- **`Method/`**：底層壓縮演算法實作（`quantize.py` 含 GPTQ/AWQ/BNB/QQQ、`asvd.py`、`sparse.py`）。

![Decision-maker prompt 結構](../png/02_decision_maker_prompt_structure.png)
![Executor 壓縮流程](../png/03_executor_compression_pipeline.png)

### 2.1 搜尋空間：8 種壓縮模式

`asvd_only`、`gptq`、`awq`、`qqq`、`bnb`、`sparse_unstructured`（SparseGPT 非結構化）、`sparse_structured`（2:4 / 4:8）、`hybrid_asvd_bnb`（ASVD + BNB 混合），每種模式底下再有各自的連續／離散超參數（詳見 `Systematic_Tuner/README.md`）。

### 2.2 計分公式

![計分公式](../png/04_score_formula.png)

```
norm_acc  = avg_acc  / base_acc
norm_lat  = base_lat / avg_lat
norm_vram = base_vram / max_vram
norm_emit = base_emit / avg_emit

weight_score = 1.0 + acc_w·log(norm_acc) + lat_w·log(norm_lat)
                    + vram_w·log(norm_vram) + emit_w·log(norm_emit)

penalty      = pen_a · max(0, (base_acc − pen_t) − avg_acc)
final_score  = weight_score − penalty
```

所有實驗實際使用的權重與參數（`Strategy/executors.py`、兩個 orchestrator 的 CLI 預設值，也是 `experiment_config.json` 裡實際記錄的值）：

| 參數 | 值 | 意義 |
|---|---|---|
| `acc_weight` | 3.0 | 準確度權重最高，明確以「準確度不能崩」為前提去換資源 |
| `lat_weight` / `vram_weight` / `emit_weight` | 1.0 / 1.0 / 1.0 | 延遲、VRAM、碳排放同權重 |
| `pen_t`（容忍閾值） | 0.15 | 準確度掉超過 15 個百分點才開始罰 |
| `pen_a`（懲罰係數） | 10.0 | 一旦超過閾值，懲罰增長很快 |

log-scale 讓「低基準值的相對改善」被放大權重，`score > 1.0` 代表優於未壓縮基線。

### 2.3 LLM 的 4 種記憶策略（`--memory_type`）

| 策略 | 做法 | 取捨 |
|---|---|---|
| `full` | 每輪把完整歷史丟給 LLM | 資訊最完整，但 token 成本最高、易受長上下文干擾 |
| `window` | 只給最近 N 筆 trial | 省 token，但可能忘記早期教訓、重蹈覆轍 |
| `summary` | 用次要 LLM 定期把歷史濃縮成「知識摘要」 | 保留長期洞察且省 token，但多一次摘要呼叫、偶有失真 |
| `tool` | 給 LLM 一個 `retrieve_trials` 工具，用查詢主動檢索相關歷史 | 最貼近人類 debug 方式，基礎 prompt 最精簡 |

---

## 3. 實驗設計

固定變因：模型 `Llama-3.2-3B-Instruct`、任務 `GSM8K`（500 題）、baseline（未壓縮）指標：

| Accuracy | Latency | VRAM | Emissions |
|---|---|---|---|
| 0.718 | 1422.6 s | 6.195 GB | 0.0937 |

共比較 **7 種搜尋方法**：LLM 的 4 種記憶策略（`Global_Tuner`）＋ TPE / NSGA-II / Random（`Systematic_Tuner`）。每種方法各跑 3 次（不同 random seed / run），據此估計平均表現與變異。

三組資料夾對應三個實驗批次（皆為同一套流程，差異列於下表）：

| 資料夾 | Trial 預算 | 涵蓋方法 | 重複設定安全網 | 用途 |
|---|---|---|---|---|
| `final_results/` | 20 trial × 3 run | 全部 7 種 | 開（最多重打 5 次 API） | 主要比較：LLM vs 統計搜尋 |
| `final_results_30/` | 30 trial × 3 run | 全部 7 種 | 開 | Trial 預算加大後結論是否穩固 |
| `final_results_30_noretry/` | 30 trial × 3 run | 僅 LLM 4 種記憶策略 | 關（重複設定直接記為跳過、不重打 API） | 安全網機制的消融實驗 |

`final_results/baselines/`、`final_results_30/baselines/` 快取 baseline 評測結果供各 run 重用；`final_results_30_noretry/` 沒有獨立 baseline 快取，改用各 run 自帶的 baseline 欄位（與上面數字一致）。

此外，`results/` 與 `results_llmc/` 是更早期、逐一手動跑單一壓縮設定（GPTQ / AWQ / BNB / SparseGPT 等）並在 **5 個任務**（GSM8K、BBH、HumanEval、TruthfulQA、CommonsenseQA）上評測的驗證性實驗，涵蓋 1B/3B/8B 三種模型尺寸；`results_llmc` 額外用另一套量化工具鏈（`llmc`）交叉驗證同樣的方法。這兩批資料不是「搜尋方法比較」的一部分，而是搜尋框架落地前，確認底層壓縮 executor 本身正確可用、且結論可跨任務／跨模型泛化的支撐材料。

---

## 4. 成果

### 4.1 20 trial（`final_results/`）——主要比較

| 方法 | 平均最佳分數 | 標準差 | 正向 trial 率 | 災難性失敗率 (score < −10) |
|---|---|---|---|---|
| **LLM-Full** | **4.694** | 0.036 | 53.3% | **5.0%** |
| LLM-Tool | 4.679 | 0.031 | 42.4% | 5.1% |
| LLM-Window | 4.622 | 0.045 | 56.9% | 8.6% |
| Random | 4.530 | 0.055 | 42.4% | 28.8% |
| LLM-Summary | 4.521 | 0.134 | 61.1% | 7.4% |
| TPE | 4.468 | 0.179 | 54.4% | 21.1% |
| NSGA-II | 4.426 | 0.021 | 44.1% | 30.5% |

### 4.2 30 trial（`final_results_30/`）——預算加大後結論是否穩固

| 方法 | 平均最佳分數 | 標準差 | 正向 trial 率 | 災難性失敗率 |
|---|---|---|---|---|
| **LLM-Full** | **4.720** | 0.038 | 66.7% | 5.6% |
| LLM-Window | 4.713 | 0.062 | 58.0% | 3.7% |
| LLM-Tool | 4.707 | 0.018 | 50.6% | 6.0% |
| LLM-Summary | 4.706 | 0.052 | 63.3% | 6.3% |
| Random | 4.645 | 0.034 | 42.5% | 33.3% |
| TPE | 4.603 | 0.005 | 57.5% | 23.0% |
| NSGA-II | 4.578 | 0.023 | 36.4% | 36.4% |

**結論一致且更明顯**：無論 20 或 30 trial，四種 LLM 策略的最佳分數全數贏過三種統計方法，且災難性失敗率穩定壓在個位數／低雙位數 %，而統計方法始終落在 21%～36%。trial 預算加大後，LLM 方法的正向 trial 率也同步提升（多數 > 58%），顯示優勢不是「trial 數少時的偶然」，而是持續性的。

### 4.3 重複設定安全網消融（`final_results_30_noretry/`，僅 LLM 4 策略；完整分析見 [`retry_ablation_analysis.md`](retry_ablation_analysis.md)）

| 方法 | 平均最佳分數 | 標準差 | 正向 trial 率 | 災難性失敗率 | **浪費（跳過）率** |
|---|---|---|---|---|---|
| **LLM-Full** | **4.698** | 0.026 | 66.7% | 7.9% | 30.0% |
| LLM-Window | 4.643 | 0.118 | 66.0% | 8.0% | 44.4% |
| LLM-Tool | 4.612 | 0.092 | 42.6% | 9.3% | 40.0% |
| LLM-Summary | 4.598 | 0.120 | 65.2% | 8.7% | 48.9% |

拿掉安全網後，各策略最終**最佳分數**下降幅度不大（LLM-Full 4.720→4.698，僅 −0.5%），但有 **30%～49% 的 trial 直接被浪費**（LLM 建議了與過去完全重複的設定，不重打 API、直接跳過）。也就是說：

- 安全網的價值主要不是「讓最終最佳解變好」，而是「讓每一次 API 呼叫的產出效率變高」——關掉安全網等於平白損失 3 成到 5 成的有效 trial 預算。
- `summary`／`window` 策略（依賴壓縮過的歷史）浪費率明顯高於 `full`（看得到完整歷史），符合直覺：資訊被壓縮後，LLM 較容易「忘記」自己已經建議過某個設定。
- 標準差全面上升（例如 LLM-Summary 0.052→0.120），代表拿掉安全網也讓不同 run 之間的穩定性變差。
- 這批資料有真實 token 記錄(非估計值)。把跳過率一併算進 token 成本後，**Full 與 Window 性價比最好，Tool 最差**——多輪工具檢索平均每個有效 trial 要多花約 2 倍 token，卻沒有換到更高分數(細節見 [`retry_ablation_analysis.md`](retry_ablation_analysis.md) §7)。

### 4.4 質化分析重點(詳見 [`final_results_analysis.md`](final_results_analysis.md))

- **有組織的探索**：LLM 在前 8 個 iteration 系統性地把 8 種壓縮模式各試過一輪，而非隨機亂試；之後轉為針對最佳模式（GPTQ 4-bit）做超參數微調。
- **機制性歸因**：LLM 能推理出「ASVD + BNB 混合效果差是因為相容性問題」而不只是記住分數低，因此後續能更精準迴避地雷。
- **收斂速度**：LLM 平均在第 10 個 trial 左右就找到最終最佳配置，TPE 平均要到第 18 個。
- **最終推薦設定**：**GPTQ 4-bit（group_size=128, damp_percent ≈ 0.03～0.05）**——以約 5～6% 的準確度代價，換取延遲 −66.5%、VRAM −62.6%、碳排放 −82.4%，是目前跨所有實驗一致勝出的 Green AI 交換比。

### 4.5 互動查驗

上述數字皆可在 `streamlit run dashboard/app.py` 中重現與細看：

- 「Final Results (20 Trial)」「Final Results 30 (30 Trial)」：各自的完整 trial 明細、Pareto 散佈圖。
- 「No-Retry 消融 (30 Trial)」：安全網開關對照。
- 「綜合比較」：三個資料夾並排看 headline 指標。

### 4.6 各記憶策略的 reasoning 特別發現(詳見 [`llm_reasoning_findings.md`](llm_reasoning_findings.md))

§4.4 談的是「LLM 整體上做對了什麼」；這份新文件做的是更細的事——直接讀四種記憶策略
(`full`/`window`/`summary`/`tool`)的原始 `reasoning` 文字逐輪比對，找**各模式特有**的行為
模式與盲點，重點案例都已用原始 json 二次核對：

- **Window**：有一個乾淨的「視窗外遺忘」實例——iter 9 找到較好設定，iter 17 因視窗看不到
  iter 9 而誤判成「尚未嘗試」，反而選了更激進、未驗證的設定，accuracy 從 0.376 崩到 0.028。
- **Summary**：摘要更新 prompt 明文禁止記住具體參數，代價是 `gptq(group_size=16)` 被重複
  試了三次都拿到偏低分數，摘要沒辦法講出「這個地雷」這麼具體的結論。
- **Tool**：會誤讀自己剛檢索到的證據——明明工具回傳 `group_size=128`，reasoning 卻寫成
  「full matrix」，然後重複同一個爛設定。
- **跨模式共同盲點**：QQQ 搭配 `group_size=128` 幾乎必然崩潰，四種模式的 reasoning 全部
  誤判成 `damp_percent` 的問題，整個資料集只有兩筆 trial 試過 `group_size=-1`，且都是誤打
  誤撞而非因果推理——與記憶架構無關，是決策者本身的局限。
- 文件末尾也盤點了 dashboard 現有分析手法哪些做得細緻、哪些資料存在但沒被用到
  (例如 `tool_debug_log.jsonl` 的查詢理由欄位從未被顯示過)。

---

## 5. 現有限制與後續方向

- 目前所有搜尋方法比較都限定在單一模型（3B）＋單一任務（GSM8K），`results/`／`results_llmc/` 的多任務結果只驗證了個別壓縮設定本身有效，尚未把「LLM-guided 搜尋」跑在別的模型／任務組合上做同等規模比較。
- LLM 的最佳絕對分數並非全域最高（統計方法在無預算限制、多次運氣好的情況下偶爾能小幅超車），LLM 的優勢明確建立在「trial 預算有限」的場景。
- `tool` 記憶策略目前只有量化數據，尚缺 [`final_results_analysis.md`](final_results_analysis.md) 那種逐輪推理過程的質化解讀，可作為後續分析補充。

---

*本文件由分析 `dashboard/common/loaders.py` 的彙總邏輯重新跑一遍 `final_results*` 三個資料夾實際數據、並比對 [`final_results_analysis.md`](final_results_analysis.md) 的質化內容整理而成，數字口徑與 dashboard 內顯示的一致。*
