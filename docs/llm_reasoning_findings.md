# LLM Reasoning 特別發現:四種記憶策略的行為模式 + dashboard 分析盤點

> 前三份文件(`EXPERIMENT_OVERVIEW.md`、`final_results_analysis.md`、`retry_ablation_analysis.md`)都是「數字層級」的比較——分數、跳過率、token 成本。本文做的是不一樣的事:直接讀 `full` / `window` / `summary` / `tool` 四種記憶策略在 `optimization_results.json` 裡 `suggestion.reasoning` 欄位的**原始文字**,以及 `tool` 模式獨有的 `tool_debug_log.jsonl`,找每種模式特有的行為模式、盲點與失誤;並順手盤點一次 `dashboard/` 現有的分析手法,記錄哪些做得細緻、哪些資料存在但沒被用到。
>
> 方法:對每種模式,從 `final_results/`、`final_results_30/`、`final_results_30_noretry/` 三批各挑代表性 run,逐輪讀取 reasoning 文字並對照當輪與後續的實際 metrics。下面每一則發現都附檔案路徑與 iteration,重要的幾則已用 Python 重新讀取原始 json 二次核對過(標註「已核對」)。

---

## 1. Full(全部歷史)——資訊最完整,但仍會調錯旋鈕

Full 模式每輪都拿到完整歷史,理應最不容易忘記過去的教訓,實際上確實浪費率最低,但**看得到失敗不等於能正確診斷失敗原因**:

- `final_results_30/exp_Llama-3.2-3B-Instruct_gsm8k_full_20260828_191143`:iter 5 試 `qqq(group_size=128, damp=0.005)`,accuracy 崩到 0.01(score −16.85);iter 21 換成 `qqq(group_size=128, damp=0.001)`,結果幾乎一樣差(score −15.61)。兩次都只調整了 `damp_percent`,`group_size` 從未被當成懷疑對象——詳見 §5 跨模式共同盲點。
- 用詞精確、會指名道姓引用過去的 iteration 與分數,例如 iter 16(`final_results/exp_..._full_20260824_150718`)寫道:「Iter 10 and Iter 14, showing strong performances... damp_percent at 0.006 in Iteration 14 offered the highest score」。
- No-retry 消融下浪費率 **30.0%**,四種模式中最低(見 [`retry_ablation_analysis.md`](retry_ablation_analysis.md))。

---

## 2. Window(滑動視窗)——「視窗外遺忘」的教科書案例

這是四種模式裡發現得最乾淨的失敗模式,**已用 Python 重新讀取 json 核對**,reasoning 原文與實際 metrics 完全吻合:

`final_results/exp_Llama-3.2-3B-Instruct_gsm8k_window_20260323_165306`:

| Iteration | 設定 | Accuracy | Score | Reasoning 摘錄 |
|---|---|---|---|---|
| 9 | `asvd_only`,調高 `param_ratio_target` | 0.376 | −0.21 | 「a revisit to the asvd_only mode with a new param_ratio_target could fill the remaining gaps... resulted in a significant accuracy drop but also showed a great reduction...」 |
| 17(視窗已看不到 iter 9) | `asvd_only`,更激進的壓縮率 | **0.028** | **−11.48** | 「The 'asvd_only' mode with the specified settings **has only been tried early (Iter 3)** with a high param_ratio_target (0.90)... A trial with a heavier compression...」 |

iter 17 完全遺忘了 iter 9 那次比較好的結果(0.376),誤以為 asvd_only 從 iter 3 之後就沒再試過,於是選了一個從未驗證過的更激進壓縮率,直接把 accuracy 打到 0.028。這正是「滑動視窗只看最近 N 筆」的結構性代價。

其他發現:

- No-retry 消融下 `final_results_30_noretry/exp_..._window_20260903_162712` 的 25 個 iteration 裡跳過 13 個(52%),包含連續 3 次(iter 9–11)與連續 4 次(iter 13–15、18)跳過。
- **模板洩漏 bug**:iter 19 的 reasoning 直接輸出了 prompt 格式範例的佔位文字——「A detailed explanation of why this mode fills a gap in current coverage, followed by a comprehensive analysis of the expected trade-offs and potential risks.」——這段文字逐字對應 `llm_client.py` 裡要求 LLM 輸出格式的說明文字,不是真正的推理。另一個 window run(`final_results_30_noretry/exp_..._window_20260903_194900`)的 **iter 1**(第一輪!)也出現一模一樣的洩漏。

---

## 3. Summary(LLM 摘要)——摘要機制設計上就禁止記住具體參數,代價可驗證

`Global_Tuner/llm_client.py` 裡負責更新摘要的 prompt 明文寫著:

> 🛑 CRITICAL CONSTRAINT: DO NOT specify exact parameter combinations to run next. Define the "rules of the game."

這個限制不是理論上的疑慮,而是有實際代價:`final_results_30/exp_Llama-3.2-3B-Instruct_gsm8k_summary_20260827_093022` 裡,`gptq(group_size=16)` 被重複試了**三次**:

| Iteration | Score | Reasoning 態度 |
|---|---|---|
| 18 | 2.67 | 「out of curiosity for its potential unexplored effects」 |
| 21 | 2.74 | 仍當作新方向嘗試 |
| 24 | 2.58 | 仍當作新方向嘗試 |

三次分數都落在 2.6~2.7,比 group_size 32/64/128 的基準(約 4.3~4.7)低了整整 2 分左右,但因為摘要只能講「安全區/危險區」這種模糊規則,講不出「group_size=16 是地雷」這種具體結論,每次都被重新包裝成「還沒探索過的新想法」。

其他發現:

- 四個記憶策略中,浪費率最高、也退化得最嚴重:`final_results_30_noretry/exp_..._summary_20260904_030115` 的 25 個 iteration 裡跳過 15 個(**60%**),而且最後 **iter 20–25 連續 6 次全部跳過**——整個 run 在尾聲直接停擺。
- reasoning 文字會直接沿用摘要的專有詞彙,例如「These configurations fall within the 'safe zones' identified」(`final_results_30_noretry/exp_..._summary_20260904_030115` iter 17)、「reinforcing a direction away from these strategies」(iter 19)——顯示摘要的結構會反過來限制、甚至扭曲後續決策的語言與思路,不只是遺失資訊而已。
- 也有「預期與現實不符」的例子:`final_results/exp_..._summary_20260324_064745` iter 14 預測幫 `hybrid_asvd_bnb` 加上 `use_double_quant=True` 會「help retain accuracy」,實際上 accuracy 從 iter 5(無 double-quant)的 0.162 掉到 iter 14 的 0.148,調整方向反而是錯的。

---

## 4. Tool(工具檢索)——最囉唆,且會誤讀自己剛檢索到的證據

**已用 `tool_debug_log.jsonl` 二次核對的幻覺案例**——`final_results_30/exp_Llama-3.2-3B-Instruct_gsm8k_tool_20260828_020146`:

- iter 19 的工具檢索(`tool_debug_log.jsonl`)明確回傳:qqq 過去唯一一次 trial 的設定是 `"group_size": 128`(score −16.85)。
- iter 20 的 reasoning 卻寫:「a fallback to a simpler, well-balanced approach like QQQ might be worth evaluating again... since it was **only tested once and with a full matrix** which usually offers better accuracy than hybrid」——把剛剛檢索到的 `group_size=128` 記成了 full matrix(`group_size=-1`)。
- 於是 iter 20 重複了同一個爛設定(`group_size=128, damp=0.01`),拿到幾乎一樣差的結果:score −16.28。

也就是說,LLM 不是「忘記」了證據(它剛拿到、也引用了),而是**在自己的上下文裡編造了一個跟證據矛盾的細節**,這跟 window/summary 的「資訊遺失型」失誤性質不同,是「誤讀型」失誤。

其他發現:

- 同一個 run 也有工具用對地方的正面案例:iter 16 之後針對 `asvd_only` 的 `ratio` 反覆追問微調(0.92→0.95→0.96),accuracy 單調上升 0.278→0.408→0.44,顯示工具檢索用在「已知有效方向的精修」上確實有幫助。
- Reasoning 平均字數是四種模式裡最長的(約 124.6 字 vs 其他三者約 108 字),常常花額外篇幅覆述檢索到的證據內容。
- No-retry 浪費率 40.0%,排第二低(僅次於 Full)。
- `tool_debug_log.jsonl` 的 `reason` 欄位(工具自己解釋「為什麼要查這個」)本身也顯示出偏執傾向:例如 `final_results/exp_..._tool_20260823_220429` 從 iter 16 起幾乎只反覆查詢 `asvd` 與 `hybrid`——恰好是表現最差的兩種模式——用近乎相同的措辭(「Explore variations of asvd_only to improve results...」),卻很少回頭查詢真正表現穩定的 `bnb`、`awq`。工具檢索被拿來「搶救」失敗方法,而不是「鞏固」已知有效的方法。

---

## 5. 跨模式共同盲點:QQQ 的 group_size 從未被正確診斷

這是四種記憶策略共有、與記憶設計本身無關的盲點,值得特別標出:

`qqq` 搭配預設的 `group_size=128` 幾乎每次都讓 accuracy 崩到 0.01~0.03(score −13~−17),四種模式、幾十次 trial 一致如此。**每一次失敗,所有模式的 reasoning 都把矛頭指向 `damp_percent`**(Hessian dampening),嘗試調整它來「修正」問題——Full 模式甚至把 damp 從 0.005 調到 0.001,方向調錯也沒能發現。

整個資料集裡,只有兩筆 trial 試過 `group_size=-1`(full matrix,已用 Python 核對):

| Run | Iteration | Accuracy | Score |
|---|---|---|---|
| `final_results_30/exp_..._window_20260826_171618` | 25 | 0.454 | 1.03 |
| `final_results_30/exp_..._tool_20260828_134529` | 19 | 0.448 | 0.94 |

兩次都只是「換個新設定試試看」誤打誤撞碰到的,reasoning 裡並沒有任何一次明確指出「group_size 才是關鍵」的因果推論。**這代表無論給 LLM 多完整的歷史、多好的摘要或多強的檢索工具,它在這個特定失敗模式上都沒有形成正確的因果模型**——是決策者本身(或 prompt 設計)的局限,不是記憶架構的問題。

---

## 6. Dashboard 分析手法盤點(附錄)

### 6.1 已經做得細緻、值得記錄的手法

- **Weight 敏感度 rescore 工具**(`report_30trial.py` 等):6 個 slider(acc/lat/vram/emit weight + pen_t + pen_a)即時用正式的 log-scale score 公式重新排序所有 trial——是 dashboard 裡唯一真正「可互動改變結論」的功能,不只是換個視角看同一份排名。
- **Decision-pattern 行為指紋表**:直接從 trial 序列(而非最終分數)萃取「首次嘗試 GPTQ 的 iteration」「首次失敗後復原的 iteration」「首個連續 3 次 GPTQ streak」等行為模式,是輕量但巧妙的分析角度。
- **Within-run vs between-run 變異數分解**(類 ANOVA)、**Cumulative Regret**、**Sample Complexity CDF**(用 Plotly `updatemenus` 按鈕切換 4 種分數門檻、不用重新渲染頁面)——比一般展示型 dashboard 更接近 bandit/AutoML 論文等級的評估工具。
- Token 成本分析誠實區分 **EXACT(可逐字重建)vs 隱藏事件(估計值)**,並在文字裡明確講清楚估計值可能低估的原因。

### 6.2 已確認的分析缺口

**已於本文完成後動手補上的**(2 項,見 §7 的實作紀錄):

- ~~`tool_debug_log.jsonl` 的 `reason` 欄位從未被任何 dashboard 頁面讀取或顯示~~——已接進三個報告頁的「🤖 LLM Suggestion 記錄」區塊,tool 模式的 trial 現在展開後會直接看到當輪的工具查詢記錄。
- ~~Summary 模式的知識摘要演化文字從未存檔~~——`Global_Tuner/llm_client.py` 現在每次更新摘要都會多寫一份 `knowledge_summary_history.jsonl`;**注意這只對之後新跑的 run 有效,本文 §3 分析用的既有資料仍然沒有這份檔案**,§3 的結論依然只能從摘要對決策的影響反推。

**還沒補、之後可以補的**:

1. **dashboard 完全沒有即時統計檢定**——`dashboard/` 底下沒有任何檔案 `import scipy`,所有 Mann-Whitney U / Fisher's exact 的數字都是 `analysis/analyze_retry_ablation.py` 跑過一次後,手動貼成 markdown 裡的字面值,不是即時算出來的。
2. **沒有 config diff 工具**:儘管有很多並排的 Pareto 表、最佳 trial 表,卻沒有一個「選兩個 trial/實驗,結構化比較設定差異」的小工具。
3. **`llm_usage_log.jsonl` 存在但沒被即時解析**——`report_30trial_noretry.py` 第 6 節顯示「真實記錄,非估計」的 token 表,實際上是寫死在 Python 裡的常數,不是即時讀檔算出來的,儘管這個檔案裡的 `dedup_attempt`/`json_attempt`/`validation_passed` 等欄位其實可以拿來重新驗證這些數字。
4. **沒有對 reasoning / tool query 文字的關鍵字搜尋或篩選功能**——目前瀏覽 reasoning 的方式只有「展開單筆 trial 的 expander」或「作者手選幾個 iteration 區間精讀」,無法直接搜尋「哪些 trial 的 reasoning 提到了 ASVD」這類問題。

**資料層級已經消失、既有 run 補不回來的**:

5. **既有 run 的 Summary 模式知識摘要演化文字已永久遺失**——原因同上,原始 1.5GB stdout log 也已刪除,現有三批資料(`final_results*`)都無法回頭補這份記錄,只有未來新跑的 run 才會有。

---

## 7. 給未來實驗的具體建議

對應上面「資料還在、可以補」的缺口,若要繼續跑新一批實驗,值得考慮:

- ✅ **已實作:在 `Global_Tuner/llm_client.py` 把每輪的 `knowledge_summary` 存到磁碟**——`_update_knowledge_summary()` 現在每次更新都會多寫一行到 `knowledge_summary_history.jsonl`(`{timestamp, iteration, summarized_through_idx, knowledge_summary}`),`dashboard/pages_custom/_landing.py` 的擴充檔案清單也同步補上這個檔名。只對之後新跑的 run 有效,既有資料補不回來。
- ✅ **已實作:把 `tool_debug_log.jsonl` 的 `reason` 欄位接進 dashboard**——`report_20trial.py`/`report_30trial.py`/`report_30trial_noretry.py` 的「🤖 LLM Suggestion 記錄」區塊,tool 模式的每個 trial 展開後現在會多顯示當輪的工具查詢記錄(查了什麼、為什麼查、查到什麼),不用再手動翻 `tool_debug_log.jsonl`。
- **`report_30trial_noretry.py` 的 token 用量表改成即時解析 `llm_usage_log.jsonl`**,而非目前寫死的常數——順便可以拿真實的 `dedup_attempt`/`json_attempt` 欄位驗證舊有的估計值準不準。(尚未實作)
- 若要驗證 §5 的「QQQ group_size 盲點」是不是 prompt 設計問題,可以考慮在 prompt 的 few-shot 範例或參數說明裡,明確提示 `group_size=-1` 是 QQQ 的一個合法選項並說明其代價/好處,看看加了明確提示後 LLM 會不會自己發現這個因果關係。這項會改變未來實驗的 prompt 設計本身,需要跑新一批實驗才能驗證有沒有用,故本次未一併實作。(尚未實作)

---

*本文件的四種模式發現整理自對 `optimization_results.json`(欄位 `suggestion.reasoning`)與 `tool_debug_log.jsonl` 的直接閱讀,重點案例(§2 window 遺忘案例、§4 tool 幻覺案例、§5 QQQ group_size=-1 的兩筆 trial)已用 Python 重新讀取對應檔案二次核對過,並非只憑摘要轉述。§6 的 dashboard 盤點來自逐檔閱讀 `dashboard/pages_custom/*.py` 與 `dashboard/common/*.py`。*
