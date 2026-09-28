"""首頁：導覽說明，沒有任何資料載入邏輯，純粹是入口說明頁。"""
import streamlit as st

st.set_page_config(page_title="Green AI 量化搜尋分析", page_icon="🌿", layout="wide")

st.title("🌿 Green AI 量化搜尋分析總覽")
st.markdown("""
本站整合三組實驗配置的分析，模型與任務皆為 **Llama-3.2-3B-Instruct / GSM8K**：

| 配置 | Trial 數 | 去重複重試 | 內容 |
|------|---------|-----------|------|
| **20-Trial** | 20/run | 5 次安全網 | 7 種搜尋策略（LLM ×4 + TPE/NSGA-II/Random）完整比較，含 LLM 優勢分析報告 |
| **30-Trial** | 30/run | 5 次安全網 | 同上，trial 數擴大到 30，含 Late-stage 改善分析 |
| **No-Retry 消融（30 Trial）** | 30/run | **關閉**（重複即浪費該 trial） | 只涵蓋 4 種 LLM 記憶策略，驗證安全網被拿掉後，各模式的原始記憶可靠度 |

請從左側導覽選擇要查看的頁面：

- **📁 Final Results (20 Trial)** / **📁 Final Results 30 (30 Trial)**：各自完整的詳細分析
  dashboard（Benchmark 比較、跨實驗比較、Trial 彩色分析、Pareto Frontier、LLM 優勢分析報告等），
  內容包含針對各自資料集客製化撰寫的文字說明，兩者結構相同但文字與部分圖表因 trial 數不同而有調整。
- **🧪 No-Retry 消融 (30 Trial)**：新增的第三組配置，聚焦在去重複重試機制的消融實驗。
  標題特意標出 trial 數，因為之後若出現其他 trial 數的 no-retry 對照組
  （例如 20-Trial No-Retry），才不會跟這個混淆。
- **🆚 綜合比較**：跨三組配置的輕量 headline 比較，不重複前面三頁的詳細內容，
  適合用來快速看這三組設定之間差在哪。
""")
