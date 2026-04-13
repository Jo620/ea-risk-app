"""
基于机器学习的术后躁动风险预测系统
streamlit run app.py
"""
import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import streamlit as st
warnings.filterwarnings("ignore")
plt.rcParams["axes.unicode_minus"] = False

EN2ZH = {
    "ALB":                                      "白蛋白",
    "Hb":                                       "血红蛋白",
    "Analgesics before the end of the surgery": "手术结束前给予镇痛药",
    "Age":                                      "年龄",
    "Duration of anesthesia":                   "麻醉时长",
    "Weight":                                   "体重",
    "Education Degree":                         "文化程度",
    "Dexmedetomidine":                          "右美托咪定",
    "Ultrasound-guided nerve block":            "超声引导神经阻滞",
    "ASA class1f1cat1on":                       "ASA 分级",
    "Hypertension":                             "高血压",
    "Type of surgery":                          "手术类型",
    "PLT":                                      "血小板",
    "Blood calcium":                            "血钙",
}
CAT_FEATS = {
    "Analgesics before the end of the surgery","Dexmedetomidine",
    "Ultrasound-guided nerve block","ASA class1f1cat1on",
    "Hypertension","Type of surgery","Education Degree",
}

st.set_page_config(page_title="EA Risk Predictor", page_icon="🏥",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif;}
.stApp{background:#f0f4f8;}

section[data-testid="stSidebar"]{background:#1a2332 !important;border-right:none !important;}
section[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
section[data-testid="stSidebar"] .stSelectbox>div>div{
  background:#253045 !important;border-color:#3d4f6b !important;border-radius:8px !important;}
section[data-testid="stSidebar"] input{
  background:#253045 !important;border-color:#3d4f6b !important;border-radius:8px !important;}
section[data-testid="stSidebar"] label{color:#94a3b8 !important;font-size:.8rem !important;}
section[data-testid="stSidebar"] hr{border-color:#2d3f58 !important;}

.hero{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#0f4c81 100%);
  padding:24px 32px 20px;border-radius:16px;margin-bottom:22px;
  box-shadow:0 4px 24px rgba(15,23,42,.18);position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-40px;right:-40px;width:180px;height:180px;
  border-radius:50%;background:rgba(59,130,246,.12);}
.hero h1{color:#fff;font-size:1.45rem;font-weight:800;margin:0;letter-spacing:-.01em;}
.hero p{color:#7eb8f7;font-size:.78rem;margin:6px 0 0 0;}

.card{background:#fff;border-radius:14px;padding:18px 20px;margin-bottom:14px;
  box-shadow:0 1px 8px rgba(15,23,42,.07);border:1px solid #e8edf3;}
.card-title{font-size:.65rem;font-weight:700;color:#94a3b8;letter-spacing:.12em;
  text-transform:uppercase;padding-bottom:8px;margin-bottom:10px;
  border-bottom:1px solid #f1f5f9;}

.prob-wrap{text-align:center;padding:14px 0 10px;}
.prob-num{font-size:3.6rem;font-weight:800;line-height:1;letter-spacing:-.02em;}
.prob-tag{display:inline-block;margin-top:8px;padding:4px 14px;border-radius:999px;
  font-size:.78rem;font-weight:700;letter-spacing:.06em;}
.c-high{color:#dc2626;} .bg-high{background:#fee2e2;color:#dc2626;}
.c-mid{color:#d97706;}  .bg-mid{background:#fef3c7;color:#d97706;}
.c-low{color:#16a34a;}  .bg-low{background:#dcfce7;color:#16a34a;}

div.stButton>button{background:linear-gradient(135deg,#1e40af,#2563eb);
  color:white;border:none;border-radius:10px;padding:13px 0;
  font-size:.88rem;font-weight:700;width:100%;letter-spacing:.04em;
  box-shadow:0 4px 12px rgba(37,99,235,.35);}
div.stButton>button:hover{opacity:.88;transform:translateY(-1px);}
.feat-zh{font-size:.7rem;color:#64748b;font-weight:600;margin-bottom:1px;}
</style>""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model_and_meta():
    meta_path  = os.path.join(BASE_DIR, "meta.json")
    model_path = os.path.join(BASE_DIR, "CatBoost.pkl")
    meta, model = {}, None
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    return model, meta

model, meta = load_model_and_meta()
features = meta.get("features", [])

@st.cache_resource
def get_explainer(_model, _features):
    """
    使用新版 shap.Explainer（与 notebook Cell 26 保持一致）。
    返回的 SHAP 值在 log-odds 空间，base_value 约为 -4.251，与 notebook 结果完全一致。
    """
    try:
        import shap
        from sklearn.pipeline import Pipeline
        if isinstance(_model, Pipeline):
            prep   = _model.named_steps["prep"]
            core   = _model.named_steps["model"]
            # 用全零背景（与 app 无训练数据的部署场景匹配）
            bg = np.zeros((1, len(_features)))
            bg_t = prep.transform(pd.DataFrame(bg, columns=_features))
            explainer = shap.Explainer(core, bg_t, feature_names=_features)
        else:
            bg = np.zeros((1, len(_features)))
            explainer = shap.Explainer(_model, bg, feature_names=_features)
        return explainer
    except Exception as e:
        st.warning(f"shap.Explainer 初始化失败: {e}")
        return None

if model is None:
    st.error("未找到 CatBoost.pkl，请将模型文件放在 app.py 同一目录。"); st.stop()
if not features:
    st.warning("meta.json 中未找到 features 列表。"); st.stop()

st.markdown("""
<div class="hero">
  <h1>🏥 成人全麻术后苏醒期躁动（EA）风险预测系统</h1>
  <p>ML-Based Emergence Agitation Risk Prediction &nbsp;·&nbsp; 输入患者参数 → 实时预测风险</p>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧾 Patient Input")
    st.markdown("<span style='color:#64748b;font-size:.8rem'>患者参数输入</span>", unsafe_allow_html=True)
    st.markdown("---")
    input_vals = {}
    for feat in features:
        zh = EN2ZH.get(feat,"")
        if zh:
            st.markdown(f"<div class='feat-zh'>{zh}</div>", unsafe_allow_html=True)
        
        if feat == "Analgesics before the end of the surgery":
            # 0=无（最高风险）> 1=阿片类 > 2=NSAIDs > 3=两者均用（最低风险）
            opts = {
                "0 — None / 无（风险最高）": 0,
                "1 — Opioids / 阿片类": 1,
                "2 — NSAIDs / 非甾体类": 2,
                "3 — Both / 阿片类+非甾体类（风险最低）": 3
            }
            sel = st.selectbox(feat, list(opts.keys()), key=feat)
            input_vals[feat] = opts[sel]
        elif feat == "Education Degree":
            # 0=文盲（风险最高）→ 4=大专及以上（风险最低）
            opts = {
                "0 — Illiteracy / 文盲（风险最高）": 0,
                "1 — Elementary school / 小学": 1,
                "2 — Middle school / 中学": 2,
                "3 — Technical secondary school / 高中": 3,
                "4 — College degree or above / 大专及以上（风险最低）": 4
            }
            sel = st.selectbox(feat, list(opts.keys()), key=feat)
            input_vals[feat] = opts[sel]
        elif feat == "ASA class1f1cat1on":
            opts = {
                "1 — Ⅰ级": 1,
                "2 — Ⅱ级": 2,
                "3 — Ⅲ级": 3
            }
            sel = st.selectbox(feat, list(opts.keys()), key=feat)
            input_vals[feat] = opts[sel]
        elif feat == "Type of surgery":
            opts = {
                "0 — Joint surgery / 关节外科": 0,
                "1 — Urology / 泌尿外科": 1,
                "2 — Gynecology / 妇科": 2,
                "3 — General surgery / 普外科": 3,
                "4 — Thoracic surgery / 胸外科": 4,
                "5 — E.N.T / 耳鼻喉": 5,
                "6 — Trauma Orthopedics / 创骨": 6,
                "7 — Spine surgery / 脊柱": 7
            }
            sel = st.selectbox(feat, list(opts.keys()), key=feat)
            input_vals[feat] = opts[sel]
        elif feat in CAT_FEATS:
            opts = {"0 — No / 否":0,"1 — Yes / 是":1}
            sel  = st.selectbox(feat, list(opts.keys()), key=feat)
            input_vals[feat] = opts[sel]
        else:
            input_vals[feat] = st.number_input(feat, value=0.0, format="%.2f", key=feat)
    st.markdown("---")
    calc_btn = st.button("🔍  Calculate Risk / 计算风险")

if calc_btn:
    X_input = pd.DataFrame([input_vals])[features].astype(float)
    prob    = float(model.predict_proba(X_input)[0,1])
    pct     = prob * 100
    if pct > 66.3:
        rc,rtxt,bc,tc = "c-high","HIGH RISK / 高风险","#dc2626","bg-high"
    elif pct >= 30:
        rc,rtxt,bc,tc = "c-mid","MODERATE / 中等风险","#d97706","bg-mid"
    else:
        rc,rtxt,bc,tc = "c-low","LOW RISK / 低风险","#16a34a","bg-low"
    st.session_state.update(dict(X_input=X_input,prob=prob,pct=pct,rc=rc,rtxt=rtxt,bc=bc,tc=tc))

if "prob" not in st.session_state:
    st.markdown("""
    <div style="text-align:center;padding:60px 0;color:#94a3b8;background:#fff;
                border-radius:16px;border:1px solid #e8edf3;
                box-shadow:0 1px 8px rgba(15,23,42,.06)">
      <div style="font-size:3rem">📋</div>
      <div style="margin-top:12px;font-size:1rem;font-weight:600;color:#64748b">
        请在左侧填写患者参数后点击「Calculate Risk」</div>
      <div style="margin-top:6px;font-size:.82rem">
        Enter patient details in the sidebar and click Calculate Risk.</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

X_input = st.session_state["X_input"]
prob    = st.session_state["prob"]
pct     = st.session_state["pct"]
rc      = st.session_state["rc"]
rtxt    = st.session_state["rtxt"]
bc      = st.session_state["bc"]
tc      = st.session_state["tc"]

# ── 行1 ────────────────────────────────────────────────────────────────────────
r1l, r1r = st.columns([1, 2.6], gap="large")
with r1l:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Prediction Probability / 预测概率</div>
      <div class="prob-wrap">
        <div class="prob-num {rc}">{pct:.1f}%</div>
        <div class="prob-tag {tc}">{rtxt}</div>
      </div>
      <div style="background:#f1f5f9;border-radius:999px;height:10px;overflow:hidden;margin:10px 0 4px">
        <div style="width:{pct:.1f}%;background:{bc};height:100%;border-radius:999px"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:.68rem;color:#94a3b8;margin-top:3px">
        <span>0%</span><span>50%</span><span>100%</span>
      </div>
    </div>""", unsafe_allow_html=True)

with r1r:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Internal Input Check / 输入数据汇总</div>', unsafe_allow_html=True)
    st.dataframe(X_input.round(2), use_container_width=True, height=76)
    st.markdown('</div>', unsafe_allow_html=True)

# ── SHAP ───────────────────────────────────────────────────────────────────────
explainer = get_explainer(model, features)
if explainer is None:
    st.info("SHAP explainer 初始化失败"); st.stop()

try:
    import shap
    from sklearn.pipeline import Pipeline

    if isinstance(model, Pipeline):
        prep = model.named_steps["prep"]
        X_t  = prep.transform(X_input)
        # shap.Explainer 返回 Explanation 对象
        shap_exp = explainer(X_t)
    else:
        shap_exp = explainer(X_input.values)

    # 取第一个样本的 SHAP 值（log-odds 空间，与 notebook 一致）
    sv       = np.array(shap_exp.values[0]).flatten()
    base_val = float(shap_exp.base_values[0])

    n  = min(len(sv), len(features))
    sv, fn = sv[:n], features[:n]

    r2l, r2r = st.columns([1, 1], gap="large")

    # ── 双向条形图 ─────────────────────────────────────────────────────────────
    with r2l:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Feature Impact · 特征影响</div>', unsafe_allow_html=True)
        st.caption("🔴 Increases Risk  ·  🟢 Decreases Risk")

        order  = np.argsort(np.abs(sv))[::-1]
        sv_p   = sv[order]; fn_p = [fn[i] for i in order]
        vv_p   = [float(X_input.iloc[0][f]) for f in fn_p]
        # ★ 修复：直接显示实际数值，不用科学计数法
        labels = [f"{f} = {v:g}" for f, v in zip(fn_p, vv_p)]

        fig1, ax1 = plt.subplots(figsize=(6,5))
        fig1.patch.set_facecolor("#ffffff"); ax1.set_facecolor("#ffffff")
        colors = ["#ef4444" if v>0 else "#22c55e" for v in sv_p]
        ax1.barh(labels[::-1], sv_p[::-1], color=colors[::-1], height=0.62,
                 edgecolor="none", alpha=0.88)
        ax1.axvline(0, color="#cbd5e1", linewidth=1.2, zorder=0)
        ax1.set_xlabel("SHAP Value (log-odds)", fontsize=8.5, color="#64748b", labelpad=6)
        ax1.tick_params(axis="y", labelsize=8, colors="#374151", length=0)
        ax1.tick_params(axis="x", labelsize=8, colors="#94a3b8")
        # ★ 修复：关闭 x 轴科学计数法
        ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax1.ticklabel_format(style='plain', axis='x')
        ax1.spines[["top","right","left","bottom"]].set_visible(False)
        ax1.grid(axis="x", color="#f1f5f9", linewidth=0.8, zorder=0)
        ax1.legend(handles=[
            mpatches.Patch(facecolor="#22c55e",alpha=.88,label="Decrease Risk"),
            mpatches.Patch(facecolor="#ef4444",alpha=.88,label="Increase Risk"),
        ], fontsize=8, frameon=False, loc="lower right")
        plt.tight_layout(pad=1.2)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Waterfall Force Plot ───────────────────────────────────────────────────
    with r2r:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Decision Force Plot · 决策力图</div>', unsafe_allow_html=True)
        st.caption("Waterfall: each bar shows one feature's contribution (log-odds space)")

        top_f   = 8
        idx_s   = np.argsort(np.abs(sv))[::-1][:top_f]
        idx_s   = idx_s[np.argsort(sv[idx_s])]
        sv_s    = sv[idx_s]
        fn_s    = [fn[i] for i in idx_s]
        vv_s    = [float(X_input.iloc[0][f]) for f in fn_s]

        n_bars  = len(sv_s)
        fig2, ax2 = plt.subplots(figsize=(6, 0.7 * n_bars + 2.2))
        fig2.patch.set_facecolor("#ffffff")
        ax2.set_facecolor("#ffffff")

        running = base_val
        starts, widths, cols = [], [], []
        for sv_i in sv_s:
            starts.append(running)
            widths.append(sv_i)
            cols.append("#ef4444" if sv_i >= 0 else "#22c55e")
            running += sv_i

        y_pos = np.arange(n_bars)

        for i,(s,w,c) in enumerate(zip(starts, widths, cols)):
            ax2.barh(y_pos[i], w, left=s, height=0.52,
                     color=c, alpha=0.88, edgecolor="white", linewidth=0.8)
            x_txt = s + w + (0.003 if w >= 0 else -0.003)
            ha    = "left" if w >= 0 else "right"
            ax2.text(x_txt, y_pos[i], f"{sv_s[i]:+.3f}",
                     ha=ha, va="center", fontsize=8, color="#374151", fontweight="600")

        for i in range(n_bars - 1):
            x_end = starts[i] + widths[i]
            ax2.plot([x_end, x_end], [y_pos[i]+0.28, y_pos[i+1]-0.28],
                     color="#cbd5e1", linewidth=0.9, linestyle="--")

        # ★ 修复：y 轴标签用 :g 格式，消除科学计数法
        ylabels = [f"{f}  =  {v:g}" for f, v in zip(fn_s, vv_s)]
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(ylabels, fontsize=8.5, color="#374151")

        ax2.axvline(base_val, color="#94a3b8", lw=1.4, linestyle="--", zorder=0)
        ax2.axvline(running,  color="#1e40af", lw=2.0, linestyle="-",  zorder=0)

        ybot = -0.8
        ax2.text(base_val, ybot, f"Baseline\nE[f]={base_val:.3f}",
                 ha="center", va="top", fontsize=7.5, color="#64748b",
                 fontweight="600", linespacing=1.4)
        ax2.text(running, ybot, f"Prediction\nf(x)={running:.3f}\n(prob={prob:.3f})",
                 ha="center", va="top", fontsize=7.5, color="#1e40af",
                 fontweight="700", linespacing=1.4)

        all_x = starts + [running, base_val]
        xmin, xmax = min(all_x), max(all_x)
        pad = max((xmax - xmin) * 0.25, 0.08)
        ax2.set_xlim(xmin - pad, xmax + pad)
        ax2.set_ylim(-1.2, n_bars - 0.3)

        ax2.spines[["top","right","left"]].set_visible(False)
        ax2.spines["bottom"].set_color("#e2e8f0")
        ax2.tick_params(axis="x", labelsize=8, colors="#94a3b8")
        ax2.tick_params(axis="y", length=0)
        ax2.set_xlabel("Model output value (log-odds)", fontsize=8, color="#64748b", labelpad=6)
        ax2.grid(axis="x", color="#f8fafc", linewidth=0.8, zorder=0)
        # ★ 修复：关闭 x 轴科学计数法
        ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax2.ticklabel_format(style='plain', axis='x')

        ax2.legend(handles=[
            mpatches.Patch(facecolor="#22c55e",alpha=.88,label="Decrease Risk"),
            mpatches.Patch(facecolor="#ef4444",alpha=.88,label="Increase Risk"),
        ], fontsize=8, frameon=False, loc="upper left")

        plt.tight_layout(pad=1.4)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.warning(f"SHAP 图生成出错（不影响概率）: {e}")

st.markdown("""
<div style="text-align:center;padding:14px 0 6px;color:#94a3b8;font-size:.75rem">
  ⚠️ 本工具仅供科研辅助，不构成临床诊断依据 &nbsp;·&nbsp;
  For research use only, not for clinical diagnosis
</div>""", unsafe_allow_html=True)
