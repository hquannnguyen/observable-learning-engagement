"""Streamlit dashboard for observable learning participation (T35–T38).

The app is a read-only viewer over experiment artifacts. It never trains or
re-implements the pipeline; all aggregation/export comes from ``src.reporting``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.contracts.labels import CANONICAL_TRAINING_LABELS, to_display_label
from src.reporting.aggregation import aggregate_timeline, aggregate_zones
from src.reporting.exporter import export_bundle
from src.reporting.bias_analysis import analyze_positions

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "reports" / "experiments"
OVERLAYS = ROOT / "data" / "poses"

st.set_page_config(
    page_title="CVF25PRJ10 — Observable Participation",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    /* Ensure selectbox trigger & dropdown popover menu never truncate option text */
    div[data-baseweb="popover"] {
        min-width: max-content !important;
        max-width: 560px !important;
    }
    div[data-baseweb="popover"] li,
    ul[data-baseweb="menu"] li,
    div[role="listbox"] li,
    div[role="option"] {
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
        text-overflow: unset !important;
        line-height: 1.35 !important;
        padding-top: 6px !important;
        padding-bottom: 6px !important;
    }
    div[data-baseweb="select"] > div {
        min-height: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def list_experiments() -> List[str]:
    if not EXPERIMENTS.is_dir():
        return []
    return sorted(
        [
            path.name
            for path in EXPERIMENTS.iterdir()
            if path.is_dir() and (path / "metadata.json").is_file()
        ],
        reverse=True,
    )


@st.cache_data(show_spinner=False)
def load_json(path: str) -> Dict[str, Any]:
    file = Path(path)
    if not file.is_file():
        return {}
    return json.loads(file.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    file = Path(path)
    if not file.is_file():
        return pd.DataFrame()
    return pd.read_csv(file)


def list_overlays() -> List[str]:
    blurred: List[str] = []
    if OVERLAYS.is_dir():
        blurred.extend(sorted(str(p) for p in OVERLAYS.glob("*blur*.mp4")))
    inf_root = ROOT / "reports" / "inference"
    if inf_root.is_dir():
        blurred.extend(sorted(str(p) for p in inf_root.glob("*/head_blurred_preview.mp4")))
    other: List[str] = []
    if OVERLAYS.is_dir():
        other.extend(sorted(str(p) for p in OVERLAYS.glob("*overlay*.mp4") if "blur" not in p.name.lower()))
    # Deduplicate while keeping head-blurred previews first
    seen = set()
    ordered: List[str] = []
    for item in blurred + other:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def resolve_model_type(exp_dir: Path, meta: Dict[str, Any]) -> str:
    mtype = meta.get("model_type")
    if mtype and str(mtype).strip() not in {"?", "N/A", ""}:
        return str(mtype).upper()
    name_low = exp_dir.name.lower()
    if "bilstm" in name_low:
        return "BILSTM"
    if "tcn" in name_low:
        return "TCN"
    if "lstm" in name_low:
        return "LSTM"
    return "LSTM"


def format_experiment_short(exp_name: str) -> str:
    """Concise, non-truncating display label for an experiment folder name."""
    exp_dir = EXPERIMENTS / exp_name
    meta = load_json(str(exp_dir / "metadata.json"))
    m_type = resolve_model_type(exp_dir, meta)
    f_schema = str(meta.get("feature_schema_version", "posefeat-v1"))
    dim_tag = "94d" if "v2" in f_schema else "77d"
    parts = exp_name.split("-", 2)
    if len(parts) == 3 and parts[0].isdigit() and len(parts[0]) == 8:
        date_tag = f"{parts[0][6:8]}/{parts[0][4:6]}"
        slug = parts[2]
        for token in ("bilstm-", "lstm-", "tcn-"):
            slug = slug.replace(token, "")
        return f"{m_type} • {slug} ({dim_tag}, {date_tag})"
    return f"{m_type} • {exp_name} ({dim_tag})"


@st.cache_data(show_spinner=False, ttl=15)
def build_known_files_map() -> Dict[str, str]:
    """Index local video and pose CSV files so picking a file in the browser resolves its full Windows path in 0ms without uploading."""
    mapping: Dict[str, str] = {}
    search_roots = [
        ROOT / "data",
        ROOT,
        Path.home() / "Downloads",
        Path.home() / "Desktop",
        Path.home() / "Videos",
        Path.home() / "Documents",
    ]
    valid_exts = {".mp4", ".mov", ".avi", ".mkv", ".csv"}
    for base in search_roots:
        if not base.is_dir():
            continue
        try:
            items = base.rglob("*") if base in (ROOT / "data", ROOT) else base.glob("*")
            for p in items:
                if p.is_file() and p.suffix.lower() in valid_exts and p.name not in mapping:
                    mapping[p.name] = str(p.resolve())
        except OSError:
            continue
    return mapping


def resolve_input_path(raw_text: str) -> Path | None:
    """Resolve either a full path, relative path, or filename picked from the Windows File Explorer dialog."""
    cleaned = raw_text.strip().strip('"').strip("'")
    if not cleaned:
        return None
    cand = Path(cleaned)
    if cand.is_absolute() and cand.is_file():
        return cand.resolve()
    rel = (ROOT / cand).resolve()
    if rel.is_file():
        return rel
    known = build_known_files_map()
    if cand.name in known:
        resolved = Path(known[cand.name])
        if resolved.is_file():
            return resolved
    return None


def render_instant_folder_picker_button(accept_csv: str) -> None:
    """Render a single 40px '📂 Chọn...' button that opens the native Windows Folder dialog with ZERO file upload."""
    known_json = json.dumps(build_known_files_map(), ensure_ascii=False)
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, sans-serif; }}
      body {{ background: transparent; overflow: hidden; }}
      .pick-btn {{
        width: 100%;
        height: 40px;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        background: #FFFFFF;
        color: #0F172A;
        font-size: 0.92rem;
        font-weight: 500;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        transition: background 0.15s, border-color 0.15s;
        white-space: nowrap;
      }}
      .pick-btn:hover {{
        background: #F1F5F9;
        border-color: #2563EB;
        color: #2563EB;
      }}
      .pick-btn:active {{
        background: #E2E8F0;
      }}
    </style>
    </head>
    <body>
      <input type="file" id="fpicker" accept="{accept_csv}" style="display:none" />
      <button type="button" class="pick-btn" id="btnPick" title="Mở cửa sổ Folder Windows để chọn tệp">📂 Chọn...</button>
      <script>
        const knownMap = {known_json};
        const fpicker = document.getElementById('fpicker');
        const btn = document.getElementById('btnPick');
        btn.addEventListener('click', () => {{
          fpicker.value = '';
          fpicker.click();
        }});
        fpicker.addEventListener('change', (e) => {{
          const file = e.target.files && e.target.files[0];
          if (!file) return;
          const resolvedPath = knownMap[file.name] || file.name;
          const pdoc = window.parent.document;
          const inputs = pdoc.querySelectorAll('input[type="text"]');
          for (const inp of inputs) {{
            const al = (inp.getAttribute('aria-label') || '');
            if (al.includes('Đường dẫn tệp') || al.includes('Tệp Video') || al.includes('Tệp Pose')) {{
              const nativeSetter = Object.getOwnPropertyDescriptor(window.parent.HTMLInputElement.prototype, 'value').set;
              nativeSetter.call(inp, resolvedPath);
              inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
              inp.dispatchEvent(new Event('change', {{ bubbles: true }}));
              inp.focus();
              inp.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }}));
              inp.blur();
              break;
            }}
          }}
        }});
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=42, scrolling=False)


def discover_checkpoints(experiments: List[str], selected_exp: str | None = None) -> Dict[str, str]:
    options: Dict[str, str] = {}
    seen_paths = set()
    ordered_exps = ([selected_exp] + [e for e in experiments if e != selected_exp]) if selected_exp else experiments

    for exp_name in ordered_exps:
        if not exp_name:
            continue
        exp_dir = EXPERIMENTS / exp_name
        meta = load_json(str(exp_dir / "metadata.json"))
        ckpt = meta.get("checkpoint_path")
        if ckpt:
            ckpt_path = Path(ckpt)
            if not ckpt_path.is_absolute():
                ckpt_path = ROOT / ckpt_path
            if ckpt_path.is_file():
                resolved = str(ckpt_path.resolve())
                short_lbl = format_experiment_short(exp_name)
                label = f"★ {short_lbl}" if exp_name == selected_exp else short_lbl
                options[label] = str(ckpt_path)
                seen_paths.add(resolved)

    ckpt_dir = ROOT / "models" / "checkpoints"
    if ckpt_dir.is_dir():
        for pt_file in sorted(ckpt_dir.glob("*.pt")):
            resolved = str(pt_file.resolve())
            if resolved not in seen_paths:
                options[f"Local • {pt_file.stem}"] = str(pt_file)
                seen_paths.add(resolved)
    return options


@st.cache_data(show_spinner=False)
def build_experiment_comparison(experiments: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for exp_name in experiments:
        exp_dir = EXPERIMENTS / exp_name
        meta = load_json(str(exp_dir / "metadata.json"))
        rep = load_json(str(exp_dir / "evaluation_report.json"))
        raw = rep.get("raw_metrics", {})
        acc = rep.get("accepted_metrics", {})
        rows.append(
            {
                "experiment_id": meta.get("experiment_id", exp_name),
                "model_type": resolve_model_type(exp_dir, meta),
                "feature_schema": meta.get("feature_schema_version", "?"),
                "metric_scope": meta.get("metric_scope", rep.get("metric_scope", "weak_rule_labels")),
                "split": meta.get("split_version", "split-v0.1"),
                "raw_macro_f1": raw.get("macro_f1"),
                "accepted_macro_f1": acc.get("macro_f1"),
                "coverage": rep.get("coverage"),
                "evaluated_windows": rep.get("evaluated_windows"),
            }
        )

    m0_path = EXPERIMENTS / "m0_heuristic_baseline" / "m0_baseline_results.json"
    if m0_path.is_file():
        m0 = load_json(str(m0_path))
        test_split = m0.get("splits", {}).get("test", {})
        if test_split:
            rows.append(
                {
                    "experiment_id": "m0_heuristic_baseline",
                    "model_type": m0.get("model", "M0-Heuristic"),
                    "feature_schema": "pose_rules",
                    "metric_scope": "weak_rule_labels",
                    "split": "test",
                    "raw_macro_f1": round(float(test_split.get("macro_f1_all", 0.0)), 4),
                    "accepted_macro_f1": round(float(test_split.get("macro_f1_accepted", 0.0)), 4),
                    "coverage": round(float(test_split.get("coverage", 0.0)), 4),
                    "evaluated_windows": test_split.get("total_count"),
                }
            )

    m1_path = EXPERIMENTS / "m1_aggregate_baseline" / "m1_baseline_results.json"
    if m1_path.is_file():
        m1 = load_json(str(m1_path))
        for model_name, splits in m1.get("models", {}).items():
            target_split = "test" if "test" in splits else ("val" if "val" in splits else None)
            if target_split:
                s_data = splits[target_split]
                rows.append(
                    {
                        "experiment_id": f"m1_{model_name}",
                        "model_type": f"M1-{model_name}",
                        "feature_schema": f"agg_{m1.get('feature_dim', '?')}",
                        "metric_scope": "weak_rule_labels",
                        "split": target_split,
                        "raw_macro_f1": round(float(s_data.get("macro_f1", 0.0)), 4),
                        "accepted_macro_f1": None,
                        "coverage": 1.0,
                        "evaluated_windows": None,
                    }
                )
    return pd.DataFrame(rows)


def run_predict_subprocess(cmd: List[str], log_box: Any) -> Tuple[int, str]:
    lines: List[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        log_box.code("\n".join(lines[-15:]), language="text")
    proc.wait()
    return proc.returncode, "\n".join(lines)


def metric_row(label: str, value: Any) -> None:
    st.metric(label=label, value=value)


def main() -> None:
    experiments = list_experiments()
    if not experiments:
        st.error(
            "Không tìm thấy experiment nào trong `reports/experiments/`. "
            "Hãy chạy `python scripts/train.py` và `python scripts/evaluate.py` trước."
        )
        st.stop()

    latest_run = st.session_state.get("latest_inference")
    use_latest_for_charts = False

    with st.sidebar:
        st.markdown("### 🎓 CVF25PRJ10")
        st.caption("Mã đề tài: `CVFP26BCPPRJC012`")
        selected = st.selectbox("Experiment chuẩn", experiments, format_func=format_experiment_short)
        if latest_run and Path(latest_run).is_dir():
            data_source_choice = st.radio(
                "Nguồn dữ liệu Tab 2 & 3",
                ["Lượt suy luận mới", "Experiment chuẩn"],
                horizontal=True,
            )
            use_latest_for_charts = data_source_choice == "Lượt suy luận mới"

    exp_dir = EXPERIMENTS / selected
    metadata = load_json(str(exp_dir / "metadata.json"))
    report = load_json(str(exp_dir / "evaluation_report.json"))
    predictions = load_csv(str(exp_dir / "predictions.csv"))
    zone_predictions = load_csv(str(exp_dir / "zone_predictions.csv"))
    metric_scope = metadata.get("metric_scope", report.get("metric_scope", "weak_rule_labels"))

    resolved_mtype = resolve_model_type(exp_dir, metadata)

    with st.sidebar:
        with st.container(border=True):
            st.caption("THÔNG TIN EXPERIMENT")
            st.markdown(
                f"**ID:** `{selected}`  \n"
                f"**Model:** `{resolved_mtype}`  \n"
                f"**Schema:** `{metadata.get('feature_schema_version', 'N/A')}`  \n"
                f"**Split:** `{metadata.get('split_version', 'N/A')}`  \n"
                f"**Scope:** `{metric_scope}`  \n"
                f"**Accepted F1:** `{report.get('accepted_metrics', {}).get('macro_f1', 'N/A')}`"
            )

    analysis_dir, analysis_scope, analysis_id = exp_dir, metric_scope, metadata.get("experiment_id")
    if use_latest_for_charts and latest_run:
        analysis_dir = Path(latest_run)
        predictions = load_csv(str(analysis_dir / "predictions.csv"))
        zone_predictions = pd.DataFrame()
        analysis_scope = "inference_no_ground_truth"
        inference_meta = load_json(str(analysis_dir / "inference_meta.json"))
        analysis_id = Path(inference_meta.get("checkpoint", "inference")).stem

    st.markdown("## Phân tích mức độ tham gia học tập quan sát được")
    scope_note = (
        " • ⚠️ Metric hiện tại là **weak-label agreement** (chưa có human test set)."
        if metric_scope == "weak_rule_labels"
        else (" • ⚠️ Metric trộn HUMAN và WEAK_RULE." if metric_scope == "mixed_labels" else "")
    )
    st.caption(
        "Chỉ mô tả tín hiệu tư thế/chuyển động quan sát được — không suy diễn cảm xúc, động cơ, ý định hay danh tính."
        + scope_note
    )

    tabs = st.tabs(
        [
            "Chạy suy luận",
            "Diễn biến thời gian",
            "Vùng lớp 3×3 & Báo cáo",
            "Tổng quan Experiment",
            "Preview ẩn danh",
        ]
    )

    with tabs[0]:
        checkpoint_options = discover_checkpoints(experiments, selected_exp=selected)
        in_left, in_right = st.columns([1.15, 1], gap="medium")
        with in_left:
            with st.container(border=True):
                st.markdown("**1. Nguồn dữ liệu đầu vào**")
                default_pilot = ROOT / "data" / "raw" / "session_001" / "cam_A_pilot.mp4"
                default_pose = ROOT / "data" / "poses" / "session_001_cam_A_poses.csv"
                if "local_path_input" not in st.session_state:
                    st.session_state["local_path_input"] = (
                        str(default_pilot.resolve()) if default_pilot.is_file() else ""
                    )

                def _on_input_mode_change() -> None:
                    mode = st.session_state.get("input_mode_radio", "Đường dẫn Local")
                    cur = st.session_state.get("local_path_input", "")
                    if mode == "Tải lên Pose (.csv)" and not cur.lower().endswith(".csv"):
                        if default_pose.is_file():
                            st.session_state["local_path_input"] = str(default_pose.resolve())
                    elif mode == "Tải lên Video" and cur.lower().endswith(".csv"):
                        if default_pilot.is_file():
                            st.session_state["local_path_input"] = str(default_pilot.resolve())

                input_mode = st.radio(
                    "Phương thức nhập",
                    ["Đường dẫn Local", "Tải lên Video", "Tải lên Pose (.csv)"],
                    horizontal=True,
                    key="input_mode_radio",
                    on_change=_on_input_mode_change,
                )
                local_source_path: Path | None = None
                uploaded = None

                if input_mode == "Tải lên Video":
                    allowed_exts = ["mp4", "mov", "avi", "mkv"]
                    box_label = "Tệp Video đã chọn (.mp4, .mov, .avi, .mkv)"
                elif input_mode == "Tải lên Pose (.csv)":
                    allowed_exts = ["csv"]
                    box_label = "Tệp Pose đã chọn (.csv)"
                else:
                    allowed_exts = ["mp4", "mov", "avi", "mkv", "csv"]
                    box_label = "Đường dẫn tệp (Video / Pose .csv)"

                accept_str = ",".join(f".{ext}" for ext in allowed_exts)
                p_col, b_col, f_col = st.columns([2.2, 0.8, 0.95], vertical_alignment="bottom")
                with p_col:
                    raw_local = st.text_input(
                        box_label,
                        key="local_path_input",
                        placeholder=f"VD: {(default_pose if input_mode == 'Tải lên Pose (.csv)' else default_pilot).resolve()}",
                    )
                with b_col:
                    render_instant_folder_picker_button(accept_str)
                with f_col:
                    max_frames = st.number_input(
                        "Số frame (0 = hết)",
                        min_value=0,
                        max_value=90000,
                        value=100,
                        step=50,
                    )
                if raw_local.strip():
                    cand = resolve_input_path(raw_local)
                    valid_set = {f".{ext}" for ext in allowed_exts}
                    if cand is not None and cand.suffix.lower() in valid_set:
                        local_source_path = cand
                    elif cand is None:
                        st.error("Không tìm thấy tệp hợp lệ tại đường dẫn này.")
                    else:
                        st.warning(f"Chế độ '{input_mode}' yêu cầu tệp định dạng: {', '.join(valid_set)}. Hãy bấm '📂 Chọn...' để chọn tệp phù hợp.")

        with in_right:
            with st.container(border=True):
                st.markdown("**2. Cấu hình Mô hình & Cổng bất định**")
                m_col1, m_col2 = st.columns([1.35, 1])
                with m_col1:
                    if checkpoint_options:
                        ckpt_label = st.selectbox(
                            "Mô hình suy luận",
                            list(checkpoint_options),
                            index=0,
                        )
                    else:
                        ckpt_label = ""
                        st.text_input("Mô hình suy luận", value="Chưa có checkpoint (.pt)", disabled=True)
                with m_col2:
                    pose_configs = {
                        "YOLOv8n Pose": "configs/baseline.yaml",
                        "YOLO11m Pose HD": "configs/experiments/pose_yolo11m_hd.yaml",
                    }
                    pose_choice = st.selectbox("Cấu hình Pose", list(pose_configs))

                g_col1, g_col2 = st.columns([1, 1.4], vertical_alignment="center")
                with g_col1:
                    override_probability = st.checkbox("Tùy chỉnh ngưỡng Gate")
                with g_col2:
                    min_prob = st.slider(
                        "min_probability",
                        min_value=0.50,
                        max_value=0.95,
                        value=0.60,
                        step=0.05,
                        disabled=not override_probability,
                        help="Nếu xác suất cao nhất < ngưỡng này, cửa sổ được gắn UNCERTAIN.",
                    )

        has_input = local_source_path is not None or uploaded is not None
        run_clicked = st.button(
            "🚀 Chạy suy luận Pipeline",
            type="primary",
            use_container_width=True,
            disabled=not (has_input and bool(checkpoint_options)),
        )

        if has_input and run_clicked:
            if not checkpoint_options or ckpt_label not in checkpoint_options:
                st.error("Chưa có checkpoint mô hình hợp lệ.")
            else:
                run_id = uuid4().hex
                if uploaded is not None:
                    upload_dir = ROOT / "data" / "raw" / "uploads" / run_id
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    safe_name = Path(uploaded.name).name
                    source_path = upload_dir / safe_name
                    source_path.write_bytes(uploaded.getbuffer())
                else:
                    assert local_source_path is not None
                    source_path = local_source_path
                    safe_name = source_path.name

                out_dir = ROOT / "reports" / "inference" / run_id
                is_csv_input = source_path.suffix.lower() == ".csv"
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "predict_video.py"),
                    "--checkpoint", checkpoint_options[ckpt_label],
                    "--out-dir", str(out_dir),
                    "--config", pose_configs[pose_choice],
                    "--session-id", run_id,
                ]
                if override_probability:
                    cmd.extend(["--min-probability", str(min_prob)])
                if is_csv_input:
                    cmd.extend(["--poses", str(source_path)])
                else:
                    cmd.extend(["--video", str(source_path)])
                    if max_frames > 0:
                        cmd.extend(["--max-frames", str(int(max_frames))])

                with st.expander("Nhật ký xử lý (Live Log)", expanded=True):
                    log_box = st.empty()
                    returncode, full_log = run_predict_subprocess(cmd, log_box)

                if returncode != 0:
                    st.error("Pipeline thất bại. Chi tiết lỗi:")
                    st.code(full_log[-3000:])
                else:
                    load_csv.clear()
                    load_json.clear()
                    st.session_state["latest_inference"] = str(out_dir)
                    st.session_state["latest_input_name"] = safe_name
                    new_preds = load_csv(str(out_dir / "predictions.csv"))
                    inf_meta = load_json(str(out_dir / "inference_meta.json"))
                    total_w = len(new_preds)
                    acc_w = int((new_preds["gated_label"] != "UNCERTAIN").sum()) if total_w else 0
                    unc_w = int((new_preds["gated_label"] == "UNCERTAIN").sum()) if total_w else 0
                    high_w = int((new_preds["gated_label"] == CANONICAL_TRAINING_LABELS[0]).sum()) if total_w else 0
                    mod_w = int((new_preds["gated_label"] == CANONICAL_TRAINING_LABELS[1]).sum()) if total_w else 0
                    low_w = int((new_preds["gated_label"] == CANONICAL_TRAINING_LABELS[2]).sum()) if total_w else 0
                    runs = st.session_state.setdefault("inference_runs", [])
                    runs.append(
                        {
                            "run_id": run_id[:8],
                            "input": safe_name,
                            "checkpoint": Path(checkpoint_options[ckpt_label]).stem,
                            "min_probability": inf_meta.get("min_probability", min_prob if override_probability else "default"),
                            "total_windows": total_w,
                            "accepted": acc_w,
                            "uncertain": unc_w,
                            "coverage_pct": round(100.0 * acc_w / max(total_w, 1), 1),
                            "high": high_w,
                            "moderate": mod_w,
                            "low": low_w,
                            "elapsed_s": inf_meta.get("elapsed_s"),
                        }
                    )
                    st.success(f"Suy luận hoàn tất! Kết quả lưu tại: `{out_dir}`")

        active_out = st.session_state.get("latest_inference")
        if active_out and Path(active_out).is_dir():
            active_dir = Path(active_out)
            new_preds = load_csv(str(active_dir / "predictions.csv"))
            summary_file = active_dir / "export" / "summary.json"
            preview_file = active_dir / "head_blurred_preview.mp4"

            with st.container(border=True):
                h_col1, h_col2 = st.columns([3, 1], vertical_alignment="center")
                with h_col1:
                    st.markdown("**3. Kết quả lượt suy luận mới nhất**")
                with h_col2:
                    if summary_file.is_file():
                        stem = Path(st.session_state.get("latest_input_name", "inference")).stem
                        st.download_button(
                            "📥 Tải summary.json",
                            data=summary_file.read_bytes(),
                            file_name=f"summary_{stem}.json",
                            mime="application/json",
                            use_container_width=True,
                        )

                if not new_preds.empty:
                    c1, c2, c3, c4 = st.columns(4)
                    total_w = len(new_preds)
                    acc_w = int((new_preds["gated_label"] != "UNCERTAIN").sum())
                    unc_w = int((new_preds["gated_label"] == "UNCERTAIN").sum())
                    cov_pct = round(100.0 * acc_w / max(total_w, 1), 1)
                    with c1:
                        metric_row("Tổng số window", total_w)
                    with c2:
                        metric_row("Được chấp nhận (Accepted)", acc_w)
                    with c3:
                        metric_row("Bất định (Uncertain)", unc_w)
                    with c4:
                        metric_row("Độ phủ (Coverage)", f"{cov_pct}%")

                    pos_df, _ = analyze_positions(new_preds)
                    if preview_file.is_file():
                        p_left, p_right = st.columns([1.2, 1], gap="medium")
                        with p_left:
                            st.markdown("**Preview video ẩn danh (head-blur)**")
                            st.video(str(preview_file))
                            st.caption("Làm mờ tự động theo detector; kiểm tra trực quan trước khi chia sẻ.")
                        with p_right:
                            st.markdown("**Thống kê theo vị trí**")
                            st.dataframe(pos_df, width="stretch", hide_index=True)
                    else:
                        st.markdown("**Thống kê theo vị trí**")
                        st.dataframe(pos_df, width="stretch", hide_index=True)

                    st.markdown("**Bảng dự đoán chi tiết & Truy vết cửa sổ**")
                    fcol1, fcol2, fcol3 = st.columns(3)
                    with fcol1:
                        label_choices = ["Tất cả"] + sorted(new_preds["gated_label"].dropna().astype(str).unique().tolist())
                        sel_lbl = st.selectbox("Lọc theo gated_label", label_choices, key="trace_lbl")
                    with fcol2:
                        if "gate_reason" in new_preds.columns:
                            reason_choices = ["Tất cả"] + sorted(new_preds["gate_reason"].dropna().astype(str).unique().tolist())
                            sel_reason = st.selectbox("Lọc theo gate_reason", reason_choices, key="trace_reason")
                        else:
                            sel_reason = "Tất cả"
                    with fcol3:
                        if "track_id" in new_preds.columns:
                            track_choices = ["Tất cả"] + sorted(new_preds["track_id"].dropna().astype(str).unique().tolist())
                            sel_track = st.selectbox("Lọc theo track_id", track_choices, key="trace_track")
                        else:
                            sel_track = "Tất cả"

                    filtered_preds = new_preds
                    if sel_lbl != "Tất cả":
                        filtered_preds = filtered_preds[filtered_preds["gated_label"].astype(str) == sel_lbl]
                    if sel_reason != "Tất cả" and "gate_reason" in filtered_preds.columns:
                        filtered_preds = filtered_preds[filtered_preds["gate_reason"].astype(str) == sel_reason]
                    if sel_track != "Tất cả" and "track_id" in filtered_preds.columns:
                        filtered_preds = filtered_preds[filtered_preds["track_id"].astype(str) == sel_track]

                    st.dataframe(filtered_preds, width="stretch", hide_index=True)

        if st.session_state.get("inference_runs"):
            with st.container(border=True):
                st.markdown("**Đối chiếu các lượt suy luận trong phiên (So sánh ngưỡng Uncertainty)**")
                st.dataframe(pd.DataFrame(st.session_state["inference_runs"]), width="stretch", hide_index=True)

    with tabs[1]:
        if predictions.empty:
            st.info("Chưa có `predictions.csv` cho nguồn dữ liệu đang chọn.")
        else:
            with st.container(border=True):
                timeline_input = predictions
                phases = (
                    sorted(set(predictions["class_phase"].dropna()) & {"lecture", "individual_work", "discussion"})
                    if "class_phase" in predictions
                    else []
                )
                fcol_t1, fcol_t2 = st.columns(2)
                with fcol_t1:
                    if phases:
                        phase = st.selectbox("Pha hoạt động", ["Tất cả"] + phases)
                        if phase != "Tất cả":
                            timeline_input = predictions[predictions["class_phase"].eq(phase)]
                    else:
                        st.selectbox("Pha hoạt động", ["Tất cả (không chia pha)"], disabled=True)
                timeline = aggregate_timeline(timeline_input)
                sessions = ["Tất cả"] + sorted(timeline["session_id"].unique().tolist())
                with fcol_t2:
                    chosen = st.selectbox("Session", sessions)
                view = timeline if chosen == "Tất cả" else timeline[timeline["session_id"] == chosen]
                if view.empty:
                    st.info("Không có dữ liệu timeline.")
                else:
                    long_df = view.melt(
                        id_vars=["session_id", "minute_index"],
                        value_vars=["n_high", "n_moderate", "n_low", "n_uncertain"],
                        var_name="label",
                        value_name="count",
                    )
                    long_df["label"] = long_df["label"].map(
                        {
                            "n_high": "HIGH",
                            "n_moderate": "MODERATE",
                            "n_low": "LOW",
                            "n_uncertain": "UNCERTAIN",
                        }
                    )
                    try:
                        import plotly.express as px

                        fig = px.bar(
                            long_df,
                            x="minute_index",
                            y="count",
                            color="label",
                            facet_col="session_id" if chosen == "Tất cả" else None,
                            labels={"minute_index": "Phút", "count": "Số window"},
                            title="Phân bố cửa sổ quan sát theo phút",
                        )
                        st.plotly_chart(fig, width="stretch")
                    except ImportError:
                        st.dataframe(view, width="stretch", hide_index=True)
                    with st.expander("Xem bảng số liệu chi tiết theo phút"):
                        st.dataframe(view, width="stretch", hide_index=True)

    with tabs[2]:
        zone_df = zone_predictions if not zone_predictions.empty else predictions
        zones = aggregate_zones(zone_df)
        if zones.empty:
            st.info(
                "Chưa có `zone_id` thật trong predictions. Không hiển thị heatmap để "
                "tránh diễn giải sai. Chạy `scripts/assign_zones.py` và build lại dataset."
            )
        else:
            with st.container(border=True):
                st.caption(
                    ("Nguồn zone: `zone_predictions.csv` • " if not zone_predictions.empty else "")
                    + "Zone là gán hình học 3×3 (không phải sơ đồ chỗ ngồi đã xác minh). Mọi bảng phải đọc kèm N."
                )
                z_left, z_right = st.columns([1, 1.15], gap="medium")
                with z_left:
                    metric = st.selectbox(
                        "Chỉ số hiển thị trên Heatmap",
                        ["n_windows", "pct_high", "pct_moderate", "pct_low", "pct_uncertain"],
                        index=0,
                    )
                    rows = ["BACK", "MIDDLE", "FRONT"]
                    cols = ["LEFT", "CENTER", "RIGHT"]
                    matrix = pd.DataFrame(index=rows, columns=cols, dtype=float)
                    for _, row in zones.iterrows():
                        parts = str(row["zone_id"]).split("-")
                        if len(parts) == 2 and parts[0] in rows and parts[1] in cols:
                            matrix.loc[parts[0], parts[1]] = float(row[metric])
                    try:
                        import plotly.express as px

                        fig = px.imshow(
                            matrix.astype(float),
                            text_auto=True,
                            aspect="auto",
                            labels={"x": "Cột", "y": "Hàng", "color": metric},
                            title=f"{metric} theo vùng 3×3 (N nhỏ → chỉ tham khảo)",
                        )
                        st.plotly_chart(fig, width="stretch")
                    except ImportError:
                        st.dataframe(matrix, width="stretch")
                with z_right:
                    st.markdown("**Tổng hợp theo vùng 3×3**")
                    st.dataframe(zones, width="stretch", hide_index=True)
                    if zones["n_windows"].min() < 20:
                        st.warning("Có vùng N < 20; không kết luận chênh lệch theo vị trí.")

                positional, evidence = analyze_positions(zone_df)
                with st.expander("Chất lượng & bằng chứng đánh giá theo vị trí"):
                    if evidence["status"] == "HUMAN_REVIEW_REQUIRED":
                        st.info("Chưa có nhãn người để đánh giá độ chính xác theo vùng. Điểm weak agreement chỉ phản ánh mức khớp với quy tắc.")
                    st.dataframe(positional, width="stretch", hide_index=True)

        with st.container(border=True):
            st.markdown("**Xuất gói báo cáo ẩn danh (Standardized Export Bundle)**")
            if predictions.empty:
                st.info("Chưa có predictions để export.")
            else:
                ex_c1, ex_c2 = st.columns([1, 2], vertical_alignment="bottom")
                with ex_c1:
                    bucket = st.number_input("Khoảng thời gian tổng hợp (giây)", min_value=10, max_value=600, value=60, step=10)
                with ex_c2:
                    export_clicked = st.button(
                        "📦 Tạo gói báo cáo ẩn danh (5 tệp chuẩn)",
                        type="primary",
                        use_container_width=True,
                    )
                if export_clicked:
                    out_dir = analysis_dir / "export_ui"
                    paths = export_bundle(
                        predictions,
                        out_dir,
                        experiment_id=analysis_id,
                        metric_scope=analysis_scope,
                        bucket_seconds=int(bucket),
                    )
                    st.success(f"Đã xuất {len(paths)} tệp vào `{out_dir}`.")
                    dl_cols = st.columns(len(paths))
                    for col_btn, (name, path) in zip(dl_cols, paths.items()):
                        file = Path(path)
                        with col_btn:
                            st.download_button(
                                label=f"📥 {file.name}",
                                data=file.read_bytes(),
                                file_name=file.name,
                                mime="text/csv" if file.suffix == ".csv" else ("text/markdown" if file.suffix == ".md" else "application/json"),
                                key=f"dl-{name}",
                                use_container_width=True,
                            )

    with tabs[3]:
        col1, col2, col3, col4 = st.columns(4)
        accepted = report.get("accepted_metrics", {})
        with col1:
            metric_row("Raw Macro-F1", report.get("raw_metrics", {}).get("macro_f1", "N/A"))
        with col2:
            metric_row("Accepted Macro-F1", accepted.get("macro_f1", "N/A"))
        with col3:
            metric_row("Coverage", report.get("coverage", "N/A"))
        with col4:
            metric_row("Evaluated windows", report.get("evaluated_windows", "N/A"))

        meta_c1, meta_c2 = st.columns(2, gap="medium")
        with meta_c1:
            with st.container(border=True):
                st.markdown("**Cấu hình & Phạm vi Experiment**")
                meta_rows = [
                    {"Thuộc tính": "Experiment ID", "Giá trị": str(metadata.get("experiment_id", selected))},
                    {"Thuộc tính": "Model / Feature Schema", "Giá trị": f"{resolved_mtype} / {metadata.get('feature_schema_version', 'N/A')}"},
                    {"Thuộc tính": "Split / Seed", "Giá trị": f"{metadata.get('split_version', 'N/A')} (seed={metadata.get('seed', 'N/A')})"},
                    {"Thuộc tính": "Metric Scope", "Giá trị": str(metric_scope)},
                    {"Thuộc tính": "Test Windows (Total / Unlabeled)", "Giá trị": f"{report.get('total_test_windows', 'N/A')} / {report.get('excluded_unlabeled_windows', 'N/A')}"},
                    {"Thuộc tính": "Git Commit", "Giá trị": str(metadata.get("git_commit", "N/A"))[:12]},
                ]
                st.dataframe(pd.DataFrame(meta_rows), width="stretch", hide_index=True)
        with meta_c2:
            with st.container(border=True):
                st.markdown("**Ngưỡng Uncertainty Gate & Lý do lọc**")
                thresh = report.get("thresholds", {})
                gate_counts = report.get("gate_reason_counts", {})
                t_rows = [
                    {"Tham số / Lý do Gate": f"Ngưỡng: {k}", "Giá trị / Số cửa sổ": str(v)}
                    for k, v in thresh.items()
                ] + [
                    {"Tham số / Lý do Gate": f"Gate: {k}", "Giá trị / Số cửa sổ": str(v)}
                    for k, v in gate_counts.items()
                ]
                if t_rows:
                    st.dataframe(pd.DataFrame(t_rows), width="stretch", hide_index=True)
                else:
                    st.caption("Không có thông tin ngưỡng/gate_reason trong report.")

        with st.container(border=True):
            eval_view = st.radio(
                "Chế độ xem chỉ số phân loại",
                ["Accepted (sau Uncertainty Gate)", "Raw (trước Gate)"],
                horizontal=True,
            )
            metrics_block = (
                report.get("accepted_metrics", {})
                if "Accepted" in eval_view
                else report.get("raw_metrics", {})
            )
            per_class = metrics_block.get("per_class", {})
            cm = metrics_block.get("confusion_matrix")
            if per_class or cm:
                mcol1, mcol2 = st.columns([1.15, 1], gap="medium")
                with mcol1:
                    if per_class:
                        st.markdown("**Per-class Precision / Recall / F1**")
                        pc_rows = []
                        for cname in CANONICAL_TRAINING_LABELS:
                            cinfo = per_class.get(cname, {})
                            pc_rows.append(
                                {
                                    "Lớp": to_display_label(cname),
                                    "Precision": cinfo.get("precision"),
                                    "Recall": cinfo.get("recall"),
                                    "F1": cinfo.get("f1"),
                                    "Support": cinfo.get("support"),
                                }
                            )
                        st.dataframe(pd.DataFrame(pc_rows), width="stretch", hide_index=True)
                with mcol2:
                    if cm:
                        st.markdown("**Ma trận nhầm lẫn (Confusion Matrix)**")
                        labels_short = [to_display_label(c) for c in CANONICAL_TRAINING_LABELS]
                        cm_df = pd.DataFrame(cm, index=labels_short, columns=labels_short)
                        try:
                            import plotly.express as px

                            fig_cm = px.imshow(
                                cm_df,
                                text_auto=True,
                                aspect="auto",
                                labels={"x": "Dự đoán (Pred)", "y": "Thực tế (True)", "color": "Số mẫu"},
                            )
                            st.plotly_chart(fig_cm, width="stretch")
                        except ImportError:
                            st.dataframe(cm_df, width="stretch")

        history_df = load_csv(str(exp_dir / "training_history.csv"))
        if not history_df.empty and "epoch" in history_df.columns:
            with st.container(border=True):
                st.markdown("**Đường cong huấn luyện (`training_history.csv`)**")
                try:
                    import plotly.express as px

                    hcol1, hcol2 = st.columns(2)
                    with hcol1:
                        loss_cols = [c for c in ("train_loss", "val_loss") if c in history_df.columns]
                        if loss_cols:
                            fig_loss = px.line(history_df, x="epoch", y=loss_cols, title="Train / Val Loss")
                            st.plotly_chart(fig_loss, width="stretch")
                    with hcol2:
                        f1_cols = [c for c in ("val_macro_f1", "val_acc") if c in history_df.columns]
                        if f1_cols:
                            fig_f1 = px.line(history_df, x="epoch", y=f1_cols, title="Validation Macro-F1 / Acc")
                            st.plotly_chart(fig_f1, width="stretch")
                except ImportError:
                    st.dataframe(history_df, width="stretch", hide_index=True)

        comp_df = build_experiment_comparison(experiments)
        if not comp_df.empty:
            with st.container(border=True):
                st.markdown("**Bảng đối chiếu các Experiment & Baseline (M0 / M1 / LSTM / TCN)**")
                st.caption("Đọc cùng cột `metric_scope` và `feature_schema` trước khi kết luận.")
                st.dataframe(comp_df, width="stretch", hide_index=True)

        exp_preds = load_csv(str(exp_dir / "predictions.csv"))
        if not exp_preds.empty:
            with st.expander("Truy vết dự đoán của Experiment (`predictions.csv`)"):
                st.caption(f"Checkpoint: `{metadata.get('checkpoint_path', 'N/A')}` • Hiển thị tối đa 300 dòng khớp bộ lọc.")
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    sess_opts = ["Tất cả"] + sorted(exp_preds["session_id"].dropna().astype(str).unique().tolist()) if "session_id" in exp_preds.columns else ["Tất cả"]
                    sel_esess = st.selectbox("Session", sess_opts, key="exp_trace_sess")
                with ec2:
                    elbl_opts = ["Tất cả"] + sorted(exp_preds["gated_label"].dropna().astype(str).unique().tolist()) if "gated_label" in exp_preds.columns else ["Tất cả"]
                    sel_elbl = st.selectbox("gated_label", elbl_opts, key="exp_trace_lbl")
                with ec3:
                    ereason_opts = ["Tất cả"] + sorted(exp_preds["gate_reason"].dropna().astype(str).unique().tolist()) if "gate_reason" in exp_preds.columns else ["Tất cả"]
                    sel_ereason = st.selectbox("gate_reason", ereason_opts, key="exp_trace_reason")

                ep_view = exp_preds
                if sel_esess != "Tất cả" and "session_id" in ep_view.columns:
                    ep_view = ep_view[ep_view["session_id"].astype(str) == sel_esess]
                if sel_elbl != "Tất cả" and "gated_label" in ep_view.columns:
                    ep_view = ep_view[ep_view["gated_label"].astype(str) == sel_elbl]
                if sel_ereason != "Tất cả" and "gate_reason" in ep_view.columns:
                    ep_view = ep_view[ep_view["gate_reason"].astype(str) == sel_ereason]
                st.dataframe(ep_view.head(300), width="stretch", hide_index=True)

        with st.expander("Hạn chế & quy tắc diễn giải kết quả"):
            st.markdown(
                """
                - Đây là **observable participation**, không phải cảm xúc, động cơ hay ý định nội tại.
                - `UNCERTAIN` là trạng thái thiếu căn cứ của hệ thống, không phải lớp hành vi thứ tư.
                - Không kết luận chênh lệch theo vùng lớp khi `N` quá nhỏ hoặc mức che khuất khác nhau.
                - `head_orientation_proxy` chỉ là xấp xỉ hình học 2D, không phải gaze.
                """
            )

    with tabs[4]:
        overlays = list_overlays()
        if not overlays:
            st.info("Không tìm thấy video overlay trong `data/poses/` hoặc `reports/inference/`.")
        else:
            def _format_overlay(value: str) -> str:
                p = Path(value)
                tag = " [ĐÃ LÀM MỜ ĐẦU]" if "blur" in p.name.lower() else " [CHƯA HEAD-BLUR — CHỈ DÙNG NỘI BỘ]"
                return f"{p.parent.name}/{p.name}{tag}"

            with st.container(border=True):
                choice = st.selectbox("Chọn video preview", overlays, format_func=_format_overlay)
                is_blurred = "blur" in Path(choice).name.lower()
                v_col1, v_col2 = st.columns([1.6, 1], gap="medium")
                with v_col1:
                    st.video(choice)
                with v_col2:
                    st.markdown("**Trạng thái Quyền riêng tư**")
                    if is_blurred:
                        st.success("Đã bật làm mờ vùng đầu (Head-Blur) cho người được phát hiện.")
                    else:
                        st.warning("Video này chỉ có khung xương overlay, chưa bật head-blur. Chỉ dùng kiểm tra nội bộ.")
                    st.caption(
                        f"- Tệp: `{Path(choice).name}`\n"
                        f"- Thư mục: `{Path(choice).parent.name}`\n"
                        "- Kiểm tra trực quan các trường hợp detector bỏ sót trước khi trình chiếu công khai."
                    )


if __name__ == "__main__":
    main()
