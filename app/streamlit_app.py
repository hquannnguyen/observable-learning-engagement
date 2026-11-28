"""Streamlit dashboard for observable learning participation (T35–T38).

The app is a read-only viewer over experiment artifacts. It never trains or
re-implements the pipeline; all aggregation/export comes from ``src.reporting``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import pandas as pd
import streamlit as st

from src.reporting.aggregation import aggregate_timeline, aggregate_zones
from src.reporting.exporter import export_bundle
from src.reporting.bias_analysis import analyze_positions

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "reports" / "experiments"
OVERLAYS = ROOT / "data" / "poses"
DEMO_ROOT = ROOT / "reports" / "demo_test"
TEST_SESSIONS_INFO = {
    "session_005 (Test · 900s · 218 GT windows)": ("session_005", 900),
    "session_006 (Test · 250s · 55 GT windows)": ("session_006", 250),
}
SHORT_LABEL_MAP = {
    "HIGH_OBSERVED_PARTICIPATION": "HIGH",
    "MODERATE_OBSERVED_PARTICIPATION": "MODERATE",
    "LOW_OBSERVED_PARTICIPATION": "LOW",
    "UNCERTAIN": "UNCERTAIN",
}

st.set_page_config(
    page_title="CVF25PRJ10 — Observable Participation",
    page_icon="🎓",
    layout="wide",
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
    if not OVERLAYS.is_dir():
        return []
    return sorted([str(p) for p in OVERLAYS.glob("*blur*.mp4")] +
                  [str(p) for p in (ROOT / "reports" / "inference").glob("*/head_blurred_preview.mp4")])


def metric_row(label: str, value: Any) -> None:
    st.metric(label=label, value=value)


@st.cache_data(show_spinner=False)
def list_demo_runs() -> List[str]:
    if not DEMO_ROOT.is_dir():
        return []
    return sorted(
        [
            p.name
            for p in DEMO_ROOT.iterdir()
            if p.is_dir() and (p / "demo_summary.json").is_file()
        ],
        reverse=True,
    )


def _render_test_video_demo_tab(experiments: List[str]) -> None:
    st.subheader("🎬 Demo Nhận Diện Trên Video Tập Test (Tái Sử Dụng Pose + TCN/LSTM v1)")
    st.caption(
        "Chọn 1 video thuộc tập Test đóng băng (`session_005` hoặc `session_006`), chạy mô hình đã train trên `Dataset v1` "
        "và xem trực tiếp video ẩn danh có khung màu dự đoán (`HIGH / MODERATE / LOW / UNCERTAIN`) kèm đối chiếu nhãn chuẩn."
    )

    ckpt_options: Dict[str, str] = {}
    default_v1 = ROOT / "models" / "checkpoints" / "20261126-110817-tcn-adjudicated-v03-600.pt"
    if default_v1.is_file():
        ckpt_options["⭐ 20261126-110817-tcn-adjudicated-v03-600 (TCN v1 · Tốt nhất)"] = str(default_v1)
    for exp_name in experiments:
        meta = load_json(str(EXPERIMENTS / exp_name / "metadata.json"))
        ckpt = meta.get("checkpoint_path")
        if ckpt and Path(ckpt).is_file() and str(Path(ckpt)) != str(default_v1):
            ckpt_options[f"{meta.get('experiment_id', exp_name)} ({meta.get('model_type', '?')})"] = ckpt

    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1.2, 1.4, 1.0])
    with c_ctrl1:
        sess_choice = st.selectbox("Video tập Test", list(TEST_SESSIONS_INFO.keys()), key="demo_sess")
        sess_id, max_sec = TEST_SESSIONS_INFO[sess_choice]
        blur_face = st.checkbox("Làm mờ đầu/mặt ẩn danh (khuyên bật)", value=True, key="demo_blur")
    with c_ctrl2:
        ckpt_keys = list(ckpt_options.keys()) or ["Chưa có checkpoint"]
        ckpt_choice = st.selectbox("Mô hình nhận diện", ckpt_keys, key="demo_ckpt")
        use_custom_prob = st.checkbox("Tùy chỉnh ngưỡng UNCERTAIN (min_probability)", value=False, key="demo_custom_prob")
        custom_prob = st.slider(
            "Ngưỡng min_probability",
            0.40,
            0.90,
            0.60,
            0.05,
            disabled=not use_custom_prob,
            key="demo_prob_slider",
        )
    with c_ctrl3:
        default_start = 60 if max_sec >= 120 else 0
        start_s = st.number_input("Bắt đầu từ giây", min_value=0, max_value=max(0, max_sec - 10), value=default_start, step=10, key="demo_start")
        duration_s = st.slider("Độ dài đoạn demo (giây)", min_value=10, max_value=120, value=40, step=10, key="demo_dur")

    run_btn = st.button("🚀 Chạy Nhận Diện & Render Video Demo", type="primary", key="demo_run_btn")
    target_dir = DEMO_ROOT / f"{sess_id}_{int(start_s)}s_{int(duration_s)}s"

    if run_btn:
        if ckpt_choice not in ckpt_options:
            st.error("Không tìm thấy checkpoint mô hình hợp lệ.")
        else:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "demo_test_video.py"),
                "--session", sess_id,
                "--checkpoint", ckpt_options[ckpt_choice],
                "--start-s", str(int(start_s)),
                "--duration-s", str(int(duration_s)),
                "--out-dir", str(target_dir),
            ]
            if not blur_face:
                cmd.append("--no-blur")
            if use_custom_prob:
                cmd.extend(["--min-probability", str(custom_prob)])
            with st.spinner(f"Đang suy luận và xuất video H.264 cho {sess_id} ({int(start_s)}–{int(start_s + duration_s)}s)..."):
                proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
            if proc.returncode != 0:
                st.error("Không chạy được demo. Chi tiết lỗi:")
                st.code(proc.stderr[-2500:] or proc.stdout[-2500:])
            else:
                list_demo_runs.clear()
                load_json.clear()
                load_csv.clear()
                st.session_state["active_demo_dir"] = str(target_dir)
                st.success(f"Hoàn tất demo tại `{target_dir}`!")

    existing_runs = list_demo_runs()
    active_demo = st.session_state.get("active_demo_dir")
    if (not active_demo or not Path(active_demo).is_dir()) and existing_runs:
        preferred = f"{sess_id}_{int(start_s)}s_{int(duration_s)}s"
        chosen_run = preferred if preferred in existing_runs else existing_runs[0]
        active_demo = str(DEMO_ROOT / chosen_run)

    if not existing_runs and not active_demo:
        st.info("Chưa có video demo nào được render sẵn. Bấm nút **🚀 Chạy Nhận Diện & Render Video Demo** ở trên (chỉ mất ~10–20 giây).")
        return

    if existing_runs:
        run_labels = existing_runs
        cur_name = Path(active_demo).name if active_demo else run_labels[0]
        idx_def = run_labels.index(cur_name) if cur_name in run_labels else 0
        picked_name = st.selectbox("Kết quả demo đã sẵn sàng để trình chiếu:", run_labels, index=idx_def, key="demo_picker")
        demo_dir = DEMO_ROOT / picked_name
        st.session_state["active_demo_dir"] = str(demo_dir)
    else:
        demo_dir = Path(active_demo)

    summary = load_json(str(demo_dir / "demo_summary.json"))
    preds = load_csv(str(demo_dir / "predictions.csv"))
    if not summary or preds.empty:
        st.warning("Thiếu dữ liệu `demo_summary.json` hoặc `predictions.csv` trong thư mục demo.")
        return

    st.divider()
    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    lbl_counts = summary.get("label_counts", {})
    n_high = lbl_counts.get("HIGH_OBSERVED_PARTICIPATION", 0)
    n_mod = lbl_counts.get("MODERATE_OBSERVED_PARTICIPATION", 0)
    n_low = lbl_counts.get("LOW_OBSERVED_PARTICIPATION", 0)
    gt_eval = summary.get("ground_truth_eval", {})
    with k1:
        metric_row("Tổng cửa sổ (4s)", summary.get("total_windows", len(preds)))
    with k2:
        metric_row("Accepted / Coverage", f"{summary.get('accepted_windows', 0)} ({summary.get('coverage', 0)*100:.1f}%)")
    with k3:
        metric_row("HIGH / MOD / LOW", f"{n_high} / {n_mod} / {n_low}")
    with k4:
        if gt_eval.get("matched_gt_windows", 0) > 0:
            acc_f1 = gt_eval.get("gt_accepted_macro_f1")
            acc_str = f"{acc_f1:.3f}" if acc_f1 is not None else "N/A"
            metric_row(f"GT Match ({gt_eval['matched_gt_windows']} mẫu)", f"F1: {acc_str}")
        else:
            metric_row("Thời gian xử lý", f"{summary.get('elapsed_s', '?')}s")

    col_vid, col_chart = st.columns([1.35, 1.0])
    with col_vid:
        st.markdown("**Video nhận diện (🟢 HIGH · 🟡 MODERATE · 🔴 LOW · ⚪ UNCERTAIN)**")
        video_file = demo_dir / "demo_overlay.mp4"
        if video_file.is_file():
            st.video(str(video_file))
        else:
            st.info("Bản này chưa render video MP4 (`--skip-video`).")

    with col_chart:
        st.markdown("**Diễn biến mức độ tham gia theo thời gian**")
        plot_df = preds.copy()
        plot_df["time_bucket_s"] = ((plot_df["t_start_ms"].astype(float) / 1000.0) // 10 * 10).astype(int)
        plot_df["Nhãn"] = plot_df["gated_label"].map(lambda x: SHORT_LABEL_MAP.get(str(x), str(x)))
        agg = plot_df.groupby(["time_bucket_s", "Nhãn"]).size().reset_index(name="Số cửa sổ")
        try:
            import plotly.express as px

            fig = px.bar(
                agg,
                x="time_bucket_s",
                y="Số cửa sổ",
                color="Nhãn",
                color_discrete_map={
                    "HIGH": "#28a745",
                    "MODERATE": "#ffc107",
                    "LOW": "#dc3545",
                    "UNCERTAIN": "#6c757d",
                },
                labels={"time_bucket_s": "Mốc thời gian (giây)"},
                title=f"{summary.get('session_id')} ({int(summary.get('start_s', 0))}–{int(summary.get('start_s', 0) + summary.get('duration_s', 0))}s)",
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")
        except ImportError:
            st.dataframe(agg, width="stretch")

        if gt_eval.get("matched_gt_windows", 0) > 0:
            st.info(
                f"Đối chiếu Ground Truth `Dataset v1` trên **{gt_eval['matched_gt_windows']}** cửa sổ trùng khớp trong đoạn này: "
                f"Raw Acc = **{gt_eval.get('gt_raw_accuracy', 0)*100:.1f}%** (Macro-F1 `{gt_eval.get('gt_raw_macro_f1')}`), "
                f"Accepted Acc = **{(gt_eval.get('gt_accepted_accuracy') or 0)*100:.1f}%** (Macro-F1 `{gt_eval.get('gt_accepted_macro_f1')}`)."
            )

    st.markdown("**Tra cứu chi tiết từng Track / Cửa sổ 4 giây**")
    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        tracks = ["Tất cả"] + sorted([str(t) for t in preds["track_id"].dropna().unique()], key=lambda x: int(float(x)))
        sel_trk = st.selectbox("Lọc theo Track ID", tracks, key="demo_trk_filter")
    with f2:
        sel_lbl = st.selectbox("Lọc theo Nhãn dự đoán", ["Tất cả", "HIGH", "MODERATE", "LOW", "UNCERTAIN"], key="demo_lbl_filter")
    with f3:
        only_gt = st.checkbox("Chỉ hiện cửa sổ có Ground Truth v1", value=False, key="demo_only_gt")

    view = preds.copy()
    view["Thời gian"] = view.apply(lambda r: f"{float(r['t_start_ms'])/1000:.0f}s–{float(r['t_end_ms'])/1000:.0f}s", axis=1)
    view["Dự đoán"] = view["gated_label"].map(lambda x: SHORT_LABEL_MAP.get(str(x), str(x)))
    view["Nhãn chuẩn v1"] = view["true_label"].map(lambda x: SHORT_LABEL_MAP.get(str(x), "—") if pd.notna(x) and str(x).strip() else "—")
    if sel_trk != "Tất cả":
        view = view[view["track_id"].astype(str).eq(sel_trk)]
    if sel_lbl != "Tất cả":
        view = view[view["Dự đoán"].eq(sel_lbl)]
    if only_gt:
        view = view[view["Nhãn chuẩn v1"].ne("—")]

    show_cols = [c for c in ["window_id", "track_id", "Thời gian", "zone_id", "Dự đoán", "p_max", "gate_reason", "Nhãn chuẩn v1", "is_correct"] if c in view.columns]
    st.dataframe(view[show_cols], width="stretch", height=280)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "📥 Tải bảng dự đoán CSV",
            data=(demo_dir / "predictions.csv").read_bytes(),
            file_name=f"{demo_dir.name}_predictions.csv",
            mime="text/csv",
            key="dl_demo_preds",
        )
    with dl2:
        st.download_button(
            "📥 Tải demo_summary.json",
            data=(demo_dir / "demo_summary.json").read_bytes(),
            file_name=f"{demo_dir.name}_summary.json",
            mime="application/json",
            key="dl_demo_summary",
        )


def main() -> None:
    st.title("Phân tích mức độ tham gia học tập quan sát được")
    st.caption(
        "Chỉ mô tả tín hiệu tư thế/chuyển động quan sát được. "
        "Không suy diễn cảm xúc, động cơ, ý định hoặc danh tính."
    )

    experiments = list_experiments()
    if not experiments:
        st.error(
            "Không tìm thấy experiment nào trong `reports/experiments/`. "
            "Hãy chạy `python scripts/train.py` và `python scripts/evaluate.py` trước."
        )
        st.stop()

    selected = st.sidebar.selectbox("Experiment", experiments)
    exp_dir = EXPERIMENTS / selected
    metadata = load_json(str(exp_dir / "metadata.json"))
    report = load_json(str(exp_dir / "evaluation_report.json"))
    predictions = load_csv(str(exp_dir / "predictions.csv"))
    zone_predictions = load_csv(str(exp_dir / "zone_predictions.csv"))

    metric_scope = metadata.get("metric_scope", report.get("metric_scope", "weak_rule_labels"))
    if metric_scope == "weak_rule_labels":
        st.warning(
            "Các metric dưới đây là **weak-label agreement**, không phải độ chính xác "
            "trên hành vi người thật. Chưa có human test set."
        )
    elif metric_scope == "mixed_labels":
        st.warning("Metric trộn nhãn HUMAN và WEAK_RULE; không dùng để kết luận tuyệt đối.")

    tabs = st.tabs(
        [
            "🎬 Demo Video Tập Test",
            "Chạy suy luận",
            "Diễn biến theo thời gian",
            "Bản đồ vùng lớp & Xuất báo cáo",
            "Tổng quan experiment",
            "Preview ẩn danh",
        ]
    )

    with tabs[0]:
        _render_test_video_demo_tab(experiments)

    with tabs[1]:
        st.subheader("Chạy suy luận mô hình (Video / File Pose)")
        st.caption(
            "Chạy toàn bộ pipeline từ đầu vào (video hoặc file poses.csv) qua mô hình chuỗi thời gian "
            "và cổng bất định. Video upload được xử lý cục bộ và không commit."
        )
        checkpoint_options = {}
        for exp_name in experiments:
            meta = load_json(str(EXPERIMENTS / exp_name / "metadata.json"))
            ckpt = meta.get("checkpoint_path")
            if ckpt and Path(ckpt).is_file():
                label = f"{meta.get('experiment_id', exp_name)} ({meta.get('model_type', '?')})"
                checkpoint_options[label] = ckpt

        if not checkpoint_options:
            default_ckpt = ROOT / "models" / "checkpoints" / "lstm_baseline.pt"
            if default_ckpt.is_file():
                checkpoint_options["LSTM Baseline (default)"] = str(default_ckpt)

        col_in1, col_in2 = st.columns([1, 1])
        with col_in1:
            input_mode = st.radio("Loại dữ liệu đầu vào", ["Tải lên Video (.mp4, .mov, .avi, .mkv)", "Tải lên File Pose (.csv)"])
            ckpt_label = st.selectbox("Mô hình kiểm định (Checkpoint)", sorted(checkpoint_options) if checkpoint_options else ["Không tìm thấy checkpoint"])
        with col_in2:
            override_probability = st.checkbox("Chỉnh ngưỡng tin cậy")
            min_prob = st.slider(
                "Ngưỡng tin cậy cổng bất định (min_probability)",
                min_value=0.50,
                max_value=0.95,
                value=0.60,
                step=0.05,
                disabled=not override_probability,
                help="Nếu xác suất lớp cao nhất thấp hơn ngưỡng này, mẫu được phân loại là UNCERTAIN.",
            )
            pose_configs = {
                "YOLOv8n Pose": "configs/baseline.yaml",
                "YOLO11m Pose HD (1280px)": "configs/experiments/pose_yolo11m_hd.yaml",
            }
            pose_choice = st.selectbox("Cấu hình trích xuất tư thế", list(pose_configs))

        max_frames = st.number_input(
            "Giới hạn số frame sampled (để demo nhanh, 0 = toàn bộ video)",
            min_value=0,
            max_value=90000,
            value=150,
            step=50,
        )

        uploaded = st.file_uploader(
            "Chọn tệp đầu vào",
            type=["mp4", "mov", "avi", "mkv"] if "Video" in input_mode else ["csv"],
        )

        if uploaded is not None and st.button("🚀 Bắt đầu chạy suy luận", type="primary"):
            if not checkpoint_options or ckpt_label not in checkpoint_options:
                st.error("Chưa có checkpoint mô hình hợp lệ.")
            else:
                run_id = uuid4().hex
                upload_dir = ROOT / "data" / "raw" / "uploads" / run_id
                upload_dir.mkdir(parents=True, exist_ok=True)
                safe_name = Path(uploaded.name).name
                saved_path = upload_dir / safe_name
                saved_path.write_bytes(uploaded.getbuffer())
                out_dir = ROOT / "reports" / "inference" / run_id

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
                if "Video" in input_mode:
                    cmd.extend(["--video", str(saved_path)])
                    if max_frames > 0:
                        cmd.extend(["--max-frames", str(int(max_frames))])
                else:
                    cmd.extend(["--poses", str(saved_path)])

                with st.spinner("Đang chạy pipeline suy luận..."):
                    proc = subprocess.run(
                        cmd, cwd=str(ROOT), capture_output=True, text=True
                    )
                if proc.returncode != 0:
                    st.error("Pipeline thất bại. Chi tiết lỗi:")
                    st.code(proc.stderr[-3000:] or proc.stdout[-3000:])
                else:
                    load_csv.clear()
                    load_json.clear()
                    st.session_state["latest_inference"] = str(out_dir)
                    st.success(f"Suy luận hoàn tất! Kết quả lưu tại: `{out_dir}`")
                    new_preds = load_csv(str(out_dir / "predictions.csv"))
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

                        st.subheader("Bảng dự đoán chi tiết")
                        st.dataframe(new_preds, width="stretch")

                        pos_df, _ = analyze_positions(new_preds)
                        st.subheader("Thống kê theo vị trí")
                        st.dataframe(pos_df, width="stretch")

                    preview_file = out_dir / "head_blurred_preview.mp4"
                    if preview_file.is_file():
                        st.subheader("Preview video ẩn danh (làm mờ mặt/đầu)")
                        st.video(str(preview_file))
                        st.caption("Làm mờ tự động theo detector; kiểm tra các trường hợp detector bỏ sót trước khi chia sẻ.")

                    summary_file = out_dir / "export" / "summary.json"
                    if summary_file.is_file():
                        st.download_button(
                            "📥 Tải summary.json",
                            data=summary_file.read_bytes(),
                            file_name=f"summary_{Path(uploaded.name).stem}.json",
                            mime="application/json",
                        )

    latest_run = st.session_state.get("latest_inference")
    analysis_dir, analysis_scope, analysis_id = exp_dir, metric_scope, metadata.get("experiment_id")
    if latest_run and st.sidebar.radio("Dữ liệu cho biểu đồ và báo cáo", ["Lượt suy luận mới", "Experiment"]) == "Lượt suy luận mới":
        analysis_dir = Path(latest_run)
        predictions = load_csv(str(analysis_dir / "predictions.csv"))
        zone_predictions = pd.DataFrame()
        analysis_scope = "inference_no_ground_truth"
        inference_meta = load_json(str(analysis_dir / "inference_meta.json"))
        analysis_id = Path(inference_meta["checkpoint"]).stem

    with tabs[2]:
        st.subheader("Phân phối theo thời gian")
        if predictions.empty:
            st.info("Experiment này chưa có `predictions.csv`.")
        else:
            timeline_input = predictions
            if "class_phase" in predictions:
                phases = sorted(set(predictions["class_phase"].dropna()) & {"lecture", "individual_work", "discussion"})
                if phases:
                    phase = st.selectbox("Pha hoạt động", ["Tất cả"] + phases)
                    if phase != "Tất cả":
                        timeline_input = predictions[predictions["class_phase"].eq(phase)]
            timeline = aggregate_timeline(timeline_input)
            sessions = ["Tất cả"] + sorted(timeline["session_id"].unique().tolist())
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
                        title="Số cửa sổ theo phút",
                    )
                    st.plotly_chart(fig, width="stretch")
                except ImportError:
                    st.dataframe(view, width="stretch")
                st.dataframe(view, width="stretch")

    with tabs[3]:
        st.subheader("Phân bố theo vùng lớp (3×3) & Xuất báo cáo ẩn danh")
        zone_df = zone_predictions if not zone_predictions.empty else predictions
        zones = aggregate_zones(zone_df)
        if zones.empty:
            st.info(
                "Chưa có `zone_id` thật trong predictions. Không hiển thị heatmap để "
                "tránh diễn giải sai. Chạy `scripts/assign_zones.py` và build lại dataset."
            )
        else:
            if not zone_predictions.empty:
                st.caption("Nguồn zone: `zone_predictions.csv` (evaluation có zone 3×3).")
            st.caption(
                "Zone là gán hình học 3×3, không phải sơ đồ chỗ ngồi đã xác minh. "
                "Mọi bảng phải đọc kèm N."
            )
            st.dataframe(zones, width="stretch")
            metric = st.selectbox(
                "Chỉ số heatmap",
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
                    title=f"{metric} theo vùng (N nhỏ → chỉ tham khảo)",
                )
                st.plotly_chart(fig, width="stretch")
            except ImportError:
                st.dataframe(matrix, width="stretch")
            if zones["n_windows"].min() < 20:
                st.warning("Có vùng N < 20; không nên kết luận chênh lệch theo vị trí.")
            positional, evidence = analyze_positions(zone_df)
            st.subheader("Chất lượng và bằng chứng đánh giá theo vị trí")
            if evidence["status"] == "HUMAN_REVIEW_REQUIRED":
                st.info("Chưa có nhãn người để đánh giá độ chính xác theo vùng. Điểm weak agreement chỉ phản ánh mức khớp với quy tắc.")
            st.dataframe(positional, width="stretch")

        st.divider()
        st.subheader("Xuất gói báo cáo ẩn danh (Standardized Export Bundle)")
        if predictions.empty:
            st.info("Chưa có predictions để export.")
        else:
            bucket = st.number_input("Khoảng thời gian tổng hợp (giây)", min_value=10, max_value=600, value=60, step=10)
            if st.button("Tạo gói báo cáo ẩn danh (results, timeline, zone_summary, summary.json, report.md)"):
                out_dir = analysis_dir / "export_ui"
                paths = export_bundle(
                    predictions,
                    out_dir,
                    experiment_id=analysis_id,
                    metric_scope=analysis_scope,
                    bucket_seconds=int(bucket),
                )
                st.success(f"Đã xuất thành công {len(paths)} tệp vào `{out_dir}`.")
                for name, path in paths.items():
                    file = Path(path)
                    st.download_button(
                        label=f"📥 Tải {name} ({file.name})",
                        data=file.read_bytes(),
                        file_name=file.name,
                        mime="text/csv" if file.suffix == ".csv" else ("text/markdown" if file.suffix == ".md" else "application/json"),
                        key=f"dl-{name}",
                    )

    with tabs[4]:
        st.subheader("Tổng quan experiment")
        st.code(selected)
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

        st.write(
            f"Test: **{report.get('total_test_windows', 'N/A')}** window, "
            f"**{report.get('excluded_unlabeled_windows', 'N/A')}** không có nhãn."
        )
        st.json(
            {
                "seed": metadata.get("seed"),
                "split_version": metadata.get("split_version"),
                "feature_schema_version": metadata.get("feature_schema_version"),
                "metric_scope": metric_scope,
                "git_commit": metadata.get("git_commit"),
            }
        )
        with st.expander("Hạn chế & cách đọc số liệu"):
            st.markdown(
                """
                - Đây là **observable participation**, không phải cảm xúc/động cơ.
                - `UNCERTAIN` là trạng thái thiếu căn cứ, không phải lớp hành vi.
                - Không so sánh zone khi `N` quá nhỏ.
                - `head_orientation_proxy` không phải gaze.
                """
            )

    with tabs[5]:
        st.subheader("Preview overlay ẩn danh")
        overlays = list_overlays()
        if not overlays:
            st.info("Không tìm thấy video overlay trong `data/poses/`.")
        else:
            choice = st.selectbox("Video overlay", overlays, format_func=lambda value: f"{Path(value).parent.name}/{Path(value).name}")
            st.video(choice)
            st.caption(
                "Preview đã bật làm mờ vùng đầu của người được phát hiện. "
                "Cần kiểm tra trực quan các trường hợp detector bỏ sót trước khi chia sẻ."
            )


if __name__ == "__main__":
    main()
