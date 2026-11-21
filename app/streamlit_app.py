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
            "Chạy suy luận",
            "Diễn biến theo thời gian",
            "Bản đồ vùng lớp & Xuất báo cáo",
            "Tổng quan experiment",
            "Preview ẩn danh",
        ]
    )

    with tabs[0]:
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

    with tabs[1]:
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

    with tabs[2]:
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

    with tabs[3]:
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

    with tabs[4]:
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
