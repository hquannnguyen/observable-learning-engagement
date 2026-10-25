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

import pandas as pd
import streamlit as st

from src.reporting.aggregation import aggregate_timeline, aggregate_zones
from src.reporting.exporter import export_bundle

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
    return sorted([p.name for p in OVERLAYS.glob("*overlay*.mp4")])


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
        ["Tổng quan", "Timeline", "Vùng lớp", "Preview ẩn danh", "Export", "Video mới"]
    )

    with tabs[0]:
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

    with tabs[1]:
        st.subheader("Phân phối theo thời gian")
        if predictions.empty:
            st.info("Experiment này chưa có `predictions.csv`.")
        else:
            timeline = aggregate_timeline(predictions)
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
                        title="Số window theo phút (weak-label)",
                    )
                    st.plotly_chart(fig, width="stretch")
                except ImportError:
                    st.dataframe(view, width="stretch")
                st.dataframe(view, width="stretch")

    with tabs[2]:
        st.subheader("Phân bố theo vùng lớp (3×3)")
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

    with tabs[3]:
        st.subheader("Preview overlay ẩn danh")
        overlays = list_overlays()
        if not overlays:
            st.info("Không tìm thấy video overlay trong `data/poses/`.")
        else:
            choice = st.selectbox("Video overlay", overlays)
            st.video(str(OVERLAYS / choice))
            st.caption(
                "Overlay chỉ có skeleton + anonymous track ID. Không có tên/MSSV. "
                "Không dùng video này để nhận diện cá nhân."
            )

    with tabs[4]:
        st.subheader("Xuất báo cáo")
        if predictions.empty:
            st.info("Chưa có predictions để export.")
        else:
            bucket = st.number_input("Bucket (giây)", min_value=10, max_value=600, value=60, step=10)
            if st.button("Tạo bundle export"):
                out_dir = exp_dir / "export_ui"
                paths = export_bundle(
                    predictions,
                    out_dir,
                    experiment_id=metadata.get("experiment_id"),
                    metric_scope=metric_scope,
                    bucket_seconds=int(bucket),
                )
                st.success(f"Đã ghi {len(paths)} file vào `{out_dir}`.")
                for name, path in paths.items():
                    file = Path(path)
                    st.download_button(
                        label=f"Tải {name}",
                        data=file.read_bytes(),
                        file_name=file.name,
                        mime="text/csv" if file.suffix == ".csv" else "application/json",
                        key=f"dl-{name}",
                    )


    with tabs[5]:
        st.subheader("Phân tích video mới (upload → pose → model → export)")
        st.caption(
            "Chạy toàn bộ pipeline trên CPU có thể mất vài phút. "
            "Video upload được lưu local dưới `data/raw/uploads/` và không commit."
        )
        checkpoint_options = {}
        for exp_name in experiments:
            meta = load_json(str(EXPERIMENTS / exp_name / "metadata.json"))
            ckpt = meta.get("checkpoint_path")
            if ckpt and Path(ckpt).is_file():
                label = f"{meta.get('experiment_id', exp_name)} ({meta.get('model_type', '?')})"
                checkpoint_options[label] = ckpt

        if not checkpoint_options:
            st.info("Không tìm thấy checkpoint nào có metadata.")
        else:
            ckpt_label = st.selectbox("Checkpoint", sorted(checkpoint_options))
            max_frames = st.number_input(
                "Giới hạn số frame sampled (để demo nhanh)",
                min_value=20,
                max_value=9000,
                value=150,
                step=10,
            )
            uploaded = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"])
            if uploaded is not None and st.button("Chạy phân tích"):
                upload_dir = ROOT / "data" / "raw" / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                video_path = upload_dir / uploaded.name
                video_path.write_bytes(uploaded.getbuffer())
                out_dir = ROOT / "reports" / "inference" / Path(uploaded.name).stem
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "predict_video.py"),
                    "--video", str(video_path),
                    "--checkpoint", checkpoint_options[ckpt_label],
                    "--out-dir", str(out_dir),
                    "--max-frames", str(int(max_frames)),
                ]
                with st.spinner("Đang chạy pipeline..."):
                    proc = subprocess.run(
                        cmd, cwd=str(ROOT), capture_output=True, text=True
                    )
                if proc.returncode != 0:
                    st.error("Pipeline thất bại. Chi tiết cuối log:")
                    st.code(proc.stderr[-3000:] or proc.stdout[-3000:])
                else:
                    st.success(f"Hoàn tất. Kết quả: `{out_dir}`")
                    new_preds = load_csv(str(out_dir / "predictions.csv"))
                    if not new_preds.empty:
                        st.write(
                            f"{len(new_preds)} window | "
                            f"{int((new_preds['gated_label'] != 'UNCERTAIN').sum())} accepted | "
                            f"{int((new_preds['gated_label'] == 'UNCERTAIN').sum())} uncertain"
                        )
                        st.dataframe(new_preds, width="stretch")
                    summary_file = out_dir / "export" / "summary.json"
                    if summary_file.is_file():
                        st.download_button(
                            "Tải summary.json",
                            data=summary_file.read_bytes(),
                            file_name="summary.json",
                            mime="application/json",
                        )


if __name__ == "__main__":
    main()
