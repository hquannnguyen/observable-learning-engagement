"""Full-flow Streamlit Dashboard for CVF25PRJ10.

Observable Learning Participation Analysis
Group: CVF25PRJ10 | Task code: CVFP26BCPPRJC012

Features:
- Video upload & processing configuration
- YOLOv8-Pose + ByteTrack extraction
- Baseline M0 heuristic participation assessment with Uncertainty Gate
- Privacy-preserving anonymized video preview (head blur, skeleton, anonymous IDs)
- Timeline distribution & 3x3 classroom zone heatmap
- Export results.csv, timeline.csv, zone_summary.csv
"""

from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
import tempfile
import time
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.contracts.labels import (
    HIGH,
    LOW,
    MODERATE,
    UNCERTAIN_LABEL,
    to_display_label,
)
from src.models.baseline_heuristic import HeuristicBaselineClassifier, WindowAssessment
from src.pose.anonymizer import render_anonymized_frame
from src.pose.extractor import PoseExtractor, TrackedPose
from src.reporting.aggregator import (
    build_results_dataframe,
    build_timeline_dataframe,
    build_zone_summary_dataframe,
)
from src.uncertainty.gate import UncertaintyThresholds

# Page Configuration
st.set_page_config(
    page_title="CVF25PRJ10 — Observable Learning Participation",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .zone-box {
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        background: #FFFFFF;
        min-height: 90px;
    }
    .privacy-badge {
        display: inline-block;
        background-color: #DCFCE7;
        color: #15803D;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.markdown('<div class="main-header">🎓 Phân tích Mức độ Tham gia Học tập quan sát được</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Hệ thống phân tích hành vi học tập khách quan từ dữ liệu tư thế và chuyển động (CVF25PRJ10 • CVFP26BCPPRJC012)</div>',
        unsafe_allow_html=True,
    )

    # Info disclaimer banner
    st.info(
        "ℹ️ **Giới hạn phạm vi & Quyền riêng tư:** Hệ thống chỉ ước lượng *mức độ tham gia học tập quan sát được* "
        "từ tư thế/chuyển động của khung xương. Hệ thống **không** suy đoán cảm xúc, sự buồn chán, ý định hay danh tính của sinh viên."
    )

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Cấu hình xử lý")

        input_method = st.radio(
            "Nguồn video",
            [
                "Nhập đường dẫn trực tiếp trên máy (Khuyên dùng cho video > 1GB)",
                "Tải file lên từ trình duyệt",
            ],
            index=0,
            help="Đọc trực tiếp từ ổ cứng giúp bỏ qua việc tải 5GB vào RAM trình duyệt.",
        )

        video_path = None
        if input_method == "Tải file lên từ trình duyệt":
            uploaded_file = st.file_uploader(
                "Chọn file video lớp học (.mp4, .avi, .mov)",
                type=["mp4", "avi", "mov"],
                help="Đã tăng giới hạn tải file lên 10GB.",
            )
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                tfile.flush()
                video_path = tfile.name
        else:
            file_path_input = st.text_input(
                "Đường dẫn file video trên máy (Local Path)",
                value="",
                placeholder="VD: C:\\Users\\Lenovo\\Downloads\\IMG_3354.MOV",
                help="Nhập hoặc copy đường dẫn file video từ máy tính của bạn.",
            )
            if file_path_input.strip():
                clean_path = file_path_input.strip().strip('"').strip("'")
                if Path(clean_path).exists():
                    video_path = clean_path
                    file_size_gb = Path(clean_path).stat().st_size / (1024 * 1024 * 1024)
                    st.success(f"✅ Đã nhận diện file: {Path(clean_path).name} ({file_size_gb:.2f} GB)")
                else:
                    st.error("⚠️ Không tìm thấy file. Vui lòng kiểm tra lại đường dẫn.")

        st.subheader("1. Tần số & Thời lượng")
        sample_fps = st.select_slider(
            "Tần số lấy mẫu (Sample FPS)",
            options=[2, 5, 10, 15],
            value=5,
            help="5 FPS xử lý nhanh trên CPU; 10 FPS là baseline chuẩn của đồ án.",
        )

        duration_mode = st.radio(
            "Khoảng thời gian phân tích",
            ["30 giây đầu (Kiểm tra nhanh)", "1 phút đầu", "3 phút đầu", "Tùy chỉnh khoảng thời gian", "Toàn bộ video"],
            index=0,
            help="Khuyến nghị chạy 30s đến 3 phút trước để kiểm tra chất lượng trên CPU.",
        )

        max_sec = None
        start_sec = 0.0
        if duration_mode == "30 giây đầu (Kiểm tra nhanh)":
            max_sec = 30.0
        elif duration_mode == "1 phút đầu":
            max_sec = 60.0
        elif duration_mode == "3 phút đầu":
            max_sec = 180.0
        elif duration_mode == "Tùy chỉnh khoảng thời gian":
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                start_sec = st.number_input("Bắt đầu (giây)", min_value=0.0, value=0.0, step=5.0)
            with col_s2:
                max_sec = st.number_input("Thời lượng (giây)", min_value=5.0, value=30.0, step=5.0)
        else:
            max_sec = None
            st.warning("⚠️ Chạy toàn bộ video > 1 giờ trên CPU sẽ mất 1–2 giờ.")

        st.subheader("2. Bảo mật & Ẩn danh")
        anonymize_faces = st.checkbox("Làm mờ khuôn mặt (Head Blur)", value=True)
        draw_skeletons = st.checkbox("Hiển thị khung xương (Skeleton)", value=True)
        draw_boxes = st.checkbox("Hiển thị Anonymous Track ID", value=True)

        st.subheader("3. Cổng Bất định (Uncertainty Gate)")
        min_vis = st.slider("Min Visible Ratio", 0.3, 0.9, 0.60, 0.05)
        min_conf = st.slider("Min Keypoint Conf", 0.1, 0.6, 0.30, 0.05)
        min_prob = st.slider("Min Probability Threshold", 0.4, 0.9, 0.60, 0.05)

        run_btn = st.button("🚀 Bắt đầu Phân tích Video", type="primary", use_container_width=True)

    if not video_path:
        st.write("---")
        st.markdown(
            """
            ### 📌 Hướng dẫn xử lý cho video lớn (5GB):
            
            Vì video của bạn nặng **5GB (~1 tiếng 19 phút)**, bạn có 3 cách xử lý rất tiện lợi:
            
            #### 💡 Cách 1 (Nhanh nhất - Không cần cắt file):
            - Tại thanh bên trái, chọn **"Nhập đường dẫn trực tiếp trên máy"**.
            - Nhập đường dẫn file video của bạn (ví dụ: `C:\\Users\\Lenovo\\Downloads\\IMG_3354.MOV`).
            - Chọn **"30 giây đầu"** hoặc **"1 phút đầu"** và bấm **🚀 Bắt đầu Phân tích Video**. Hệ thống sẽ đọc trực tiếp từ ổ cứng mà **không cần upload 5GB** vào trình duyệt!
            
            #### ✂️ Cách 2 (Cắt video pilot 10 phút bằng script có sẵn):
            Mở terminal và chạy lệnh:
            ```powershell
            python scripts/trim_video.py --input "đường_dẫn_tới_IMG_3354.MOV" --duration-min 10 --output "data/raw/session_001/cam_A_pilot.mp4"
            ```
            Lệnh này sẽ tự động trích xuất 10 phút tiêu biểu nhất và nén file từ 5GB xuống còn ~100MB!
            
            #### 📤 Cách 3 (Tải file trực tiếp):
            - Giới hạn tải file qua trình duyệt đã được tăng lên **10GB**. Bạn có thể tải file lên bình thường nếu muốn.
            """
        )
        return
        st.write("---")
        st.subheader("⏳ Đang xử lý video...")
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        metric_cols = st.empty()

        thresholds = UncertaintyThresholds(
            min_visible_ratio=min_vis,
            min_mean_keypoint_confidence=min_conf,
            min_probability=min_prob,
        )

        extractor = PoseExtractor(model_name="yolov8n-pose.pt", confidence_threshold=min_conf)
        classifier = HeuristicBaselineClassifier(thresholds=thresholds)

        # Buffers for temporal sliding window (4s window, 2s stride)
        window_size_frames = max(4, sample_fps * 4)
        stride_frames = max(2, sample_fps * 2)

        track_history: Dict[int, List[TrackedPose]] = defaultdict(list)
        all_assessments: List[WindowAssessment] = []
        preview_frames: List[np.ndarray] = []
        all_track_latest_label: Dict[int, str] = {}

        start_time = time.time()
        processed_count = 0

        # Run extraction and streaming
        for sampled_idx, timestamp_ms, frame, poses in extractor.process_video(
            video_path,
            sample_fps=sample_fps,
            max_duration_seconds=max_sec,
            start_seconds=start_sec,
        ):
            processed_count += 1

            # Update per-track histories
            for pose in poses:
                track_history[pose.track_id].append(pose)

                # Check if we accumulated enough frames for a temporal window
                poses_buf = track_history[pose.track_id]
                if len(poses_buf) >= window_size_frames:
                    # Run heuristic assessment
                    assessment = classifier.assess_window(poses_buf[-window_size_frames:])
                    if assessment:
                        all_assessments.append(assessment)
                        all_track_latest_label[pose.track_id] = assessment.canonical_label

                    # Stride sliding
                    if len(poses_buf) > window_size_frames * 2:
                        track_history[pose.track_id] = poses_buf[-window_size_frames:]

            # Anonymize preview
            if processed_count % max(1, sample_fps // 2) == 0:
                anon_frame = render_anonymized_frame(
                    frame,
                    poses,
                    participation_labels=all_track_latest_label,
                    blur_heads=anonymize_faces,
                    draw_skeletons=draw_skeletons,
                    draw_boxes=draw_boxes,
                )
                preview_frames.append(cv2.cvtColor(anon_frame, cv2.COLOR_BGR2RGB))
                if len(preview_frames) > 20:
                    preview_frames.pop(0)

            # Update progress UI
            elapsed = time.time() - start_time
            fps = processed_count / elapsed if elapsed > 0 else 0
            status_text.text(
                f"Đã xử lý {processed_count} frame ({timestamp_ms/1000:.1f}s) | "
                f"Tốc độ: {fps:.1f} FPS | Số sinh viên theo vết: {len(track_history)}"
            )

        # Final flush for any tracks with fewer frames
        for tid, poses_buf in track_history.items():
            if len(poses_buf) >= 4 and not any(a.track_id == tid for a in all_assessments):
                assessment = classifier.assess_window(poses_buf)
                if assessment:
                    all_assessments.append(assessment)
                    all_track_latest_label[tid] = assessment.canonical_label

        progress_bar.progress(1.0)
        status_text.success(f"✅ Hoàn tất xử lý {processed_count} frame trong {time.time() - start_time:.1f} giây!")

        # Build datasets
        results_df = build_results_dataframe(all_assessments)
        timeline_df = build_timeline_dataframe(results_df, bucket_sec=max(2, int(stride_frames / sample_fps)))
        zone_df = build_zone_summary_dataframe(results_df)

        st.session_state["results_df"] = results_df
        st.session_state["timeline_df"] = timeline_df
        st.session_state["zone_df"] = zone_df
        st.session_state["preview_frames"] = preview_frames

    # Display Results if available
    if "results_df" in st.session_state:
        results_df: pd.DataFrame = st.session_state["results_df"]
        timeline_df: pd.DataFrame = st.session_state["timeline_df"]
        zone_df: pd.DataFrame = st.session_state["zone_df"]
        preview_frames: List[np.ndarray] = st.session_state["preview_frames"]

        st.write("---")
        st.header("📊 Kết quả Phân tích Tổng hợp")

        # Top metric cards
        total_tracks = results_df["track_id"].nunique() if not results_df.empty else 0
        total_windows = len(results_df)
        high_cnt = len(results_df[results_df["display_label"] == "HIGH"]) if total_windows > 0 else 0
        mod_cnt = len(results_df[results_df["display_label"] == "MODERATE"]) if total_windows > 0 else 0
        low_cnt = len(results_df[results_df["display_label"] == "LOW"]) if total_windows > 0 else 0
        unc_cnt = len(results_df[results_df["display_label"] == "UNCERTAIN"]) if total_windows > 0 else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("👥 Sinh viên theo vết", f"{total_tracks}")
        c2.metric("🟢 HIGH", f"{round(high_cnt / total_windows * 100, 1) if total_windows else 0}%", f"{high_cnt} cửa sổ")
        c3.metric("🟡 MODERATE", f"{round(mod_cnt / total_windows * 100, 1) if total_windows else 0}%", f"{mod_cnt} cửa sổ")
        c4.metric("🔘 LOW", f"{round(low_cnt / total_windows * 100, 1) if total_windows else 0}%", f"{low_cnt} cửa sổ")
        c5.metric("⚪ UNCERTAIN", f"{round(unc_cnt / total_windows * 100, 1) if total_windows else 0}%", f"{unc_cnt} bị che/thiếu")

        tab_preview, tab_timeline, tab_zone, tab_table, tab_export = st.tabs([
            "🎥 Xem trước Ẩn danh",
            "📈 Biểu đồ Thời gian",
            "🏫 Vùng Lớp học (3x3)",
            "📋 Bảng Chi tiết",
            "💾 Xuất Dữ liệu",
        ])

        with tab_preview:
            st.markdown("#### 🛡️ Khung hình Ẩn danh (Privacy Preserving)")
            st.caption("Khuôn mặt được làm mờ tự động, chỉ giữ lại khung xương và Anonymous Track ID để phân tích.")
            if preview_frames:
                frame_idx = st.slider("Chọn khung hình xem trước", 0, len(preview_frames) - 1, len(preview_frames) // 2)
                st.image(preview_frames[frame_idx], use_container_width=True)
            else:
                st.info("Không có khung hình preview khả dụng.")

        with tab_timeline:
            st.markdown("#### 📈 Tiến trình Phân bổ Mức độ Tham gia theo Thời gian")
            if not timeline_df.empty:
                chart_data = timeline_df.set_index("time_sec")[["HIGH_pct", "MODERATE_pct", "LOW_pct", "UNCERTAIN_pct"]]
                chart_data.columns = ["Tham gia Cao (HIGH)", "Trung bình (MODERATE)", "Thấp (LOW)", "Không chắc chắn (UNCERTAIN)"]
                st.area_chart(chart_data)
                st.dataframe(timeline_df, use_container_width=True)
            else:
                st.info("Chưa có đủ dữ liệu chuỗi thời gian.")

        with tab_zone:
            st.markdown("#### 🏫 Phân bố Mức độ Tham gia theo Lưới 3x3 Vùng Lớp học")
            st.caption("Quy định: FRONT (gần camera, hàng trước), MIDDLE (giữa), BACK (hàng sau). Luôn báo kèm số lượng mẫu N.")

            # Display 3x3 grid
            row_names = [("BACK", "B"), ("MIDDLE", "M"), ("FRONT", "F")]
            col_names = [("Trái (Left)", "L"), ("Giữa (Center)", "C"), ("Phải (Right)", "R")]

            for r_title, r_code in row_names:
                st.markdown(f"**Hàng {r_title}**")
                cols = st.columns(3)
                for col_idx, (c_title, c_code) in enumerate(col_names):
                    zone_key = f"{r_code}-{c_code}"
                    zone_match = zone_df[zone_df["zone_id"] == zone_key]
                    with cols[col_idx]:
                        if not zone_match.empty and zone_match.iloc[0]["sample_count_N"] > 0:
                            row_z = zone_match.iloc[0]
                            st.markdown(
                                f"""
                                <div class="zone-box">
                                    <b>{zone_key}</b> ({c_title})<br>
                                    <small>Mẫu N = {int(row_z['sample_count_N'])} | Visible: {row_z['mean_visible_ratio']*100:.0f}%</small><br>
                                    <span style="color:#16A34A;">HIGH: {row_z['HIGH_pct']}%</span> | 
                                    <span style="color:#D97706;">MOD: {row_z['MODERATE_pct']}%</span><br>
                                    <span style="color:#64748B;">UNCERTAIN: {row_z['UNCERTAIN_pct']}%</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div class="zone-box" style="background:#F1F5F9; color:#94A3B8;">
                                    <b>{zone_key}</b> ({c_title})<br>
                                    <small>N = 0 (Không có mẫu)</small>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                st.write("")

            st.write("---")
            st.dataframe(zone_df, use_container_width=True)

        with tab_table:
            st.markdown("#### 📋 Chi tiết từng Cửa sổ Quan sát (Sliding Windows)")
            st.dataframe(results_df, use_container_width=True)

        with tab_export:
            st.markdown("#### 💾 Tải xuống Báo cáo & Dữ liệu Kết quả")
            st.write("Các file xuất ra tuân thủ chuẩn schema của dự án (`docs/dataset_schema.md`):")

            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.download_button(
                    "📥 Tải results.csv",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_d2:
                st.download_button(
                    "📥 Tải timeline.csv",
                    data=timeline_df.to_csv(index=False).encode("utf-8"),
                    file_name="timeline.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_d3:
                st.download_button(
                    "📥 Tải zone_summary.csv",
                    data=zone_df.to_csv(index=False).encode("utf-8"),
                    file_name="zone_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
