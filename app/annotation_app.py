"""Independent human annotation for Quân and Phong. No model labels are shown."""
import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.contracts.annotation_io import read_annotation_csv, save_annotation_row
from src.contracts.labels import ANNOTATION_STATES, CANONICAL_TRAINING_LABELS
from src.contracts.annotation_rubric import EVIDENCE_OPTIONS, RUBRIC_VERSION, evidence_label
from src.pose.annotation_preview import render_annotation_preview
from scripts.dataset_qa import validate_annotation_dataframe

PEOPLE = {
    "Phong": ("Vũ Văn Phong", "annotator_b.csv"),
    "Tài": ("Vũ Đức Tài", "annotator_a.csv"),
    "Quân": ("Nguyễn Hữu Quân", "annotator_a.csv"),
}
DATASET_NAMES = {
    "review_v03_independent_20261124": "600 cửa sổ · phiếu độc lập Phong và Tài",
    "review_v03": "Bộ gán nhãn v0.3 (600 cửa sổ YOLO11)",
    "human_review_v0_2": "Dataset cũ · bộ gán nhãn chính",
    "external_review_20261109": "Hai video mới · đánh giá bổ sung",
}
PHASES = {"UNKNOWN": "Chưa xác định", "lecture": "Giảng bài", "individual_work": "Làm bài cá nhân", "discussion": "Thảo luận"}
STATES = {"VALID": "Đủ bằng chứng — gán mức tham gia", "OCCLUDED": "Bị che khuất", "TRANSITION": "Đang chuyển hoạt động/trạng thái", "INVALID": "Không đủ bằng chứng / chưa rõ nhiệm vụ / lỗi ID hoặc video"}
LABELS = dict(zip(CANONICAL_TRAINING_LABELS, ["Cao", "Trung bình", "Thấp"]))

st.set_page_config(page_title="Phong & Tài · Gán nhãn", layout="wide")
st.title("Gán nhãn hành vi quan sát được")
st.caption("Mỗi nhãn = một người (track ID) trong một cửa sổ khoảng 4 giây. Xem đúng người trong toàn bộ cửa sổ; Phong và Tài gán trên phiếu riêng.")
with st.expander("Hướng dẫn chọn nhãn"):
    st.markdown("""
1. Xem đúng **track ID** trong toàn bộ khoảng thời gian, xác định pha hoạt động thực tế.
2. Chọn **Đủ bằng chứng** khi có thể đánh giá; chọn che khuất/chuyển hoạt động/không đủ dữ liệu khi cần.
3. Với cửa sổ đủ bằng chứng: **Cao** = nhiều tín hiệu rõ phù hợp nhiệm vụ; **Trung bình** = tín hiệu yếu hoặc không liên tục; **Thấp** = nhiều tín hiệu không phù hợp nhiệm vụ trong phần lớn cửa sổ.

Cúi đầu khi đọc/viết có thể phù hợp; quay sang bạn khi thảo luận có thể phù hợp. Không suy luận cảm xúc, ý định hoặc mức hiểu bài. Nếu chưa biết pha, xem thêm ngữ cảnh bằng video gốc; không đoán pha để ép nhãn.
""")
    guide_path = ROOT / "docs/annotation_guide.md"
    if guide_path.is_file():
        st.download_button("Tải hướng dẫn đầy đủ", guide_path.read_text(encoding="utf-8"), file_name="annotation_guide.md", mime="text/markdown")
st.info("Tiêu chí thí điểm: xem từng người, xác định nhiệm vụ rồi đánh dấu bằng chứng trong 4 khoảng thời gian. Cần thử và thống nhất giữa hai người trước khi gán hàng loạt.")
with st.expander("Cơ sở gán nhãn: hành vi nào phù hợp với từng hoạt động?"):
    st.markdown("""
| Hoạt động | Bằng chứng có thể phù hợp | Chưa đủ để kết luận |
|---|---|---|
| Giảng bài | Thân/đầu hướng vùng giảng dạy; ghi chép phù hợp bài đang diễn ra | Chỉ ngồi yên, chỉ nhìn thẳng, chỉ cúi đầu |
| Làm bài cá nhân | Đọc tài liệu bài tập; tay viết/thao tác nhiệm vụ đã xác định | Cúi đầu nhưng không thấy đang làm gì |
| Thảo luận | Hướng về nhóm và có cử chỉ/trao đổi liên quan nhiệm vụ | Chỉ quay sang bạn; không biết nội dung/hoạt động |

**Không phù hợp** cần hành vi cụ thể mâu thuẫn với nhiệm vụ đã biết. Không tự coi dùng điện thoại, quay đầu hoặc cúi đầu là không phù hợp. Nếu chưa rõ, chọn *chưa rõ*; nếu bị che hoặc không tìm đúng người, chọn *không quan sát được*.

**Bảng quy đổi thử nghiệm:** ≥3/4 khoảng phù hợp → Cao; ≥3/4 khoảng không phù hợp → Thấp; ≥3/4 khoảng có bằng chứng rõ nhưng xen kẽ → Trung bình. Nếu ít hơn 3/4 khoảng rõ, không gán mức tham gia. Đây là lựa chọn vận hành của nhóm để thử nghiệm, không phải ngưỡng khoa học đã được xác nhận.
""")
    rubric_path = ROOT / "docs/ANNOTATION_RUBRIC_PILOT.md"
    if rubric_path.is_file():
        st.download_button("Tải bảng tiêu chí thí điểm", rubric_path.read_text(encoding="utf-8"), file_name=rubric_path.name, mime="text/markdown")

person = st.radio("Bạn là người gán nào?", list(PEOPLE), index=None, horizontal=True)
if person is None:
    st.info("Chọn Phong hoặc Tài để mở phiếu của bạn.")
    st.stop()
annotator, filename = PEOPLE[person]
forms = sorted((ROOT / "data/annotations").glob(f"**/{filename}"), key=lambda p: (p.parent.name != "human_review_v0_2", str(p)))
if not forms:
    st.info("Chưa có phiếu. Chuẩn bị bộ review bằng scripts/prepare_inference_annotations.py.")
    st.stop()
selected = Path(st.selectbox("Bộ video cần gán", [str(p) for p in forms], key=f"dataset:{person}", format_func=lambda p: DATASET_NAMES.get(Path(p).parent.name, str(Path(p).parent.relative_to(ROOT / "data/annotations")))))
if selected.name != filename:
    st.error("Phiếu không khớp người gán. Hãy tải lại trang.")
    st.stop()
if selected.parent.name == "human_review_v0_2":
    st.caption("Bộ gán nhãn lấy từ video của dataset cũ. Nhãn do luật sinh được ẩn để hai người đánh giá độc lập; giữ nguyên phân chia train/val/test.")
elif selected.parent.name == "external_review_20261109":
    st.caption("Hai video bạn cung cấp để thử trên dữ liệu mới. Bộ này dùng đánh giá bổ sung, không thay thế dataset cũ.")
try:
    df = read_annotation_csv(selected)
except (ValueError, OSError) as exc:
    st.error(f"Không đọc được phiếu: {exc}")
    st.stop()
if df.empty:
    st.info("Phiếu chưa có cửa sổ.")
    st.stop()

complete = df["annotation_state"].isin(ANNOTATION_STATES) & df["annotator_id"].ne("")
done = int(complete.sum())
st.write(f"**{annotator}** · Đã lưu **{done}/{len(df)}** cửa sổ theo từng người · Còn **{len(df) - done}**")
st.progress(done / len(df))
scope = str(selected)
window_key = f"window:{scope}"
pending_key = f"pending:{scope}"
unfinished = list(df.index[~complete])
if pending_key in st.session_state:
    st.session_state[window_key] = st.session_state.pop(pending_key)
if window_key not in st.session_state or st.session_state[window_key] not in df.index:
    pilot = [i for i in unfinished if "split" in df and df.at[i, "split"] in ("train", "val")]
    st.session_state[window_key] = pilot[0] if pilot else (unfinished[0] if unfinished else 0)

def move_window(step):
    st.session_state[window_key] = max(0, min(len(df) - 1, st.session_state[window_key] + step))

def next_unfinished():
    current = st.session_state[window_key]
    remaining = [i for i in unfinished if i > current] + [i for i in unfinished if i <= current]
    if remaining:
        st.session_state[window_key] = remaining[0]

nav = st.columns(3)
nav[0].button("← Cửa sổ trước", on_click=move_window, args=(-1,), disabled=st.session_state[window_key] == 0)
nav[1].button("Cửa sổ tiếp →", on_click=move_window, args=(1,), disabled=st.session_state[window_key] == len(df) - 1)
nav[2].button("Đến cửa sổ chưa gán", on_click=next_unfinished, disabled=not unfinished)
index = st.selectbox("Cửa sổ", list(df.index), key=window_key, format_func=lambda i: f"{'✓' if complete.at[i] else '○'} {i + 1}. [{df.at[i, 'split'] if 'split' in df else ''}] {df.at[i, 'session_id']} | track {df.at[i, 'track_id']} | {float(df.at[i, 't_start_ms']) / 1000:g}–{float(df.at[i, 't_end_ms']) / 1000:g} s")
row = df.loc[index]
if "annotation_notice" in st.session_state:
    st.success(st.session_state.pop("annotation_notice"))
st.info(f"Chỉ gán cho track {row['track_id']} · {float(row['t_start_ms']) / 1000:g}–{float(row['t_end_ms']) / 1000:g} giây · Vùng {row.get('zone_id', 'UNKNOWN')}")
video = Path(row.get("video_path", ""))
if not video.is_absolute():
    video = ROOT / video
pose_path = ROOT / "data/poses" / f"{row['session_id']}_{row['camera_id']}_poses.csv"
if not pose_path.is_file() and row.get("pose_overlay_path"):
    pose_path = Path(row["pose_overlay_path"]).parent / "poses.csv"
    if not pose_path.is_absolute():
        pose_path = ROOT / pose_path
if video.is_file() and pose_path.is_file():
    try:
        with st.spinner(f"Đang làm nổi bật người ID {row['track_id']} trong đúng đoạn video…"):
            focused, focus_meta = render_annotation_preview(video, pose_path, row["track_id"],
                float(row["t_start_ms"]), float(row["t_end_ms"]), ROOT / "data/annotations/.preview_cache")
        st.subheader(f"Chỉ gán người trong khung VÀNG — ID {row['track_id']}")
        st.caption("Bên trái: người cần gán sáng hơn và có khung vàng. Bên phải: phóng to cùng người đó. Khi không thấy đúng ID, clip báo không quan sát được; không theo người khác để đoán tiếp.")
        st.video(str(focused), loop=True)
    except (ValueError, OSError, RuntimeError) as exc:
        st.warning(f"Chưa định vị được người cần gán: {exc}. Không gán mức tham gia khi chưa xác định đúng người.")
else:
    st.warning("Chưa có video/tọa độ để làm nổi bật đúng người. Không gán mức tham gia khi chưa xác định được người cần xem.")
if st.checkbox("Xem thêm video gốc toàn cảnh để xác định nhiệm vụ (chỉ dùng local)") and video.is_file():
    st.caption("Video ngữ cảnh chưa làm mờ. Nhãn vẫn chỉ áp dụng cho người và khoảng thời gian ghi phía trên.")
    st.video(str(video), start_time=max(0, float(row["t_start_ms"]) / 1000 - 5))

st.subheader("Ghi bằng chứng trước khi chọn mức tham gia")
st.caption("Chia đoạn thành 4 khoảng gần bằng nhau. Đánh dấu hành vi của đúng người trong từng khoảng; các lựa chọn này do bạn quan sát, không lấy từ mô hình.")
try:
    saved_evidence = json.loads(row.get("annotation_evidence", "") or "{}")
except (TypeError, ValueError):
    saved_evidence = {}
stored_bins = saved_evidence.get("observations", [None] * 4)
observations = []
for part, column in enumerate(st.columns(4)):
    t0 = float(row["t_start_ms"]) / 1000 + part * (float(row["t_end_ms"]) - float(row["t_start_ms"])) / 4000
    t1 = float(row["t_start_ms"]) / 1000 + (part + 1) * (float(row["t_end_ms"]) - float(row["t_start_ms"])) / 4000
    previous = stored_bins[part] if len(stored_bins) == 4 else None
    observations.append(column.selectbox(f"Khoảng {part + 1}: {t0:.2f}–{t1:.2f} s", list(EVIDENCE_OPTIONS),
        index=list(EVIDENCE_OPTIONS).index(previous) if previous in EVIDENCE_OPTIONS else None,
        format_func=EVIDENCE_OPTIONS.get, key=f"evidence:{scope}:{row['window_id']}:{part}"))
rubric_label, rubric_reason = evidence_label(observations)
st.write(f"**Theo bảng tiêu chí thí điểm: {LABELS.get(rubric_label, 'Chưa đủ bằng chứng')}**. {rubric_reason}")

with st.form(f"annotation:{scope}:{row['window_id']}"):
    cols = st.columns(2)
    phase = cols[0].selectbox("Pha hoạt động thực tế", list(PHASES), index=list(PHASES).index(row["class_phase"]) if row["class_phase"] in PHASES else 0, format_func=PHASES.get)
    phase_evidence = st.text_input("Bằng chứng xác định nhiệm vụ / pha hoạt động", value=row.get("phase_evidence", ""), help="Ví dụ: thấy giáo viên giảng trước bảng; đề bài yêu cầu làm cá nhân; hoạt động nhóm đã được xác minh. Nếu chưa biết, giữ Chưa xác định.")
    state = cols[1].selectbox("Khả năng quan sát", list(STATES), index=list(STATES).index(row["annotation_state"]) if row["annotation_state"] in STATES else None, format_func=STATES.get)
    label = st.selectbox("Mức tham gia quan sát được nếu VALID", list(LABELS), index=list(LABELS).index(row["participation_label"]) if row["participation_label"] in LABELS else None, format_func=LABELS.get, help="Chỉ lưu mức tham gia khi đủ bằng chứng và đã xác định pha.")
    behavior = st.text_input("Hành vi quan sát được", value=row.get("atomic_behavior", ""), help="Ví dụ READING_WRITING; HAND_RAISED; PEER_ORIENTED. Dùng dấu ; khi có nhiều hành vi.")
    confidence = st.slider("Độ tin cậy của người gán", 0.0, 1.0, float(row["annotation_confidence"]) if pd.notna(row["annotation_confidence"]) else 0.8, 0.05)
    occlusions = ["none", "partial", "heavy"]
    occlusion = st.selectbox("Mức che khuất", occlusions, index=occlusions.index(row["occlusion_level"]) if row["occlusion_level"] in occlusions else 0, format_func={"none": "Không", "partial": "Một phần", "heavy": "Nhiều"}.get)
    notes = st.text_area("Ghi chú", value=row.get("notes", ""))
    confirmed = st.checkbox("Tôi đã xem hết đoạn, xác định đúng người và xác nhận bằng chứng theo bảng tiêu chí")
    save_next = st.form_submit_button("Lưu và đến cửa sổ chưa gán tiếp", type="primary")
    save = st.form_submit_button("Lưu tại cửa sổ này")
if save or save_next:
    if state is None or (state == "VALID" and (label is None or phase == "UNKNOWN")):
        st.error("Chọn khả năng quan sát. Với VALID, cần xác định pha thực tế và mức tham gia.")
    elif state == "VALID" and (not confirmed or not behavior.strip() or not phase_evidence.strip()):
        st.error("Cần ghi hành vi cụ thể, bằng chứng xác định nhiệm vụ và xác nhận đã xem đúng người trong toàn bộ đoạn.")
    elif state == "VALID" and (rubric_label is None or label != rubric_label):
        st.error(f"Bằng chứng chưa phù hợp với nhãn đã chọn. Theo bảng tiêu chí: {LABELS.get(rubric_label, 'Chưa đủ bằng chứng')}. {rubric_reason}")
    else:
        updates = {"class_phase": phase, "annotation_state": state, "participation_label": label if state == "VALID" else "",
                   "label_source": "HUMAN" if state == "VALID" else "", "label_rule_version": "", "annotator_id": annotator,
                   "atomic_behavior": behavior, "annotation_confidence": confidence, "occlusion_level": occlusion,
                   "notes": notes, "annotation_guide_version": RUBRIC_VERSION, "phase_verified": phase != "UNKNOWN",
                   "phase_evidence": phase_evidence.strip(),
                   "annotation_evidence": json.dumps({"rubric_version": RUBRIC_VERSION, "observations": observations,
                       "basis": "DIRECT_HUMAN_OBSERVATION", "confirmed": confirmed}, ensure_ascii=False),
                   "phase_source": "HUMAN_WINDOW_REVIEW" if phase != "UNKNOWN" else "UNKNOWN"}
        candidate = df.loc[[index]].copy()
        for key, value in updates.items():
            candidate.at[index, key] = value
        valid, errors = validate_annotation_dataframe(candidate)
        if not valid:
            st.error("; ".join(errors))
        else:
            try:
                saved = save_annotation_row(selected, row["window_id"], updates)
            except (OSError, ValueError) as exc:
                st.error(f"Chưa lưu được phiếu: {exc}")
            else:
                remaining = list(saved.index[~(saved["annotation_state"].isin(ANNOTATION_STATES) & saved["annotator_id"].ne(""))])
                following = [i for i in remaining if i > index] + [i for i in remaining if i < index]
                if save_next and following:
                    st.session_state[pending_key] = following[0]
                st.session_state["annotation_notice"] = f"Đã lưu nhãn của {person}." + (" Bạn đã hoàn tất phiếu." if not remaining else "")
                st.rerun()

st.download_button("Tải CSV phiếu của tôi", selected.read_bytes(), file_name=f"{selected.parent.name}_{person}.csv", mime="text/csv")
st.caption("Lưu vào CSV trên máy chủ đang chạy tool. Bấm tải CSV để giữ một bản riêng. Chuyển cửa sổ khi chưa lưu sẽ bỏ các thay đổi chưa gửi.")
