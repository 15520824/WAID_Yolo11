import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
import tempfile
import os

from ultralytics import YOLO
import supervision as sv

# ================== HÀM XỬ LÝ TRACKING ==================

def run_tracking(model_path, video_file, conf_thres=0.5, img_size=640, min_frames=8):
    # --- Lưu model tạm ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
        tmp_model.write(model_path.read())
        model_disk_path = tmp_model.name

    # --- Lưu video tạm ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
        tmp_vid.write(video_file.read())
        video_disk_path = tmp_vid.name

    # ---- 1. Load YOLO ----
    model = YOLO(model_disk_path)
    class_names = model.names

    # ---- 2. Mở video ----
    cap = cv2.VideoCapture(video_disk_path)
    if not cap.isOpened():
        raise RuntimeError("Không mở được video input")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---- 3. VideoWriter output ----
    out_path = os.path.join(tempfile.gettempdir(), "output_tracked.mp4")

    # Thử nhiều codec để tăng khả năng play được trong browser
    writer = None
    used_codec = None
    for codec in ["avc1", "mp4v"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        vw = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if vw.isOpened():
            writer = vw
            used_codec = codec
            break

    if writer is None:
        cap.release()
        raise RuntimeError("Không tạo được VideoWriter với các codec ['avc1', 'mp4v'].")

    out_writer = writer

    # ---- 4. ByteTrack + annotators ----
    byte_tracker = sv.ByteTrack(
        0.5,  # track_thresh
        30,    # track_buffer
        0.8,   # match_thresh
        fps    # frame_rate
    )

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # tracking_stats: track_id -> info
    # info sẽ chứa:
    #   class_id, class_name, first_frame, last_frame, num_dets, sum_conf
    #   last_seen, current_run, max_run  (để kiểm tra "liên tục")
    tracking_stats = {}
    frame_idx = 0
    prev_time = time.time()

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- YOLO inference ----
        results = model(
            frame,
            conf=conf_thres,
            imgsz=img_size,
            verbose=False
        )
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            annotated_frame = frame.copy()
        else:
            # ---- Convert sang Detections ----
            detections = sv.Detections.from_ultralytics(result)

            # ---- ByteTrack update ----
            if hasattr(byte_tracker, "update_with_detections"):
                tracked_detections = byte_tracker.update_with_detections(detections)
            else:
                tracked_detections = byte_tracker.update(detections)

            # ---- Cập nhật stats (dùng cho CSV & đếm unique) ----
            for xyxy, conf, cls_id, track_id in zip(
                tracked_detections.xyxy,
                tracked_detections.confidence,
                tracked_detections.class_id,
                tracked_detections.tracker_id,
            ):
                tid = int(track_id)
                cid = int(cls_id)
                cname = class_names.get(cid, str(cid))
                conf = float(conf)

                if tid not in tracking_stats:
                    # lần đầu thấy track này
                    tracking_stats[tid] = {
                        "class_id": cid,
                        "class_name": cname,
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "num_dets": 1,
                        "sum_conf": conf,
                        # dùng cho kiểm tra liên tục
                        "last_seen": frame_idx,
                        "current_run": 1,   # chuỗi liên tục hiện tại
                        "max_run": 1,       # chuỗi liên tục dài nhất
                    }
                else:
                    stat = tracking_stats[tid]
                    stat["last_frame"] = frame_idx
                    stat["num_dets"] += 1
                    stat["sum_conf"] += conf

                    # kiểm tra xem frame này có ngay sau frame trước của track không
                    if frame_idx == stat["last_seen"] + 1:
                        stat["current_run"] += 1
                    else:
                        stat["current_run"] = 1  # bị đứt chuỗi, reset

                    stat["last_seen"] = frame_idx
                    # cập nhật max_run
                    if stat["current_run"] > stat["max_run"]:
                        stat["max_run"] = stat["current_run"]

            # ---- Tạo label hiển thị cho từng bbox ----
            labels = []
            for conf, cls_id, track_id in zip(
                tracked_detections.confidence,
                tracked_detections.class_id,
                tracked_detections.tracker_id,
            ):
                class_name = class_names.get(int(cls_id), str(cls_id))
                labels.append(f"ID:{int(track_id)} {class_name} {float(conf):.2f}")

            # ---- Annotate bbox + ID ----
            annotated_frame = box_annotator.annotate(
                scene=frame.copy(), detections=tracked_detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=tracked_detections,
                labels=labels,
            )

            # ---- ĐẾM SỐ LƯỢNG ĐỘNG VẬT TRONG FRAME HIỆN TẠI & VẼ LABEL COUNT ----
            class_counts = {}
            for cls_id in tracked_detections.class_id:
                cname = class_names.get(int(cls_id), str(cls_id))
                class_counts[cname] = class_counts.get(cname, 0) + 1

            y0 = 60
            for cname, cnt in class_counts.items():
                cv2.putText(
                    annotated_frame,
                    f"{cname}: {cnt}",
                    (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                y0 += 30

        # ---- Tính FPS để overlay ----
        curr_time = time.time()
        fps_inst = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        cv2.putText(
            annotated_frame,
            f"FPS: {fps_inst:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        out_writer.write(annotated_frame)

        # ---- Cập nhật progress ----
        frame_idx += 1
        if total_frames > 0:
            progress = min(frame_idx / total_frames, 1.0)
        else:
            progress = 0.0

        progress_bar.progress(progress)
        status_text.text(f"Đang xử lý frame {frame_idx}/{total_frames} ...")

    cap.release()
    out_writer.release()

    # Kiểm tra file video output
    try:
        file_size = os.path.getsize(out_path)
    except FileNotFoundError:
        raise RuntimeError("Không tìm thấy file video output sau khi ghi.")

    if file_size == 0:
        raise RuntimeError("File video output có kích thước 0 bytes (không play được).")

    # ================== TẠO CSV & BẢNG ĐẾM LOÀI (8 FRAME LIÊN TỤC) ==================
    rows = []
    for tid, info in tracking_stats.items():
        first_f = info["first_frame"]
        last_f = info["last_frame"]
        total_span = last_f - first_f + 1
        avg_conf = info["sum_conf"] / max(info["num_dets"], 1)

        # 🔴 Dùng max_run để kiểm tra số frame LIÊN TỤC dài nhất
        max_run = info.get("max_run", info["num_dets"])

        # CHỈ GIỮ TRACK NÀO CÓ ÍT NHẤT min_frames FRAME LIÊN TỤC
        if max_run < min_frames:
            continue

        rows.append({
            "Track_ID": tid,
            "Class_ID": info["class_id"],
            "Class_Name": info["class_name"],
            "Frame_First": first_f,
            "Frame_Last": last_f,
            "Frames_Span": total_span,
            "Max_Consecutive_Frames": max_run,
            "Num_Detections": info["num_dets"],
            "Avg_Confidence": round(avg_conf, 4),
        })

    if rows:
        df_summary = pd.DataFrame(rows)
        df_summary = df_summary.sort_values(["Class_Name", "Track_ID"]).reset_index(drop=True)

        # Bảng đếm loài: mỗi Track_ID là 1 cá thể (đã lọc theo max_run >= min_frames)
        df_counts = (
            df_summary
            .groupby("Class_Name")["Track_ID"]
            .nunique()
            .reset_index(name="So_Luong_Ca_The")
            .sort_values("Class_Name")
            .reset_index(drop=True)
        )
    else:
        df_summary = pd.DataFrame()
        df_counts = pd.DataFrame(columns=["Class_Name", "So_Luong_Ca_The"])

    progress_bar.empty()
    status_text.empty()

    return out_path, df_summary, df_counts


# ================== UI STREAMLIT ==================

st.set_page_config(page_title="YOLO + ByteTrack Demo", layout="wide")
st.title("🐾 YOLO + ByteTrack – Demo Đếm Động Vật Từ Video")

st.sidebar.header("⚙️ Cài đặt")
conf_thres = st.sidebar.slider("Ngưỡng confidence", 0.1, 0.9, 0.5, 0.05)
img_size = st.sidebar.selectbox("Kích thước input (imgsz)", [480, 640, 800, 960], index=1)

# SỐ FRAME TỐI THIỂU ĐỂ TÍNH 1 TRACK (LIÊN TỤC)
min_frames = st.sidebar.slider(
    "Số frame LIÊN TỤC tối thiểu để tính 1 cá thể",
    1, 50, 8, 1
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Bước 1:** Upload model `.pt` và video cần xử lý")

uploaded_model = st.file_uploader("📁 Upload model YOLO (.pt)", type=["pt"])
uploaded_video = st.file_uploader("🎬 Upload video", type=["mp4", "avi", "mov", "mkv"])

process_btn = st.button("🚀 Bắt đầu xử lý")

if process_btn:
    if uploaded_model is None or uploaded_video is None:
        st.error("Vui lòng upload **đủ cả model .pt và video** trước khi chạy.")
    else:
        with st.spinner("Đang chạy YOLO + ByteTrack, vui lòng chờ..."):
            try:
                out_path, df_summary, df_counts = run_tracking(
                    uploaded_model,
                    uploaded_video,
                    conf_thres=conf_thres,
                    img_size=img_size,
                    min_frames=min_frames,
                )
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý: {e}")
            else:
                st.success("✅ Xử lý xong!")

                # --- Hiển thị video output ---
                st.subheader("🎥 Video đã gán ID + số lượng từng loài theo frame")
                st.video(out_path)

                # --- Nút download video ---
                with open(out_path, "rb") as f:
                    video_bytes = f.read()

                st.download_button(
                    "📥 Tải video kết quả",
                    data=video_bytes,
                    file_name="output_tracked.mp4",
                    mime="video/mp4",
                )

                # --- Bảng đếm loài ---
                st.subheader("🐾 Số lượng cá thể theo từng loài (Track có ≥ min_frames frame liên tục)")
                if not df_counts.empty:
                    st.dataframe(df_counts, use_container_width=True)
                else:
                    st.info("Không có đối tượng nào đủ số frame liên tục để tính.")

                # --- Bảng chi tiết tracking + download CSV ---
                st.subheader("📊 Chi tiết tracking (mỗi dòng là 1 Track_ID hợp lệ)")
                if not df_summary.empty:
                    st.dataframe(df_summary, use_container_width=True)

                    csv_bytes = df_summary.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 Tải CSV chi tiết tracking",
                        data=csv_bytes,
                        file_name="tracking_summary.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("Không có dữ liệu tracking để xuất CSV.")
else:
    st.info("👉 Hãy upload model + video, sau đó bấm **“Bắt đầu xử lý”**.")
