from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATASET_FOLDER'] = 'static/images/dataset'
app.config['MAX_IMAGES_PER_FACE'] = 8
embeddings_file = 'embeddings.pkl'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)

def extract_face_embeddings(image):
    try:
        image = Image.fromarray(image)
        boxes, _ = mtcnn.detect(image)

        if boxes is None:
            return [], []  # Không phát hiện thấy khuôn mặt

        embeddings = []
        for box in boxes:
            # Convert the box coordinates to integers
            x1, y1, x2, y2 = map(int, box)

            # Crop the face from the image
            face = image.crop((x1, y1, x2, y2))
            face = transforms.functional.resize(face, (160, 160))
            face = transforms.functional.to_tensor(face)
            face = transforms.functional.normalize(face, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            face = face.unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(face).cpu().numpy().flatten()
                embeddings.append(embedding)

        return embeddings, boxes

    except Exception as e:
        print(f"Lỗi khi xử lý ảnh: {str(e)}")
        return [], []



def load_embeddings():
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            return pickle.load(f)
    return {}

def save_embeddings(embeddings_dict):
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        face_id = request.form['face_id']
        if not face_id:
            return redirect(url_for('index'))

        face_dir = os.path.join(app.config['DATASET_FOLDER'], face_id)
        if not os.path.exists(face_dir):
            os.makedirs(face_dir)

        embeddings_dict = load_embeddings()
        current_image_count = len([f for f in os.listdir(face_dir) if f.endswith('.jpg')])

        cap = cv2.VideoCapture(0)

        while current_image_count < app.config['MAX_IMAGES_PER_FACE']:
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces, boxes = extract_face_embeddings(rgb_frame)

                if faces:
                    for i, face_embedding in enumerate(faces):
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        face_img_resized = rgb_frame[y1:y2, x1:x2]
                        face_img_path = os.path.join(face_dir, f'face_{current_image_count}.jpg')
                        cv2.imwrite(face_img_path, cv2.cvtColor(face_img_resized, cv2.COLOR_RGB2BGR))
                        current_image_count += 1
                        print(f'Đã lưu {face_img_path}')

                        embeddings_dict[face_img_path] = {
                            'embedding': face_embedding.tolist(),
                            'name': face_id
                        }
                        save_embeddings(embeddings_dict)

                        cv2.putText(frame, f'Đã lưu: {current_image_count}/{app.config["MAX_IMAGES_PER_FACE"]}', (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        if current_image_count >= app.config['MAX_IMAGES_PER_FACE']:
                            break
                else:
                    print("Không phát hiện thấy khuôn mặt trong khung hình!")

                cv2.imshow('Camera', frame)
                if cv2.waitKey(30) & 0xFF == ord('q') or current_image_count >= app.config['MAX_IMAGES_PER_FACE']:
                    break
            else:
                print("Lỗi khi đọc khung hình từ camera. Đang thoát.")
                break

        cap.release()
        cv2.destroyAllWindows()

        if current_image_count >= app.config['MAX_IMAGES_PER_FACE']:
            print(f'{face_id} đã có đủ ảnh')
            return render_template('success.html', face_id=face_id)

    return render_template('upload.html')

import time

# Global variable to track attendance
attendance_records = {}
ATTENDANCE_TIMEOUT = 60  # Time in seconds to allow re-attendance

def record_attendance(name):
    current_time = time.time()
    if name not in attendance_records or (current_time - attendance_records[name]) > ATTENDANCE_TIMEOUT:
        attendance_records[name] = current_time
        return True
    return False

@app.route('/take_attendance', methods=['GET', 'POST'])
def take_attendance():
    if request.method == 'POST':
        embeddings_dict = load_embeddings()
        if not embeddings_dict:
            return render_template('attendance.html', message='Không có dữ liệu để điểm danh.')

        embeddings = []
        labels = []
        for image_path, data in embeddings_dict.items():
            embeddings.append(np.array(data['embedding']))
            labels.append(data['name'])

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không thể mở camera.")
            return "Không thể mở camera.", 500

        recognized_people = []
        SIMILARITY_THRESHOLD = 0.7

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc khung hình từ camera.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces, boxes = extract_face_embeddings(rgb_frame)

            if faces:
                for i, face_embedding in enumerate(faces):
                    if boxes is not None:
                        x1, y1, x2, y2 = map(int, boxes[i])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    similarities = cosine_similarity([face_embedding], embeddings)
                    max_similarity_idx = np.argmax(similarities)
                    max_similarity = similarities[0, max_similarity_idx]

                    if max_similarity > SIMILARITY_THRESHOLD:
                        predicted_label = labels[max_similarity_idx]

                        # Record attendance only if it's not already recorded
                        if record_attendance(predicted_label):
                            recognized_people.append({
                                'name': predicted_label,
                                'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            cv2.putText(frame, f'{predicted_label} - {max_similarity:.2f}', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, f'{predicted_label} - Already recorded', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    else:
                        cv2.putText(frame, 'Unknown', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow('Attendance', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return render_template('attendance.html', attendance=recognized_people)

    return render_template('attendance.html')


@app.route('/static/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['DATASET_FOLDER']):
        os.makedirs(app.config['DATASET_FOLDER'])
    app.run(debug=True)
