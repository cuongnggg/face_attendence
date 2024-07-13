import os
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import cv2
import numpy as np
from PIL import Image
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import datetime


# Thiết lập device để sử dụng GPU nếu có
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tải pre-trained model InceptionResnetV1 từ facenet-pytorch
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Tạo MTCNN để phát hiện và cắt ảnh khuôn mặt từ ảnh đầu vào
mtcnn = MTCNN(keep_all=True, device=device)

# Load embeddings từ file pickle
embeddings_file = 'embeddings.pkl'
with open(embeddings_file, 'rb') as f:
    embeddings_dict = pickle.load(f)

# Extract embeddings và labels từ embeddings_dict
embeddings = []
labels = []
for image_path, embedding in embeddings_dict.items():
    embeddings.append(embedding)
    labels.append(os.path.basename(os.path.dirname(image_path)))  # Label là tên thư mục cha của ảnh

# Khởi tạo danh sách để lưu thông tin người được nhận diện và thời gian
recognized_people = []

# Ngưỡng độ tương đồng để xác định người nhận diện
SIMILARITY_THRESHOLD = 0.7

# Hàm để trích xuất embeddings từ ảnh khuôn mặt từ camera
def extract_face_embeddings_from_camera(image):
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        boxes, _ = mtcnn.detect(image_pil)

        if boxes is None:
            return None, None

        box = boxes[0]  # Chỉ lấy bounding box đầu tiên
        x, y, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])

        # Cắt và chuẩn hóa khuôn mặt
        face = image_rgb[y:y + h, x:x + w]
        face = Image.fromarray(face)
        face = face.resize((160, 160))
        face = np.array(face)
        face = face / 255.0  # Chuẩn hóa về khoảng [0, 1]
        face = torch.FloatTensor(face).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            embeddings = model(face).cpu().numpy()

        return embeddings.flatten(), (x, y, w, h)

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, None


# Khởi tạo camera
cap = cv2.VideoCapture(0)  # Số 0 là camera mặc định của máy tính

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không thể đọc dữ liệu từ camera.")
        break

    # Phát hiện khuôn mặt và trích xuất embeddings từ frame camera
    embeddings_camera, bbox = extract_face_embeddings_from_camera(frame)

    if embeddings_camera is not None:
        # Vẽ bounding box lên frame camera
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        similarities = cosine_similarity([embeddings_camera], embeddings) #tính độ tương đồng giữa emb từ cam và emb đã train
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[0, max_similarity_idx]

        #gán nhãn nếu tìm thấy độ tương đồng cao nhaast
        predicted_label = labels[max_similarity_idx]

        # Tính toán độ chính xác
        accuracy = max_similarity

        # Nếu độ chính xác lớn hơn ngưỡng xác định, thêm thông tin vào danh sách điểm danh
        if accuracy > SIMILARITY_THRESHOLD:
            # Kiểm tra xem đã điểm danh người này trước đó chưa
            already_recognized = any(person[0] == predicted_label for person in recognized_people)

            if not already_recognized:
                # Lưu thông tin người được nhận diện và thời gian vào danh sách
                recognized_people.append((predicted_label, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        else:
            # Nếu không đạt ngưỡng độ tương đồng, gán nhãn là "Unknown" và thông báo
            predicted_label = "Unknown-face"
            cv2.putText(frame, 'This user dosent have data', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị kết quả nhận diện lên màn hình
        cv2.putText(frame, f'{predicted_label}-{accuracy:.2f}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị frame camera với bounding box và kết quả nhận diện
    cv2.imshow('Attandance Records', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# In danh sách các người đã được nhận diện
print("\nDanh sách người đã được nhận diện và điểm danh:")
if recognized_people:
    for person, time in recognized_people:
        print(f"{person} - {time}")
else:
    print("Không có người nào được nhận diện và điểm danh.")

cap.release()
cv2.destroyAllWindows()
