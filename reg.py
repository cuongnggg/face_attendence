import os
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import cv2
import numpy as np
from PIL import Image
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from torchvision import transforms


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

        # Vẽ bounding box vào frame
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cắt và chuẩn hóa khuôn mặt
        face = image_rgb[y:y + h, x:x + w]
        face = Image.fromarray(face)
        face = transforms.functional.resize(face, (160, 160))
        face = transforms.functional.to_tensor(face)
        face = transforms.functional.normalize(face, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        face = face.unsqueeze(0).to(device)

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

        # So sánh embeddings của camera với các embeddings trong dữ liệu đã huấn luyện
        similarities = cosine_similarity([embeddings_camera], embeddings)
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[0, max_similarity_idx]

        # Tính toán độ chính xác
        accuracy = 1 - max_similarity

        # Xác định người tương ứng với embeddings có độ tương đồng cao nhất
        predicted_label = labels[max_similarity_idx]

        # Hiển thị kết quả nhận diện lên màn hình
        cv2.putText(frame, f'{predicted_label}-{accuracy:.2f}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị frame camera với bounding box và kết quả nhận diện
    cv2.imshow('Face Recognition', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ OpenCV
cap.release()
cv2.destroyAllWindows()
