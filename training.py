import os
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import pickle
from tqdm import tqdm

# Cài đặt device để sử dụng GPU nếu có sẵn, nếu không thì sử dụng CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Tải mô hình FaceNet pre-trained từ thư viện facenet-pytorch
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Tạo MTCNN để phát hiện và cắt ảnh khuôn mặt từ ảnh đầu vào
mtcnn = MTCNN(keep_all=True, device=device)


# Hàm để trích xuất embeddings từ ảnh khuôn mặt
def extract_face_embeddings(image):
    try:
        # Chuyển đổi kích thước và chuẩn hóa ảnh
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        boxes, _ = mtcnn.detect(image)

        if boxes is None:
            return None

        boxes = boxes[0]
        face = transforms.functional.crop(image, *boxes)
        face = transforms.functional.resize(face, (160, 160))
        face = transforms.functional.to_tensor(face)
        face = transforms.functional.normalize(face, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        face = face.unsqueeze(0).to(device)

        with torch.no_grad():
            embeddings = model(face).cpu().numpy()

        return embeddings.flatten()

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None


# Hàm để lấy đường dẫn của tất cả các file ảnh trong dataset
def get_image_paths(root_dir):
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths


# Đường dẫn đến thư mục chứa dataset của bạn
dataset_dir = 'dataset'

# Lấy đường dẫn của tất cả các file ảnh trong dataset
image_paths = get_image_paths(dataset_dir)

# Dictionary để lưu trữ embeddings của từng ảnh
embeddings_dict = {}

# Duyệt qua từng đường dẫn ảnh và trích xuất embeddings
for image_path in tqdm(image_paths):
    try:
        # Đọc ảnh từ đường dẫn
        image = cv2.imread(image_path)

        # Trích xuất embeddings từ ảnh khuôn mặt
        embeddings = extract_face_embeddings(image)

        # Lưu embeddings vào dictionary
        if embeddings is not None:
            embeddings_dict[image_path] = embeddings

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        continue

# Lưu embeddings_dict vào file bằng pickle
output_file = 'embeddings.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(embeddings_dict, f)

print(f'File đã được lưu vào {output_file}')
