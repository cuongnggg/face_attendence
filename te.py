import os
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms
import pickle
from tqdm import tqdm

# Thiết lập device để sử dụng GPU nếu có sẵn, nếu không thì sử dụng CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model InceptionResnetV1
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Khởi tạo MTCNN
mtcnn = MTCNN(keep_all=True, device=device)

# Hàm trích xuất embeddings từ ảnh khuôn mặt
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
        print(f"Lỗi xử lý ảnh: {str(e)}")
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

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Thư mục lưu trữ dataset
dataset_dir = 'dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

while True:
    face_id = input("Nhập tên và mã số sinh viên : ")
    if face_id.lower() == 'q':
        break

    face_dir = os.path.join(dataset_dir, face_id)
    if not os.path.exists(face_dir):
        os.makedirs(face_dir)

    # Số lượng ảnh tối đa muốn lưu cho mỗi khuôn mặt
    max_images_per_face = 8
    current_image_count = 0

    while current_image_count < max_images_per_face:
        # Đọc frame từ camera
        ret, frame = cap.read()
        if ret:
            # Chuyển đổi frame sang định dạng RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Phát hiện khuôn mặt
            boxes, _ = mtcnn.detect(rgb_frame)

            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = boxes[0].astype(int)

                # Vẽ bounding box quanh khuôn mặt
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if y2 > y1 and x2 > x1:
                    # Cắt và resize ảnh khuôn mặt
                    face_img = rgb_frame[y1:y2, x1:x2]
                    face_img_resized = cv2.resize(face_img, (160, 160))

                    if face_img_resized.size != 0:
                        # Lưu ảnh khuôn mặt vào thư mục dataset/face_id
                        face_img_path = os.path.join(face_dir, f'face_{current_image_count}.jpg')
                        cv2.imwrite(face_img_path, cv2.cvtColor(face_img_resized, cv2.COLOR_RGB2BGR))
                        current_image_count += 1
                        print(f'Đã lưu {face_img_path}')

                        # Trích xuất embeddings từ ảnh khuôn mặt vừa lưu
                        embeddings = extract_face_embeddings(face_img_resized)

                        # Lưu embeddings vào dictionary
                        if embeddings is not None:
                            embeddings_dict[face_img_path] = embeddings

                        # Hiển thị số lượng ảnh đã lưu
                        cv2.putText(frame, f'Số ảnh đã lưu: {current_image_count}/{max_images_per_face}', (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        # Kiểm tra nếu đã đủ số lượng ảnh thì thoát
                        if current_image_count >= max_images_per_face:
                            break
                    else:
                        print("Ảnh khuôn mặt sau khi resize trống!")

                else:
                    print("Vùng khuôn mặt không hợp lệ!")

            # Hiển thị frame với bounding box và số lượng ảnh đã lưu
            cv2.imshow('Open the camera', frame)

            # Thoát nếu nhấn 'q' hoặc đã lưu đủ số lượng ảnh
            if cv2.waitKey(30) & 0xFF == ord('q') or current_image_count >= max_images_per_face:
                break

        else:
            print("Lỗi khi đọc frame từ camera. Thoát.")
            break

    # Nếu đã đủ số lượng ảnh thì thoát vòng lặp nhập face_id
    if current_image_count >= max_images_per_face:
        print(f'{face_id} đã thêm đủ số ảnh')
        break

# Giải phóng camera và đóng tất cả cửa sổ OpenCV
cap.release()
cv2.destroyAllWindows()

# Lưu embeddings_dict vào file bằng pickle
output_file = 'embeddings.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(embeddings_dict, f)

print(f'Đã lưu file embeddings vào {output_file}')
