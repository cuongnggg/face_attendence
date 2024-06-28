import cv2
import os
from facenet_pytorch import MTCNN

# Initialize MTCNN
mtcnn = MTCNN()

cap = cv2.VideoCapture(0)
dataset_dir = 'dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

while True:
    face_id = input("Input your Name and StudentIDs: ")
    if face_id.lower() == 'q':
        break

    face_dir = os.path.join(dataset_dir, face_id)
    if not os.path.exists(face_dir):
        os.makedirs(face_dir)

    # Nhập số lượng ảnh tối đa muốn lưu
    max_images_per_face = 100
    current_image_count = 0

    while current_image_count < max_images_per_face:
        # Read a frame from the camera
        ret, frame = cap.read()
        if ret:
            # Convert BGR image to RGB (MTCNN takes RGB image as input)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces using MTCNN
            boxes, _ = mtcnn.detect(rgb_frame)

            if boxes is not None and len(boxes) > 0:
                # Lấy bounding box của khuôn mặt đầu tiên được phát hiện
                x1, y1, x2, y2 = boxes[0].astype(int)

                # Vẽ hộp giới hạn xung quanh khuôn mặt
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Kiểm tra và lưu ảnh khuôn mặt
                if y2 > y1 and x2 > x1:
                    face_img = rgb_frame[y1:y2, x1:x2]
                    face_img_resized = cv2.resize(face_img, (160, 160))

                    # Đảm bảo ảnh khuôn mặt đã resize không rỗng
                    if face_img_resized.size != 0:
                        # Lưu ảnh khuôn mặt vào thư mục dataset/face_id
                        face_img_path = os.path.join(face_dir, f'face_{current_image_count}.jpg')
                        cv2.imwrite(face_img_path, cv2.cvtColor(face_img_resized, cv2.COLOR_RGB2BGR))
                        current_image_count += 1
                        print(f'Saved {face_img_path}')

                        # Hiển thị số lượng ảnh đã lưu
                        cv2.putText(frame, f'Images saved: {current_image_count}/{max_images_per_face}', (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        # Kiểm tra nếu đã lưu đủ số lượng ảnh thì thoát vòng lặp
                        if current_image_count >= max_images_per_face:
                            break
                    else:
                        print("Resized face image is empty!")
                else:
                    print("Invalid face region detected!")

            # Hiển thị frame với bounding box và số lượng ảnh đã lưu
            cv2.imshow('Open the camera', frame)

            # Thoát nếu nhấn 'q' hoặc đã lưu đủ số lượng ảnh
            if cv2.waitKey(30) & 0xFF == ord('q') or current_image_count >= max_images_per_face:
                break

        else:
            print("Error reading frame from camera. Exiting.")
            break

    # Nếu đã lưu đủ số lượng ảnh thì thoát khỏi vòng lặp nhập face_id
    if current_image_count >= max_images_per_face:
        print(f'{face_id} has added enough photos')
        break

# Giải phóng camera và đóng tất cả cửa sổ OpenCV
cap.release()
cv2.destroyAllWindows()
