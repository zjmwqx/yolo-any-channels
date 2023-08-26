from ultralytics import YOLO
import cv2
# Load a model
# model = YOLO('yolov8n.pt')
# #
# # Train the model
# model.train(data='coco128.yaml', epochs=20, imgsz=640, ch=2, workers=0)
# #

# 读取图像
# image = cv2.imread('gray_image.jpg', 1)
# print(image.shape)
#
# # 转换为灰度图像
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # 保存灰度图像
# cv2.imwrite('gray_image.jpg', gray_image)
# # Load a pretrained YOLOv8n model
# image = cv2.imread('gray_image.jpg')
#
model = YOLO('G:/cv/ultralytics/runs/detect/train42/weights/best.pt')
#
# # Run inference on 'bus.jpg' with arguments
model.predict('000000000073.jpg', save=True, conf=0.5, ch=2)
