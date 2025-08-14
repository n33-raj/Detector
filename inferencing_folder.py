import argparse
import cv2
import numpy as np
import torch
import time
import sys
import os
from pathlib import Path
import yaml

from model.yolo import YOLO
from utils.util import non_max_suppression, scale, wh2xy

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



# Define make_anchors to patch Head module
def make_anchors(feats, strides, offset=0.5):
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=feats[i].device) + offset
        sy = torch.arange(end=h, device=feats[i].device) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')  ## To Remove warning
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, device=feats[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

## Patch Head class
from model.head import Head
Head.make_anchors = staticmethod(make_anchors)

# Load model with safe unpickling
def load_model(weights_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    torch.serialization.add_safe_globals([YOLO])
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = ckpt['model'].float().fuse().eval()
    model.to(device)
    return model

# Preprocess image
def preprocess(img, new_shape=(640, 640), color=(114, 114, 114)):
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    shape = img.shape[:2]  # (h, w) original shape
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)

    img_padded = img_padded[:, :, ::-1].transpose(2, 0, 1)
    img_padded = np.ascontiguousarray(img_padded, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_padded).unsqueeze(0)
    ratio_pad = ((r, r), (dw, dh))
    return img_tensor, shape, ratio_pad

# Postprocess detections
def postprocess(preds, img_shape, ratio_pad, conf_thres=0.25, iou_thres=0.45):
    detections_list = non_max_suppression(preds, conf_thres, iou_thres)
    if not detections_list or len(detections_list[0]) == 0:
        return []
    detections = detections_list[0]
    detections[:, :4] = scale(
        detections[:, :4],
        img_shape,
        ratio_pad[0],
        ratio_pad[1]
    )
    return detections.cpu().numpy().tolist()

# Draw results on image
def draw_results(image, detections, class_names, colors=None):
    if colors is None:
        colors = np.random.randint(0, 255, size=(len(class_names), 3))
    for *xyxy, conf, cls_id in detections:
        label = f'{class_names[int(cls_id)]} {conf:.2f}'
        xyxy = [int(x) for x in xyxy]
        cv2.rectangle(image, xyxy[:2], xyxy[2:], colors[int(cls_id)].tolist(), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            image,
            (xyxy[0], xyxy[1] - th - 5),
            (xyxy[0] + tw, xyxy[1]),
            colors[int(cls_id)].tolist(), -1
        )
        cv2.putText(
            image, label, (xyxy[0], xyxy[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    return image

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Inference')
    parser.add_argument('--weights', type=str, default=r'C:\Users\lenovo\Downloads\detection\YOLOV8-Pytorch-O\custom-det.pt', help='Model weights path')
    parser.add_argument('--source', type=str, default=r'C:\Users\lenovo\Downloads\detection\YOLOV8-Pytorch-O\test_data', help='Image/video path, folder path or camera ID')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='Filter by class IDs')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.weights, device)

    try:
        with open(r'C:\Users\lenovo\Downloads\detection\YOLOV8-Pytorch-O\config\config.yml', 'r') as f:
            config = yaml.safe_load(f)
        class_names = list(config['names'].values())
    except:
        class_names = [f'class_{i}' for i in range(80)]

    Path(args.output).mkdir(parents=True, exist_ok=True)

    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']

    if args.source.isdigit():
        # Webcam
        cap = cv2.VideoCapture(int(args.source))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            tensor, img_shape, ratio_pad = preprocess(frame, args.img_size)
            tensor = tensor.to(device)
            with torch.no_grad():
                preds = model(tensor)
            detections = postprocess(preds, img_shape, ratio_pad, args.conf_thres, args.iou_thres)
            if args.classes and detections:
                detections = [d for d in detections if int(d[5]) in args.classes]
            result_img = frame.copy()
            if detections:
                result_img = draw_results(result_img, detections, class_names)
            cv2.imshow('YOLOv8 Inference', result_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif os.path.isdir(args.source):
        ## Folder with multiple images
        files = sorted([os.path.join(args.source, f) for f in os.listdir(args.source)
                        if Path(f).suffix.lower() in image_exts])
        for file_path in files:
            frame = cv2.imread(file_path)
            if frame is None:
                continue
            tensor, img_shape, ratio_pad = preprocess(frame, args.img_size)
            tensor = tensor.to(device)
            with torch.no_grad():
                preds = model(tensor)
            detections = postprocess(preds, img_shape, ratio_pad, args.conf_thres, args.iou_thres)
            if args.classes and detections:
                detections = [d for d in detections if int(d[5]) in args.classes]
            result_img = frame.copy()
            if detections:
                result_img = draw_results(result_img, detections, class_names)
            out_path = Path(args.output) / Path(file_path).name
            cv2.imwrite(str(out_path), result_img)
            print(f"Saved result to {out_path}")

    else:
        ## Single image or video
        ext = Path(args.source).suffix.lower()
        if ext in image_exts:
            frame = cv2.imread(args.source)
            tensor, img_shape, ratio_pad = preprocess(frame, args.img_size)
            tensor = tensor.to(device)
            with torch.no_grad():
                preds = model(tensor)
            detections = postprocess(preds, img_shape, ratio_pad, args.conf_thres, args.iou_thres)
            if args.classes and detections:
                detections = [d for d in detections if int(d[5]) in args.classes]
            result_img = frame.copy()
            if detections:
                result_img = draw_results(result_img, detections, class_names)
            out_path = Path(args.output) / Path(args.source).name
            cv2.imwrite(str(out_path), result_img)
            print(f"Saved result to {out_path}")

        elif ext in video_exts:
            cap = cv2.VideoCapture(args.source)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                tensor, img_shape, ratio_pad = preprocess(frame, args.img_size)
                tensor = tensor.to(device)
                with torch.no_grad():
                    preds = model(tensor)
                detections = postprocess(preds, img_shape, ratio_pad, args.conf_thres, args.iou_thres)
                if args.classes and detections:
                    detections = [d for d in detections if int(d[5]) in args.classes]
                result_img = frame.copy()
                if detections:
                    result_img = draw_results(result_img, detections, class_names)
                out_path = Path(args.output) / f'frame_{frame_count:04d}.jpg'
                cv2.imwrite(str(out_path), result_img)
                frame_count += 1
            cap.release()

if __name__ == '__main__':
    main()
