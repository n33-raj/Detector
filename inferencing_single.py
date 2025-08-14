#inferencing.py

import argparse
import cv2
import numpy as np
import torch
import time
import sys
import os
from pathlib import Path
import yaml


# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from model.yolo import YOLO
from utils.util import non_max_suppression, scale, wh2xy

# Define make_anchors to patch Head module
def make_anchors(feats, strides, offset=0.5):
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=feats[i].device) + offset
        sy = torch.arange(end=h, device=feats[i].device) + offset
        # sy, sx = torch.meshgrid(sy, sx)
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')         # debug to remove the warning after inferencing
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, device=feats[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

# Monkey-patch Head class
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

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    
    # Resize
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Compute padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # (w,h) padding
    dw /= 2
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)

    # BGR → RGB, HWC → CHW, normalize
    img_padded = img_padded[:, :, ::-1].transpose(2, 0, 1)  
    img_padded = np.ascontiguousarray(img_padded, dtype=np.float32) / 255.0

    # Convert to torch tensor with batch dimension
    img_tensor = torch.from_numpy(img_padded).unsqueeze(0)

    # ratio_pad = ((gain_x, gain_y), (pad_x, pad_y))
    ratio_pad = ((r, r), (dw, dh))

    return img_tensor, shape, ratio_pad



# Postprocess detections
def postprocess(preds, img_shape, ratio_pad, conf_thres=0.25, iou_thres=0.45):
    detections_list = non_max_suppression(preds, conf_thres, iou_thres)

    # If NMS returns nothing or empty
    if not detections_list or len(detections_list[0]) == 0:
        return []

    detections = detections_list[0]

    # Scale detections back to original image size
    detections[:, :4] = scale(
        detections[:, :4],
        img_shape,
        ratio_pad[0],  # (gain_x, gain_y)
        ratio_pad[1]   # (pad_x, pad_y)
    )

    # Convert to list of lists for uniformity
    return detections.cpu().numpy().tolist()


# Draw results on image
def draw_results(image, detections, class_names, colors=None):
    if colors is None:
        colors = np.random.randint(0, 255, size=(len(class_names), 3))
    
    for *xyxy, conf, cls_id in detections:
        label = f'{class_names[int(cls_id)]} {conf:.2f}'
        xyxy = [int(x) for x in xyxy]
        
        # Draw rectangle
        cv2.rectangle(image, xyxy[:2], xyxy[2:], colors[int(cls_id)].tolist(), 2)
        
        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            image, 
            (xyxy[0], xyxy[1] - th - 5),
            (xyxy[0] + tw, xyxy[1]),
            colors[int(cls_id)].tolist(), -1
        )
        
        # Draw text
        cv2.putText(
            image, label, (xyxy[0], xyxy[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    return image

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Inference')
    parser.add_argument('--weights', type=str, default=r'C:\Users\lenovo\Downloads\detection\YOLOV8-Pytorch-O\custom-det.pt', help='Model weights path')
    parser.add_argument('--source', type=str, default=r'C:\Users\lenovo\Downloads\detection\YOLOV8-Pytorch-O\test_data\MV5BMTg0MjkxMDA4N15BMl5BanBnXkFtZTcwMDM2MTY0OQ@@._V1_.jpg', help='Image/video path or camera ID')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='Filter by class IDs')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    args = parser.parse_args()

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.weights, device)
    
    # Load class names from config
    try:
        with open(r'C:\Users\lenovo\Downloads\detection\YOLOV8-Pytorch-O\config\config.yml', 'r') as f:
            config = yaml.safe_load(f)
        class_names = list(config['names'].values())
    except:
        class_names = [f'class_{i}' for i in range(80)]  # Default to COCO names

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Process input source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))  # Webcam
    else:
        cap = cv2.VideoCapture(args.source)  # Video file or image
        
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess
        tensor, img_shape, ratio_pad = preprocess(frame, args.img_size)
        tensor = tensor.to(device)
        
        # Inference
        start = time.time()
        with torch.no_grad():
            preds = model(tensor)
        inference_time = time.time() - start
        
        # Postprocess
        detections = postprocess(preds, img_shape, ratio_pad, args.conf_thres, args.iou_thres)
        
        # Filter by class
        if args.classes and detections:
            detections = [d for d in detections if int(d[5]) in args.classes]
        
        ## Draw results
        result_img = frame.copy()

        # if isinstance(detections, (list, tuple)) and len(detections) > 0 or \
        # isinstance(detections, torch.Tensor) and detections.numel() > 0:


        if isinstance(detections, (list, tuple)) and len(detections) > 0:
            result_img = draw_results(result_img, detections, class_names)
        
        ## Display FPS
        # fps = 1 / inference_time if inference_time > 0 else 0
        # cv2.putText(
        #     result_img, f'FPS: {fps:.1f} | Objects: {len(detections)}',
        #     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        # )
        
        # Save or show results
        if args.source.isdigit():
            cv2.imshow('YOLOv8 Inference', result_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                break
        else:
            if cap.get(cv2.CAP_PROP_FRAME_COUNT) == 1:  # Single image
                out_path = Path(args.output) / Path(args.source).name
            else:  # Video
                out_path = Path(args.output) / f'frame_{frame_count:04d}.jpg'
            cv2.imwrite(str(out_path), result_img)
            print(f'Saved result to {out_path}')
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()