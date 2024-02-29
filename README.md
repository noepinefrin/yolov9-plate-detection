**YOLOv9 Plate Detection on Image**

Training Notebook are just published [click here!](https://www.kaggle.com/code/noepinefrin/yolov9-fine-tuning-custom-dataset-plate-detection)

**Prediction format**
```python
[*xyxy, confidence, predicted_class]
```

*Clone Original YOLOv9 Repository*

```python
!git clone --recursive https://github.com/WongKinYiu/yolov9.git
# Change the directory
%cd yolov9/
```

*Install YOLOv9 Dependencies*
```python
!pip install -r requirements.txt -q
!pip install supervision datasets pyyaml -q
```

*Import Utility Functions*
```python
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox
from PIL import Image

import numpy as np
import supervision as sv
import torch
import os

HOME = os.getcwd()

%matplotlib inline
```
**Inference Function**
```python
@smart_inference_mode()
def predict(
    image_path: str,
    weights: str,
    data: str,
    imgsz: tuple,
    conf_thres: float,
    iou_thres: float,
    device: str,
):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, fp16=False, data=data)
    stride, names, pt = model.stride, model.names, model.pt

    # Load image
    image = Image.open(image_path)
    img0 = np.array(image)
    assert img0 is not None, f'Image Not Found {image_path}'
    img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Init bounding box annotator and label annotator
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_RIGHT, text_scale=0.4)

    # Inference
    pred = model(img, augment=False, visualize=False)

    # Apply NMS
    pred_t = non_max_suppression(pred[0], conf_thres, iou_thres, classes=None, max_det=1000)

    # Process detections
    for i, det in enumerate(pred_t):
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                # Transform detections to supervisions detections
                detections = sv.Detections(
                    xyxy=torch.stack(xyxy).cpu().numpy().reshape(1, -1),
                    class_id=np.array([int(cls)]),
                    confidence=np.array([float(conf)])
                )

                # Labels
                labels = [
                    f"{names[int(class_id)]} {confidence:0.2f}"
                    for class_id, confidence
                    in zip(detections.class_id, detections.confidence)
                ]

                img0 = bounding_box_annotator.annotate(img0, detections)
                img0 = label_annotator.annotate(img0, detections, labels)

        return img0
```
**Create Inference YAML file**
```python
def create_yaml(base_dir: str):
    import yaml

    data_yaml = {
        'train': 'data/train/images',
        'val': 'data/validation/images',
        'test': 'data/test/images',
        'nc': 1, # number of the classes, in our cases is just 1 because we only want to detect license plate,
        'names': {
            0: 'license_plate'
        },
    }

    with open(f'{base_dir}/custom_data.yaml', 'w') as file:
        yaml.dump(data_yaml, file)

create_yaml(base_dir=HOME)
```

**Initialize variables**
```python
FINETUNED_WEIGHTS_PATH = os.path.join(HOME, 'plate_detection.pt') # The fine-tuned weights must be under the HOME directory.
CUSTOM_DATA_YAML_PATH = os.path.join('HOME', 'custom_data.yaml') # Custom yaml file is necessary for inferencing
DEVICE = '0' if torch.cuda.is_available() else 'cpu' # IF GPU is available, its preferred.
```

**Get Inference**
```python
%matplotlib inline

image_path = '<paste-your-image-path-here>'

img = predict(
    image_path=image_path,
    weights=FINETUNED_WEIGHTS_PATH,
    data=CUSTOM_DATA_YAML_PATH,
    imgsz=640,
    conf_thres=0.5,
    iou_thres=0.2,
    device=DEVICE,
)

sv.plot_image(img)
```

![](/example/863.jpg)