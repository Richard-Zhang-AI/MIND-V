---
license: apache-2.0
pipeline_tag: mask-generation
library_name: transformers
---

Repository for SAM 2: Segment Anything in Images and Videos, a foundation model towards solving promptable visual segmentation in images and videos from FAIR. See the [SAM 2 paper](https://arxiv.org/abs/2408.00714) for more information.

The official code is publicly release in this [repo](https://github.com/facebookresearch/segment-anything-2/).

## Usage

For image prediction:

```python
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
```

For video prediction:

```python
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):

    # propagate the prompts to get masklets throughout the video
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

Refer to the [demo notebooks](https://github.com/facebookresearch/segment-anything-2/tree/main/notebooks) for details.

## Usage with 🤗 Transformers


### Automatic Mask Generation with Pipeline

SAM2 can be used for automatic mask generation to segment all objects in an image using the `mask-generation` pipeline:

```python
>>> from transformers import pipeline

>>> generator = pipeline("mask-generation", model="facebook/sam2-hiera-large", device=0)
>>> image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
>>> outputs = generator(image_url, points_per_batch=64)

>>> len(outputs["masks"])  # Number of masks generated
39
```

### Basic Image Segmentation

#### Single Point Click

You can segment objects by providing a single point click on the object you want to segment:

```python
>>> from transformers import Sam2Processor, Sam2Model
>>> import torch
>>> from PIL import Image
>>> import requests

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> model = Sam2Model.from_pretrained("facebook/sam2-hiera-large").to(device)
>>> processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")

>>> image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
>>> raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

>>> input_points = [[[[500, 375]]]]  # Single point click, 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
>>> input_labels = [[[1]]]  # 1 for positive click, 0 for negative click, 3 dimensions (image_dim, object_dim, point_label)

>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

>>> # The model outputs multiple mask predictions ranked by quality score
>>> print(f"Generated {masks.shape[1]} masks with shape {masks.shape}")
Generated 3 masks with shape torch.Size(1, 3, 1500, 2250)
```

#### Multiple Points for Refinement

You can provide multiple points to refine the segmentation:

```python
>>> # Add both positive and negative points to refine the mask
>>> input_points = [[[[500, 375], [1125, 625]]]]  # Multiple points for refinement
>>> input_labels = [[[1, 1]]]  # Both positive clicks

>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
```

#### Bounding Box Input

SAM2 also supports bounding box inputs for segmentation:

```python
>>> # Define bounding box as [x_min, y_min, x_max, y_max]
>>> input_boxes = [[[75, 275, 1725, 850]]]

>>> inputs = processor(images=raw_image, input_boxes=input_boxes, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
```

#### Multiple Objects Segmentation

You can segment multiple objects simultaneously:

```python
>>> # Define points for two different objects
>>> input_points = [[[[500, 375]], [[650, 750]]]]  # Points for two objects in same image
>>> input_labels = [[[1], [1]]]  # Positive clicks for both objects

>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> # Each object gets its own mask
>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
>>> print(f"Generated masks for {masks.shape[0]} objects")
Generated masks for 2 objects
```

### Batch Inference

#### Batched Images

Process multiple images simultaneously for improved efficiency:

```python
>>> from transformers import Sam2Processor, Sam2Model
>>> import torch
>>> from PIL import Image
>>> import requests

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> model = Sam2Model.from_pretrained("facebook/sam2-hiera-large").to(device)
>>> processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")

>>> # Load multiple images
>>> image_urls = [
...     "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg",
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dog-sam.png"
... ]
>>> raw_images = [Image.open(requests.get(url, stream=True).raw).convert("RGB") for url in image_urls]

>>> # Single point per image
>>> input_points = [[[[500, 375]]], [[[770, 200]]]]  # One point for each image
>>> input_labels = [[[1]], [[1]]]  # Positive clicks for both images

>>> inputs = processor(images=raw_images, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> # Post-process masks for each image
>>> all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
>>> print(f"Processed {len(all_masks)} images, each with {all_masks[0].shape[0]} objects")
Processed 2 images, each with 1 objects
```

#### Batched Objects per Image

Segment multiple objects within each image using batch inference:

```python
>>> # Multiple objects per image - different numbers of objects per image
>>> input_points = [
...     [[[500, 375]], [[650, 750]]],  # Truck image: 2 objects
...     [[[770, 200]]]  # Dog image: 1 object
... ]
>>> input_labels = [
...     [[1], [1]],  # Truck image: positive clicks for both objects
...     [[1]]  # Dog image: positive click for the object
... ]

>>> inputs = processor(images=raw_images, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
```

#### Batched Images with Batched Objects and Multiple Points

Handle complex batch scenarios with multiple points per object:

```python
>>> # Add groceries image for more complex example
>>> groceries_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/groceries.jpg"
>>> groceries_image = Image.open(requests.get(groceries_url, stream=True).raw).convert("RGB")
>>> raw_images = [raw_images[0], groceries_image]  # Use truck and groceries images

>>> # Complex batching: multiple images, multiple objects, multiple points per object
>>> input_points = [
...     [[[500, 375]], [[650, 750]]],  # Truck image: 2 objects with 1 point each
...     [[[400, 300]], [[630, 300], [550, 300]]]  # Groceries image: obj1 has 1 point, obj2 has 2 points
... ]
>>> input_labels = [
...     [[1], [1]],  # Truck image: positive clicks
...     [[1], [1, 1]]  # Groceries image: positive clicks for refinement
... ]

>>> inputs = processor(images=raw_images, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
```

#### Batched Bounding Boxes

Process multiple images with bounding box inputs:

```python
>>> # Multiple bounding boxes per image (using truck and groceries images)
>>> input_boxes = [
...     [[75, 275, 1725, 850], [425, 600, 700, 875], [1375, 550, 1650, 800], [1240, 675, 1400, 750]],  # Truck image: 4 boxes
...     [[450, 170, 520, 350], [350, 190, 450, 350], [500, 170, 580, 350], [580, 170, 640, 350]]  # Groceries image: 4 boxes
... ]

>>> # Update images for this example
>>> raw_images = [raw_images[0], groceries_image]  # truck and groceries

>>> inputs = processor(images=raw_images, input_boxes=input_boxes, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
>>> print(f"Processed {len(input_boxes)} images with {len(input_boxes[0])} and {len(input_boxes[1])} boxes respectively")
Processed 2 images with 4 and 4 boxes respectively
```

### Using Previous Masks as Input

SAM2 can use masks from previous predictions as input to refine segmentation:

```python
>>> # Get initial segmentation
>>> input_points = [[[[500, 375]]]]
>>> input_labels = [[[1]]]
>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Use the best mask as input for refinement
>>> mask_input = outputs.pred_masks[:, :, torch.argmax(outputs.iou_scores.squeeze())]

>>> # Add additional points with the mask input
>>> new_input_points = [[[[500, 375], [450, 300]]]]
>>> new_input_labels = [[[1, 1]]]
>>> inputs = processor(
...     input_points=new_input_points,
...     input_labels=new_input_labels,
...     original_sizes=inputs["original_sizes"],
...     return_tensors="pt",
... ).to(device)

>>> with torch.no_grad():
...     refined_outputs = model(
...         **inputs,
...         input_masks=mask_input,
...         image_embeddings=outputs.image_embeddings,
...         multimask_output=False,
...     )
```


### Video Segmentation and Tracking

SAM2's key strength is its ability to track objects across video frames. Here's how to use it for video segmentation:

#### Basic Video Tracking

```python
>>> from transformers import Sam2VideoModel, Sam2VideoProcessor
>>> import torch

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model = Sam2VideoModel.from_pretrained("facebook/sam2-hiera-large").to(device, dtype=torch.bfloat16)
>>> processor = Sam2VideoProcessor.from_pretrained("facebook/sam2-hiera-large")

>>> # Load video frames (example assumes you have a list of PIL Images)
>>> # video_frames = [Image.open(f"frame_{i:05d}.jpg") for i in range(num_frames)]

>>> # For this example, we'll use the video loading utility
>>> from transformers.video_utils import load_video
>>> video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
>>> video_frames, _ = load_video(video_url)

>>> # Initialize video inference session
>>> inference_session = processor.init_video_session(
...     video=video_frames,
...     inference_device=device,
...     torch_dtype=torch.bfloat16,
... )

>>> # Add click on first frame to select object
>>> ann_frame_idx = 0
>>> ann_obj_id = 1
>>> points = [[[[210, 350]]]]
>>> labels = [[[1]]]

>>> processor.add_inputs_to_inference_session(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
...     obj_ids=ann_obj_id,
...     input_points=points,
...     input_labels=labels,
... )

>>> # Segment the object on the first frame
>>> outputs = model(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
... )
>>> video_res_masks = processor.post_process_masks(
...     [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
... )[0]
>>> print(f"Segmentation shape: {video_res_masks.shape}")
Segmentation shape: torch.Size([1, 1, 480, 854])

>>> # Propagate through the entire video
>>> video_segments = {}
>>> for sam2_video_output in model.propagate_in_video_iterator(inference_session):
...     video_res_masks = processor.post_process_masks(
...         [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
...     )[0]
...     video_segments[sam2_video_output.frame_idx] = video_res_masks

>>> print(f"Tracked object through {len(video_segments)} frames")
Tracked object through 180 frames
```

#### Multi-Object Video Tracking

Track multiple objects simultaneously across video frames:

```python
>>> # Reset for new tracking session
>>> inference_session.reset_inference_session()

>>> # Add multiple objects on the first frame
>>> ann_frame_idx = 0
>>> obj_ids = [2, 3]
>>> input_points = [[[[200, 300]], [[400, 150]]]]  # Points for two objects (batched)
>>> input_labels = [[[1], [1]]]

>>> processor.add_inputs_to_inference_session(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
...     obj_ids=obj_ids,
...     input_points=input_points,
...     input_labels=input_labels,
... )

>>> # Get masks for both objects on first frame
>>> outputs = model(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
... )

>>> # Propagate both objects through video
>>> video_segments = {}
>>> for sam2_video_output in model.propagate_in_video_iterator(inference_session):
...     video_res_masks = processor.post_process_masks(
...         [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
...     )[0]
...     video_segments[sam2_video_output.frame_idx] = {
...         obj_id: video_res_masks[i]
...         for i, obj_id in enumerate(inference_session.obj_ids)
...     }

>>> print(f"Tracked {len(inference_session.obj_ids)} objects through {len(video_segments)} frames")
Tracked 2 objects through 180 frames
```

#### Refining Video Segmentation

You can add additional clicks on any frame to refine the tracking:

```python
>>> # Add refinement click on a later frame
>>> refine_frame_idx = 50
>>> ann_obj_id = 2  # Refining first object
>>> points = [[[[220, 280]]]]  # Additional point
>>> labels = [[[1]]]  # Positive click

>>> processor.add_inputs_to_inference_session(
...     inference_session=inference_session,
...     frame_idx=refine_frame_idx,
...     obj_ids=ann_obj_id,
...     input_points=points,
...     input_labels=labels,
... )

>>> # Re-propagate with the additional information
>>> video_segments = {}
>>> for sam2_video_output in model.propagate_in_video_iterator(inference_session):
...     video_res_masks = processor.post_process_masks(
...         [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
...     )[0]
...     video_segments[sam2_video_output.frame_idx] = video_res_masks
```

### Streaming Video Inference

For real-time applications, SAM2 supports processing video frames as they arrive:

```python
>>> # Initialize session for streaming
>>> inference_session = processor.init_video_session(
...     inference_device=device,
...     torch_dtype=torch.bfloat16,
... )

>>> # Process frames one by one
>>> for frame_idx, frame in enumerate(video_frames[:10]):  # Process first 10 frames
...     inputs = processor(images=frame, device=device, return_tensors="pt")
...
...     if frame_idx == 0:
...         # Add point input on first frame
...         processor.add_inputs_to_inference_session(
...             inference_session=inference_session,
...             frame_idx=0,
...             obj_ids=1,
...             input_points=[[[[210, 350], [250, 220]]]],
...             input_labels=[[[1, 1]]],
...             original_size=inputs.original_sizes[0], # need to be provided when using streaming video inference
...         )
...
...     # Process current frame
...     sam2_video_output = model(inference_session=inference_session, frame=inputs.pixel_values[0])
...
...     video_res_masks = processor.post_process_masks(
...         [sam2_video_output.pred_masks], original_sizes=inputs.original_sizes, binarize=False
...     )[0]
...     print(f"Frame {frame_idx}: mask shape {video_res_masks.shape}")
```

#### Video Batch Processing for Multiple Objects

Track multiple objects simultaneously in video by adding them all at once:

```python
>>> # Initialize video session
>>> inference_session = processor.init_video_session(
...     video=video_frames,
...     inference_device=device,
...     torch_dtype=torch.bfloat16,
... )

>>> # Add multiple objects on the first frame using batch processing
>>> ann_frame_idx = 0
>>> obj_ids = [2, 3]  # Track two different objects
>>> input_points = [
...     [[[200, 300], [230, 250], [275, 175]], [[400, 150]]]
... ]  # Object 2: 3 points (2 positive, 1 negative); Object 3: 1 point
>>> input_labels = [
...     [[1, 1, 0], [1]]
... ]  # Object 2: positive, positive, negative; Object 3: positive

>>> processor.add_inputs_to_inference_session(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
...     obj_ids=obj_ids,
...     input_points=input_points,
...     input_labels=input_labels,
... )

>>> # Get masks for all objects on the first frame
>>> outputs = model(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
... )
>>> video_res_masks = processor.post_process_masks(
...     [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
... )[0]
>>> print(f"Generated masks for {video_res_masks.shape[0]} objects")
Generated masks for 2 objects

>>> # Propagate all objects through the video
>>> video_segments = {}
>>> for sam2_video_output in model.propagate_in_video_iterator(inference_session):
...     video_res_masks = processor.post_process_masks(
...         [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
...     )[0]
...     video_segments[sam2_video_output.frame_idx] = {
...         obj_id: video_res_masks[i]
...         for i, obj_id in enumerate(inference_session.obj_ids)
...     }

>>> print(f"Tracked {len(inference_session.obj_ids)} objects through {len(video_segments)} frames")
Tracked 2 objects through 180 frames
```

### Citation

To cite the paper, model, or software, please use the below:
```
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```
