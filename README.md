# ADetailer (Python 3.13 + InsightFace Support)

ADetailer is an extension for the stable diffusion webui that does automatic masking and inpainting. It is similar to the Detection Detailer.

## 🎉 Features of This Modified Version

This version is a modified ADetailer designed to **work with Python 3.13 without MediaPipe**:

- **No MediaPipe Required**: Works without MediaPipe dependency (incompatible with Python 3.13)
- **InsightFace Support**: Face detection using InsightFace as a fallback when MediaPipe models are selected
- **Python 3.13 Compatible**: Fully compatible with Python 3.13+
- **YOLO Models**: Uses YOLO models (face_yolov8n, hand_yolov8n, etc.) as primary detection method
- **Automatic Model Download**: All required YOLO models are automatically downloaded on first run

### Modifications

- `mediapipe.py`: Added InsightFace-based face detection and mesh detection implementations
- `install.py`: Automatic YOLO model download from Hugging Face, Python 3.13 compatibility
- `pyproject.toml`: Updated dependencies to remove MediaPipe requirement

Original project: [Bing-su/adetailer](https://github.com/Bing-su/adetailer)

## Install

### Quick Installation

**Note**: While the repository name is `ADetailer_without_mediapipe`, the extension works under the same name as the original ADetailer. If you already have the original ADetailer installed, please uninstall it before installing this version.

1. Open "Extensions" tab in WebUI
2. Open "Install from URL" tab
3. Enter `https://github.com/ussoewwin/ADetailer_without_mediapipe.git` to "URL for extension's git repository"
4. Press "Install" button
5. Go to "Installed" tab, click "Check for updates", and then click "Apply and restart UI"
6. **All required models will be downloaded automatically on first run** (approximately 100-200MB)
7. Restart WebUI completely

### Optional: InsightFace Installation (for MediaPipe model fallback)

If you want to use `mediapipe_face_*` models, you need InsightFace:

**Python 3.13 Compatible Version**: [ussoewwin/Insightface_for_windows](https://huggingface.co/ussoewwin/Insightface_for_windows)

However, we **recommend using YOLO models** (`face_yolov8n.pt`, `face_yolov8s.pt`, etc.) which work out of the box.

## Options

| Model, Prompts                    |                                                                                    |                                                                                                                                                        |
| --------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ADetailer model                   | Determine what to detect.                                                          | `None` = disable                                                                                                                                       |
| ADetailer model classes           | Comma separated class names to detect. only available when using YOLO World models | If blank, use default values.<br/>default = [COCO 80 classes](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) |
| ADetailer prompt, negative prompt | Prompts and negative prompts to apply                                              | If left blank, it will use the same as the input.                                                                                                      |
| Skip img2img                      | Skip img2img. In practice, this works by changing the step count of img2img to 1.  | img2img only                                                                                                                                           |

| Detection                            |                                                                                              |              |
| ------------------------------------ | -------------------------------------------------------------------------------------------- | ------------ |
| Detection model confidence threshold | Only objects with a detection model confidence above this threshold are used for inpainting. |              |
| Mask min/max ratio                   | Only use masks whose area is between those ratios for the area of the entire image.          |              |
| Mask only the top k largest          | Only use the k objects with the largest area of the bbox.                                    | 0 to disable |

If you want to exclude objects in the background, try setting the min ratio to around `0.01`.

| Mask Preprocessing              |                                                                                                                                     |                                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Mask x, y offset                | Moves the mask horizontally and vertically by                                                                                       |                                                                                         |
| Mask erosion (-) / dilation (+) | Enlarge or reduce the detected mask.                                                                                                | [opencv example](https://docs.opencv.org/4.7.0/db/df6/tutorial_erosion_dilatation.html) |
| Mask merge mode                 | `None`: Inpaint each mask<br/>`Merge`: Merge all masks and inpaint<br/>`Merge and Invert`: Merge all masks and Invert, then inpaint |                                                                                         |

Applied in this order: x, y offset → erosion/dilation → merge/invert.

#### Inpainting

Each option corresponds to a corresponding option on the inpaint tab. Therefore, please refer to the inpaint tab for usage details on how to use each option.

## ControlNet Inpainting

You can use the ControlNet extension if you have ControlNet installed and ControlNet models.

Support `inpaint, scribble, lineart, openpose, tile, depth` controlnet models. Once you choose a model, the preprocessor is set automatically. It works separately from the model set by the Controlnet extension.

If you select `Passthrough`, the controlnet settings you set outside of ADetailer will be used.

## Advanced Options

API request example: [wiki/REST-API](https://github.com/Bing-su/adetailer/wiki/REST-API)

`[SEP], [SKIP], [PROMPT]` tokens: [wiki/Advanced](https://github.com/Bing-su/adetailer/wiki/Advanced)

## Media

- 🎥 [どこよりも詳しい After Detailer (adetailer)の使い方 ① 【Stable Diffusion】](https://youtu.be/sF3POwPUWCE)
- 🎥 [どこよりも詳しい After Detailer (adetailer)の使い方 ② 【Stable Diffusion】](https://youtu.be/urNISRdbIEg)

- 📜 [ADetailer Installation and 5 Usage Methods](https://kindanai.com/en/manual-adetailer/)

## Model

| Model                 | Target                | mAP 50                        | mAP 50-95                     | Notes                                      |
| --------------------- | --------------------- | ----------------------------- | ----------------------------- | ------------------------------------------ |
| face_yolov8n.pt       | 2D / realistic face   | 0.660                         | 0.366                         | ✅ **Recommended** - Works out of the box  |
| face_yolov8s.pt       | 2D / realistic face   | 0.713                         | 0.404                         | ✅ **Recommended** - Higher accuracy       |
| hand_yolov8n.pt       | 2D / realistic hand   | 0.767                         | 0.505                         | ✅ Works out of the box                    |
| person_yolov8n-seg.pt | 2D / realistic person | 0.782 (bbox)<br/>0.761 (mask) | 0.555 (bbox)<br/>0.460 (mask) | ✅ Works out of the box                    |
| person_yolov8s-seg.pt | 2D / realistic person | 0.824 (bbox)<br/>0.809 (mask) | 0.605 (bbox)<br/>0.508 (mask) | ✅ Higher accuracy                         |
| mediapipe_face_full   | realistic face        | -                             | -                             | ⚠️ Uses InsightFace fallback (less precise) |
| mediapipe_face_short  | realistic face        | -                             | -                             | ⚠️ Uses InsightFace fallback (less precise) |
| mediapipe_face_mesh   | realistic face        | -                             | -                             | ⚠️ Uses InsightFace fallback (less precise) |

**Note**: `mediapipe_face_*` models use InsightFace as a fallback since MediaPipe is not available in Python 3.13. For best results, **use YOLO models** (face_yolov8n.pt or face_yolov8s.pt) which provide better accuracy and work natively.

The YOLO models can be found on huggingface [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer).

For a detailed description of the YOLO8 model, see: https://docs.ultralytics.com/models/yolov8/#overview

YOLO World model: https://docs.ultralytics.com/models/yolo-world/

### Additional Model

Put your [ultralytics](https://github.com/ultralytics/ultralytics) yolo model in `models/adetailer`. The model name should end with `.pt`.

It must be a bbox detection or segment model and use all label.

## How it works

ADetailer works in three simple steps.

1. Create an image.
2. Detect object with a detection model and create a mask image.
3. Inpaint using the image from 1 and the mask from 2.

## Development

ADetailer is developed and tested using the stable-diffusion 1.5 model, for the latest version of [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) repository only.

## License

ADetailer is a derivative work that uses two AGPL-licensed works (stable-diffusion-webui, ultralytics) and is therefore distributed under the AGPL license.

## See Also

- https://github.com/ototadana/sd-face-editor
- https://github.com/continue-revolution/sd-webui-segment-anything
- https://github.com/portu-sim/sd-webui-bmab
