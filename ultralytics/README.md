# Ultralytics detection runtime

This directory is an inference-only subset of Ultralytics `v8.3.9`, modified
for SAR-Sense and its MBE-Net checkpoint.

Kept capabilities:

- load the local `Detct_prdc/MBE-Net/weights/best.pt` checkpoint;
- run object-detection inference on images;
- render detection boxes and labels;
- provide the custom MBE-Net modules referenced by the checkpoint.

Removed capabilities include training, export, Hub integration, model
downloads, tracking, classification, segmentation, pose, OBB, SAM, RT-DETR,
NAS and their datasets/configurations. Only local `.pt` weights are supported.

The runtime was reduced and modified on 2026-07-31. Upstream source:
[ultralytics/ultralytics v8.3.9](https://github.com/ultralytics/ultralytics/tree/v8.3.9).
The retained Ultralytics-derived code remains available under the
[GNU AGPL-3.0](LICENSE).
