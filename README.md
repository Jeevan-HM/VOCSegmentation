# A Hybrid Framework for Real-Time and Drift-Reduced Camera Pose Estimation

This repository contains the official implementation for the semantic segmentation module used in the research paper, _"A Hybrid Framework for Real-Time and Drift-Reduced Camera Pose Estimation"_.

Our full project presents a novel hybrid system that fuses the real-time performance of traditional Visual Odometry (VO) with the accuracy of neural-based methods to achieve robust, drift-reduced 6-DoF camera pose estimation for long-term operations, particularly in autonomous driving scenarios. The system integrates real-time semantic mapping with a lightweight pose refinement module inspired by UC-NeRF to continuously correct drift.

## VOCSegmentation Module

This specific module, `VOCSegmentation`, is a key component of our system's front-end. It is responsible for generating dense 3D semantic occupancy grids from multi-camera RGB inputs. This semantic understanding of the environment is crucial for robust frame-to-map alignment and for filtering out dynamic objects that would otherwise introduce errors in the pose estimation pipeline.

### Features

- **Semantic Occupancy Grids:** Utilizes a TPV-Former to project multi-view 2D images into a unified 3D semantic representation without requiring explicit depth data.
- **Dynamic Object Filtering:** Leverages semantic segmentation labels to identify and exclude non-static objects (e.g., vehicles, pedestrians) from the pose estimation process, enhancing robustness in dynamic urban environments.
- **Robust Alignment:** Provides the semantic foundation for a Generalized Iterative Closest Point (GICP) algorithm, which aligns the current frame's semantic grid with the global map.
- **Voxel Persistence Filtering:** Employs a filter to retain only stable, frequently observed voxels, increasing map stability and reducing noise from transient scene elements.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- NVIDIA GPU with CUDA support

### Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/Jeevan-HM/VOCSegmentation.git](https://github.com/Jeevan-HM/VOCSegmentation.git)
    cd VOCSegmentation
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To run the semantic segmentation module on a sequence of images from the nuScenes dataset, use the following command structure.

```bash
python run_segmentation.py --config configs/nuscenes_config.yaml --input_dir /path/to/nuscenes/sweeps/CAM_FRONT --output_dir /path/to/output
```
The script will process the images and save the resulting semantic occupancy grids to the specified output directory.Our Hybrid FrameworkOur full framework operates in a closed loop:Front-End (OCC-VO): The VOCSegmentation module provides real-time semantic maps. These are aligned frame-to-frame using a semantic-aware GICP algorithm to produce an initial pose estimate.Back-End (Pose Refinement): A lightweight module adapted from UC-NeRF intermittently refines the poses of selected keyframes. It builds a spatiotemporal correspondence graph and minimizes a geometric reprojection loss to correct accumulated drift without the high computational cost of full NeRF rendering.Closed-Loop Reintegration: The refined poses are reinjected into the OCC-VO pipeline, correcting the global map and improving the accuracy of subsequent pose estimations. This continuous feedback loop enables long-term, drift-resilient localization.