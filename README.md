<p align="center">

  <h1 align="center">Facebuilder</h1>
  <p align="center">
    <a href="https://www.cubox.ai/"><strong>CUBOX AI Lab</strong></a><sup></sup>
  <div align="center">
    <img src="./assets/facebuilder.png" alt="Logo" width="100%">
  </div>

  <p align="center">
    <a href='https://github.com/yc4ny/facebuilder'>
      <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
    <a href="" target='_blank'>
      <img src="https://visitor-badge.laobi.icu/badge?page_id=yc4ny.facebuilder&left_color=gray&right_color=orange">
    </a>
  </p>
</p>

<br/>

**Facebuilder** is a robust object detection system designed to detect hazardous items in X-ray images. The system leverages state-of-the-art <a href="https://docs.ultralytics.com/"><strong>YOLO Object Detection</strong></a><sup></sup> (You Only Look Once) architecture for real-time object detection. It is capable of identifying various dangerous objects in X-ray scans, assisting in security checks and ensuring safety in environments such as airports, checkpoints, and border security. The model is trained on a custom dataset of X-ray images and provides both bounding boxes and object classifications.
<br/>

## News :triangular_flag_on_post:
- [2024/12/30] [Code] Initial Release. Training code and pretrained checkpoints for inference are provided.⭐

## TODO
- [x] Release training code.
- [x] Release inference code. 
- [x] Release partial X-ray object dataset. 
- [ ] Release full X-ray object dataset. 

##  Instruction 📜
### Dataset description
<p align="center"><img src="./assets/dataset.png" alt="Logo" width="100%"></a> </p>
This dataset is designed for the development and evaluation of object detection models, specifically for detecting hazardous objects in X-ray images. The dataset contains images of various hazardous items commonly found in security checks and scans. The dataset is labeled with ground truth bounding boxes and class labels for each object, following the YOLO format. The dataset is organized into images and labels directories, where each image corresponds to a label file containing the ground truth annotations for the objects present in the image.


### Dataset download
Usage of the data is only permitted for research and educational purposes. Note that we only distribute the data to researchers. You must have at least ~200GB of storage space to download the full dataset. For those with limited space, there will be some example image/annotation files of the dataset. 
```
gdown --folder https://drive.google.com/drive/folders/10O7A9oYfrzod2dj0rteQCiUtmHydEdME?usp=sharing
```

### Dataset Structure
After downloading the data, please construct the layout of `dataset/` as follows:
```
|-- data
|   |-- images/  
|   |   |-- test/
|   |   |   |-- E3S690_20221007_02791894_S.xxx.png
|   |   |   |-- ...
|   |   |-- train/
|   |   |   |-- E3S690_20221007_02791894_S.xxx.png
|   |   |   |-- ...
|   |   |-- val/
|   |   |   |-- E3S690_20221007_02791894_S.xxx.png
|   |   |   |-- ...
|   |-- labels/  
|   |   |-- test/
|   |   |   |-- E3S690_20221007_02791894_S.xxx.txt
|   |   |   |-- ...
|   |   |-- train/
|   |   |   |-- E3S690_20221007_02791894_S.xxx.txt
|   |   |   |-- ...
|   |   |-- val/
|   |   |   |-- E3S690_20221007_02791894_S.xxx.txt
|   |   |   |-- ...
```
The dataset is organized into three primary sets for effective model development:

- **Train Set**: 212,119 images and corresponding labels used for training the object detection model.
- **Validation Set**: 24,104 images and labels utilized for model evaluation during training to fine-tune performance.
- **Test Set**: 24,236 images and labels reserved for assessing the model's accuracy and generalization after training.

<details>
<summary><span style="font-weight: bold;">Class Information</span></summary>

Number of Classes: 66 

'Gun', 'Bullet', 'Magazin', 'Slingshot', 'Speargun tip', 'Shuriken', 'Dart pin', 'Electroshock weapon', 'LAGs product', 'Ax', 'Knife-A', 'Knife-F', 'Knife-B', 'Other Knife', 'Knife-D', 'Knife blade', 'Knife-Z', 'Multipurpose knife', 'Scissors-A', 'Scissors', 'Knuckle', 'Hammer', 'Prohibited tool-D', 'Drill', 'Prohibited tool-A', 'Monkey wrench', 'Pipe wrench', 'Prohibited tool-C', 'Prohibited tool-B', 'Vise plier', 'Shovel', 'Prohibited tool-E', 'Bolt cutter', 'Saw', 'Electric saw', 'Dumbbel', 'Ice skates', 'Baton', 'Handscuffs', 'Explosive weapon-A', 'LAGs product(Plastic-A)', 'LAGs product(Plastic-B)', 'LAGs product(Plastic-C)', 'LAGs product(Plastic-D)', 'LAGs product(Glass)', 'LAGs product(Paper)', 'LAGs product(Stainless)', 'LAGs product(Vinyl)', 'LAGs product(Aluminum)', 'LAGs product(Tube)', 'Firecracker', 'Torch', 'Solid fuel', 'Lighter', 'Nunchaku', 'Exploding golf balls', 'Knife-E', 'Green onion slicer', 'Hex key(over 10cm)', 'Kettlebell', 'Kubotan', 'Arrow tip', 'Billiard ball', 'Drill bit(over 6cm)', 'Buttstock', 'Card knife'

</details>

### Label Format
Each label file contains detailed annotations for every object in the image, structured as follows:

- **Class ID**: An integer representing the object class (e.g., `1` for Gun, `2` for Bullet, etc.).
- **Bounding Box**: Normalized coordinates specifying the object’s position and size:
  - `center x`: Horizontal center of the bounding box (relative to image width).
  - `center y`: Vertical center of the bounding box (relative to image height).
  - `width`: Width of the bounding box (relative to image width).
  - `height`: Height of the bounding box (relative to image height).

This structure ensures compatibility with common object detection frameworks such as YOLO. 
  
## Getting Started
### Installation
Using the virtual environment by running:
```bash
conda create -n xray python==3.10
conda activate xray
pip install -r requirements.txt
```
### Pretrained Checkpoints 
We provide pretrained checkpoints of the detection model. 

```bash
gdown --folder https://drive.google.com/drive/folders/1WkS0ucYUkyRMIcYSu-OfAG_PNiynBcwh?usp=sharing
```

### Training
You can start running training with:
```bash
python -m torch.distributed.run --nproc_per_node 8 train.py --data configs/config.yaml --weights yolo11x.pt  --device 0,1,2,3,4,5,6,7
```

#### Training Command Flags

--**nproc_per_node**: Specifies the number of processes to launch per node. For multi-GPU training, this should match the number of GPUs available (e.g., 8 for 8 GPUs).

--**data**: Path to the dataset configuration file (e.g., configs/config.yaml), which defines the training, validation, and test dataset paths as well as class names.

--**weights**: Path to the pre-trained model checkpoint (e.g., yolo11x.pt), which serves as the initial weights for training.

--**device**: Specifies which GPUs to use for training. Provide a comma-separated list of GPU indices (e.g., 0,1,2,3,4,5,6,7 for 8 GPUs).

We used 8 * A100 80G GPUs for training. Adjust flags according to your environment. 

### Evaluation
<p align="center"><img src="./assets/eval.png" alt="Logo" width="100%"></a> </p>

Run evaluation after training with: 
```bash
python eval.py
```

This runs the trained model in the evaluation set and shows the performance metrics such as precision, recall, mAP (mean average precision), and other relevant metrics depending on the model and configuration.

The evaluation results for the provided pretrained checkpoints are as follows: 
| Class                         | Images | Instances | Box(P) | Box(R) | mAP50 | mAP50-95 |
|-------------------------------|--------|-----------|--------|--------|-------|----------|
| all                           | 24104  | 31292     | 0.975  | 0.932  | 0.953 | 0.871    |

### Inference
<p align="center"><img src="./assets/inference.png" alt="Logo" width="100%"></a> </p>
After training the model or downloading a pre-trained checkpoint, you can run inference on a set of images to detect hazardous objects in X-ray scans. The following instructions will guide you through the inference process.

To run inference on a set of images, use the following command:

```bash
python inference.py --input <path_to_input_folder> --ckpt <path_to_model_checkpoint> --output <path_to_output_folder> [--save_annot]
```

## Citation
If you use this data or model in your research or work, please cite this repository:

```bibtex
@misc{xraydataset2024,
  author = {CUBOX AI Lab},
  title = {X-Ray Hazardous Object Detection},
  year = {2024},
  url = {https://github.com/yc4ny/X-Ray-Object-Detector}
}
```
## Contact
For technical questions, please contact yhugestar@gmail.com. For license, please contact yonwoo.choi@cubox.ai.