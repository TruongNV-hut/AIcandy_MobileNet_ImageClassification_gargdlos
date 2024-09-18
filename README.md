# MobileNet and Image Classification

<p align="justify">
<strong>MobileNet</strong> is a family of lightweight convolutional neural networks (CNNs) designed by Google for mobile and embedded vision applications. Introduced in 2017, MobileNet models are optimized for efficiency and speed, making them ideal for devices with limited computational resources. They achieve this by using depthwise separable convolutions, which significantly reduce the number of parameters and computational complexity compared to traditional CNNs. Despite their small size, MobileNet models maintain strong performance in tasks like image classification, object detection, and face recognition, making them widely used in mobile AI applications.
</p>

## Image Classification
<p align="justify">
<strong>Image classification</strong> is a fundamental problem in computer vision where the goal is to assign a label or category to an image based on its content. This task is critical for a variety of applications, including medical imaging, autonomous vehicles, content-based image retrieval, and social media tagging.
</p>


## ❤️❤️❤️


```bash
If you find this project useful, please give it a star to show your support and help others discover it!
```

## Getting Started

### Clone the Repository

To get started with this project, clone the repository using the following command:

```bash
git clone https://github.com/TruongNV-hut/AIcandy_MobileNet_ImageClassification_gargdlos.git
```

### Install Dependencies
Before running the scripts, you need to install the required libraries. You can do this using pip:

```bash
pip install -r requirements.txt
```

### Training the Model

To train the model, use the following command:

```bash
python aicandy_mobilenet_train_enrnptys.py --train_dir ../dataset --num_epochs 100 --batch_size 32 --model_path aicandy_model_out_tdtagoyx/aicandy_model_pth_bmdmrcav.pth
```

### Testing the Model

After training, you can test the model using:

```bash
python aicandy_mobilenet_test_vtvlmtxo.py --image_path ../image_test.jpg --model_path aicandy_model_out_tdtagoyx/aicandy_model_pth_bmdmrcav.pth --label_path label.txt
```

### Converting to ONNX Format

To convert the model to ONNX format, run:

```bash
python aicandy_mobilenet_convert_onnx_ydlekvna.py --model_path aicandy_model_out_tdtagoyx/aicandy_model_pth_bmdmrcav.pth --onnx_path aicandy_model_out_tdtagoyx/aicandy_model_onnx_dngdgvcx.onnx --num_classes 2
```

### More Information

To learn more about this project, [see here](https://aicandy.vn/ung-dung-mang-mobilenet-vao-phan-loai-hinh-anh).

To learn more about knowledge and real-world projects on Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL), visit the website [aicandy.vn](https://aicandy.vn/).

❤️❤️❤️




