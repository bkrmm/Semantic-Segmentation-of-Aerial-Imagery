# Semantic-Segmentation-of-Aerial-Imagery
Semantic segmentation of satellite imagery using U-Net. Assigns labels to each pixel for land cover (buildings, vegetation, roads, water). Ideal for analyzing satellite data.
# Semantic Segmentation of Satellite Imagery Using U-Net

## Overview

This project performs semantic segmentation on satellite imagery to classify each pixel into distinct land cover categories such as buildings, roads, vegetation, water, and unlabeled areas. It uses the U-Net architecture for precise pixel-wise classification. The dataset is pre-processed into smaller patches to train the model effectively on large images, and the final model is evaluated using custom metrics.

---

## Key Features

1. **Dataset Preprocessing**:
   - Images and masks are processed into fixed-size patches using the `patchify` library.
   - Pixel values are scaled using MinMaxScaler for better model performance.
   - Labels are converted from RGB format to class indices using a mapping of land cover types.

2. **U-Net Architecture**:
   - A custom U-Net implementation is used, which features:
     - Contraction path with convolutional and max-pooling layers.
     - Expansive path with transposed convolutions and skip connections.
     - Softmax activation for multi-class segmentation.

3. **Custom Loss Functions**:
   - **Dice Loss**: Encourages overlap between predicted and true segmentations.
   - **Focal Loss**: Mitigates class imbalance by focusing on harder-to-classify pixels.
   - Combined as `Total Loss`.

4. **Performance Metrics**:
   - **Jaccard Coefficient (IoU)**: Measures overlap accuracy.
   - **Accuracy**: Pixel-wise correctness of predictions.

5. **Visualization**:
   - Random samples of images and masks can be visualized to confirm data integrity.

---

## Thought Process

1. **Data Preparation**:
   - Satellite images are high-resolution; patching ensures memory efficiency.
   - RGB masks are converted to integer class labels for categorical segmentation.

2. **Model Design**:
   - U-Net was selected for its proven success in biomedical and geospatial segmentation tasks.
   - Skip connections in U-Net help preserve spatial information.

3. **Custom Loss Functions**:
   - Dice loss handles imbalanced datasets effectively.
   - Focal loss ensures the model focuses on difficult pixels.

4. **Class Weights**:
   - Weights are applied to address the imbalance in class distribution.

5. **Evaluation**:
   - IoU (Jaccard Coefficient) was chosen as a primary metric as it directly measures segmentation quality.

---

## How to Run the Code

1. **Setup Environment**:
   - Install the required libraries:
     ```bash
     pip install tensorflow numpy matplotlib scikit-learn patchify segmentation-models-pytorch
     ```

2. **Prepare Dataset**:
   - Place the dataset in the directory structure:
     ```
     root_directory/
     ├── images/
     └── masks/
     ```

3. **Run Training**:
   - Execute the `main.py` file to train the U-Net model:
     ```bash
     python main.py
     ```

4. **Model Summary**:
   - The U-Net model architecture is displayed using Keras's `model.summary()`.

---

## Results

- The trained model can accurately predict segmentation masks for satellite imagery.
- Performance metrics such as IoU and accuracy are logged during training.

---

## File Structure

- **`main.py`**: Contains the full implementation, including data preparation, model creation, training, and evaluation.
- **`simple_multi_unet_model.py`** (Optional): Provides additional U-Net functionality if required.

---

## Dependencies

- TensorFlow 2.x
- NumPy
- OpenCV
- Patchify
- Scikit-learn
- Matplotlib
- Segmentation Models (PyTorch)

---

## Notes

- Ensure sufficient system memory and GPU support for faster training.
- Adjust the `patch_size` and `batch_size` to fit your hardware capabilities.
- Consider fine-tuning the model for specific datasets or use cases.

---

## Future Work

- Integrate data augmentation for improved generalization.
- Experiment with advanced architectures like DeepLabv3+ or transformers for segmentation.
- Deploy the trained model using frameworks like TensorFlow Serving.

---

## Acknowledgments

- The U-Net architecture was inspired by the original U-Net paper for biomedical segmentation.
- Thanks to the open-source libraries that make this project possible.
