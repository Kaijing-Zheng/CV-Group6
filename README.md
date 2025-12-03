# CV-Group6


**Potential Data processing techniques**
1. Image rotation-Increments of 90 degrees (3,000 images per letter=750 images per rotation)
2. Zooming in and out of images
3. Potentially change lighting. However, the dataset already contains varying lighting for images within each letter.
4. Downsampling or data augmentation may not be needed due to the dataset containing 3,000 images per letter.

**Current Tasks**
1. Run MLP on training, validation, and testing dataset after mediapipe processing
2. Run LSTM on same datasets-Compare performance with MLP
3. If given time, fine-tune hyperparameters for both models using optuna/grid search

**Potential Things to Try**
1. Process training dataset by using edge detection, grayscale and removing border, split into training: validation in 8:2 ratio
2. Run CNN model directly on processed data without using mediapipe (ensure the images are rotated during training process)
3. Evaluate performance of CNN compared to MLP on mediapipe data
