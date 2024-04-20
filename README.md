# ANN---Project-on-Handwritten-Digits-Detection-

This project utilizes Artificial Neural Networks (ANNs) to detect handwritten digits. The provided Jupyter notebook (`ANN - Project on Handwritten Digits Detection .ipynb`) contains the code for building and training the model.

### Libraries Used
- **TensorFlow:** An open-source machine learning framework.
- **Keras:** A high-level neural networks API, running on top of TensorFlow.
- **NumPy:** A library for numerical computing in Python.
- **Matplotlib:** A plotting library for creating visualizations in Python.
- **Seaborn:** A statistical data visualization library based on Matplotlib.

### Dataset
The project uses the MNIST dataset, which is a large database of handwritten digits. It consists of 28x28 pixel grayscale images of handwritten digits (0 through 9) along with their corresponding labels.

### Implementation Overview
1. **Data Loading:** The MNIST dataset is loaded using Keras.

2. **Data Preprocessing:** 
    - The pixel values of the images are normalized to the range [0, 1].
    - The images are flattened from 2D arrays (28x28) to 1D arrays (784).

3. **Model Architecture:**
    - The model is defined as a sequential neural network with two dense layers.
    - Each dense layer consists of 100 units and uses the sigmoid activation function.

4. **Model Compilation:**
    - The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function.
    - Accuracy is chosen as the evaluation metric.

5. **Model Training:**
    - The model is trained on the training dataset for 5 epochs.

6. **Model Evaluation:**
    - The model is evaluated on the test dataset to assess its performance.

7. **Results Visualization:**
    - Confusion matrix is generated to visualize the model's performance on each class.

8. **Model Export:**
    - The trained model is saved using the pickle library for future use.

### File Structure
- **ANN - Project on Handwritten Digits Detection .ipynb:** Jupyter notebook containing the code.
- **model_pkl:** Pickle file containing the trained model.
