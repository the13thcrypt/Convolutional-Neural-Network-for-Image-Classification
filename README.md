# PyTorch CNN for CIFAR-10 Image Classification

This project is a complete, end-to-end example of building a simple Convolutional Neural Network (CNN) using PyTorch. The model is trained to classify images from the CIFAR-10 dataset, which consists of 10 classes (e.g., 'plane', 'car', 'bird').

The Jupyter Notebook walks through the entire process, from data loading to final inference on new images.

## Features

  * **Data Loading:** Uses `torchvision` to download, transform, and load the CIFAR-10 dataset.
  * **Custom CNN:** Defines a simple CNN from scratch using `torch.nn.Module`.
  * **Training Loop:** A complete training loop using `SGD` with momentum and `CrossEntropyLoss`.
  * **Save/Load:** Demonstrates how to save the trained model's weights (`.pth` file) and load them for later use.
  * **Evaluation:** Calculates the model's accuracy on the unseen test dataset.
  * **Inference:** Includes a helper function to load, transform, and predict the class of new, external images.

## Project Workflow

1.  **Imports & Setup:** All necessary libraries from `torch`, `torchvision`, and `PIL` are imported.
2.  **Data Preparation:**
      * The CIFAR-10 dataset is loaded for both training and testing.
      * A `transform` is defined to convert images to Tensors and normalize them.
      * `DataLoader`s are created to feed data to the network in batches.
3.  **Model Definition (`NeuralNet`):**
    The network architecture is a simple CNN:
      * `Conv2d` (3 in-channels, 12 out-channels, 5x5 kernel) -\> `ReLU` -\> `MaxPool2d`
      * `Conv2d` (12 in-channels, 24 out-channels, 5x5 kernel) -\> `ReLU` -\> `MaxPool2d`
      * `Flatten`
      * `Linear` (24 \* 5 \* 5 -\> 120) -\> `ReLU`
      * `Linear` (120 -\> 84) -\> `ReLU`
      * `Linear` (84 -\> 10) (Output layer for 10 classes)
4.  **Training:**
      * The model is trained for 30 epochs.
      * The optimizer (`SGD`) and loss function (`CrossEntropyLoss`) are defined.
      * The training loop iterates through the `train_loader`, performs the forward pass, calculates loss, backpropagates, and updates the weights.
5.  **Saving & Loading:** The trained model's `state_dict` is saved to `trained_net.pth` and then re-loaded into a new instance of the model.
6.  **Evaluation:** The model is set to `eval()` mode and its accuracy is calculated on the `test_loader`.
7.  **Inference on Custom Images:**
      * A function `load_image` is provided to open a local image, resize it to 32x32, apply the same normalization transform, and predict its class using the trained model.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
2.  **Install dependencies:**
    You'll need Python and the following libraries. You can install them using `pip`:
    ```bash
    pip install torch torchvision numpy pillow jupyterlab
    ```
3.  **Run the notebook:**
    Start Jupyter Lab and open the `.ipynb` notebook file.
    ```bash
    jupyter lab
    ```
4.  **Run all cells:** You can run all cells to train the model, save it, and evaluate it.
5.  **(Optional) Test your own images:**
      * Add your own `.jpg` or `.png` images to the project directory.
      * In the last cell, change the paths in the `image_paths` list to point to your images.
      * Run the cell to see the model's predictions.

## Results

After training, the model achieves an accuracy of approximately **70%** on the test set. This demonstrates a successful end-to-end pipeline, showing the model's ability to learn and generalize features from the CIFAR-10 dataset effectively. Further tuning of hyperparameters, architecture modifications, or increased training epochs could potentially improve this result even more.
