# Cat-vs-Dog-Image-Classifier
This project is a PyTorch-based convolutional neural network (CNN) to classify images as cats or dogs. It includes training code, a single image prediction script, and a simple web app with Gradio for interactive classification.

---

## 1. Setup & Installation

- Make sure you have **Python 3.8+** installed.

- Install required packages:
  
pip install torch torchvision scikit-learn pillow gradio

Or install all dependencies using:
    pip install -r requirements.txt

## 2. Prepare Dataset

- Create a folder named `train/` in your project directory.
- Inside `train/`, create two subfolders: `cats/` and `dogs/`.
- Place cat images in `train/cats/` and dog images in `train/dogs/`.
- The code automatically filters non-RGB images and crops images to square shapes before resizing them.

---

## 2. Install Dependencies

Make sure you have Python 3.7 or higher installed.  
Then install the required packages:

pip install torch torchvision scikit-learn pillow gradio


## 3. Train the Model
Run the train.py script to:

Load and preprocess the dataset with cropping and normalization.

Filter only the specified classes (cats and dogs).

Split data into training and testing sets with stratification.

Train a CNN model with three convolutional layers, pooling, and fully connected layers.

Print training loss per epoch and test accuracy.

Save the trained model weights as cat_dog_model.pth.

  python train.py
    
## 4. Test a Single Image
Use test_img.py to predict a single image’s class with the trained model:
    python test_img.py path/to/image.jpg

## 5. Launch Web App for Interactive Predictions
Run the app.py script to start a Gradio web interface for uploading images and getting predictions in real time:
    python app.py


## Summary
Organize your images properly.

Install all dependencies.

Train your model with train.py.

Test images via command line using test_img.py.

Use the web app app.py for easy interactive classification.


