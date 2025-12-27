# Identification of Paddy Leaf Disease and Measuring Affected Area

This project implements an automated system to identify various diseases in paddy (rice) leaves and quantify the physical area of the leaf affected by the infection. It uses a hybrid approach of **VGG16 for feature extraction** and traditional machine learning classifiers, alongside **OpenCV** for image processing.

## 📊 Dataset

The dataset used for this project is sourced from **Kaggle**.

* **Dataset Link:** [Rice Leaf Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases) 


* **Description:** A collection of 1500 leaf images across four classes, including healthy leaves and common diseases, captured in various environmental conditions.

## 🦠 Diseases Detected

The system is trained to identify the following categories:

* **Bacterial Leaf Blight** 
* **Brown Spot** 
* **Leaf Smut** 
* **Narrow Brown Spot** 
* **Rice Blast** 
* **Healthy** (Negative) 


## 🛠️ Technical Stack
* **Language:** Python 
* **Feature Extraction:** VGG16 (Pre-trained CNN) 
* **Machine Learning:** Scikit-Learn (SVM, KNN, Random Forest) 
* **Image Processing:** OpenCV (cv2) 
* **IDE:** Google Colab / Jupyter Notebook 

## 📂 Methodology

1. **Data Preprocessing:** Images are resized to  pixels and normalized to a range of  using Scikit-Learn.
2. **Feature Extraction:** The convolutional base of the **VGG16** model is used to extract high-level spatial features, which are then flattened into feature vectors.
3. **Classification:** Traditional ML algorithms are used to classify the features. **SVM** achieved the highest accuracy in this project.
   
* **SVM Accuracy:** 87.5% 
* **Random Forest Accuracy:** 83.3% 
* **KNN Accuracy:** 72.9% 

4. **Area Measurement:** The system uses **OpenCV contour detection** to isolate infected spots. Using a defined pixel resolution (), it calculates the affected area in .


## 💻 Usage
1. Clone the repository.
2. Install dependencies: `pip install opencv-python tensorflow numpy scikit-learn matplotlib`.
3. Run the Jupyter notebook `MP_Phase2.ipynb` to train the models and run predictions on test images.



## 👥 Team Members (Team A1)

* **Narla Sathvika** (20WH1A0528) 
* **Peddi Vahnika** (20WH1A0554) 
* **Kashetty Deekshitha** (20WH1A0557) 
* **Project Guide:** Dr. Nara Sreekanth (Associate Professor) 

## 📜 Acknowledgments
Developed at **BVRIT HYDERABAD College of Engineering for Women** as part of the B.Tech CSE.
