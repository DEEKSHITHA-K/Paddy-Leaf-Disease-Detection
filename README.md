# Identification of Paddy Leaf Disease and Measuring Affected Area

This project implements an automated system to identify various diseases in paddy (rice) leaves and quantify the physical area of the leaf affected by the infection. It uses a hybrid approach of **VGG16 for feature extraction** and traditional machine learning classifiers, alongside **OpenCV** for image processing.

## 📊 Dataset

The dataset used for this project is sourced from **Kaggle**.

* **Dataset Link:** [Rice Leaf Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases) 
* **Enhanced Dataset used**: [Download Dataset Here](https://drive.google.com/drive/folders/1Rd_pCpp-T4QI8Nrs13_5vtP14swluVBW?usp=sharing)
* **Description:** A collection of 1500 leaf images across four classes, including healthy leaves and common diseases, captured in various environmental conditions.

---

## 🦠 Diseases Detected

The system is trained to identify the following categories:

* **Bacterial Leaf Blight** 
* **Brown Spot** 
* **Leaf Smut** 
* **Narrow Brown Spot** 
* **Rice Blast** 
* **Healthy** (Negative) 

---

## 🛠️ Technical Stack
* **Language:** Python 
* **Feature Extraction:** VGG16 (Pre-trained CNN) 
* **Machine Learning:** Scikit-Learn (SVM, KNN, Random Forest) 
* **Image Processing:** OpenCV (cv2) 
* **IDE:** Google Colab / Jupyter Notebook 

---

## 📂 Methodology

1. **Data Preprocessing:** Images are processed using `cv2` and augmented using `ImageDataGenerator` to increase model robustness against rotation, shifts, zooms, flips, and lighting variations.
2. **Feature Extraction:** The convolutional base of the **VGG16** model (pre-trained on ImageNet) is used to extract high-level spatial features from leaf images, which are then flattened into feature vectors and fed into the ML classifiers..
3. **Classification:** Traditional ML algorithms are used to classify the features. **SVM** achieved the highest accuracy in this project.
   
* **SVM Accuracy:** 87.5% 
* **Random Forest Accuracy:** 83.3% 
* **KNN Accuracy:** 72.9% 

4. **Area Measurement Logic**
After classification, the system isolates the diseased part of the leaf:
* **Contour Detection:** Identifies the boundaries of the infected spots.
* **Pixel resolution in mm/pixel:** Uses a resolution constant (0.1 mm/pixel).
* **Formula:** $$\text{Affected Area } (mm^2) = (\text{Length} \times \text{Width}) \times \text{Pixel Resolution}$$
* **Severity:** Calculated as `(Affected Area / Total Leaf Area) * 100`.

---

## 💻 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/deekshitha-k/paddy-leaf-disease-detection.git
    ```

2.  **Install dependencies:**
    ```bash
    pip install opencv-python tensorflow numpy matplotlib scikit-learn
    ```

3.  **Run the Project:**
    * Load the `Copy_of_MP_Phase2.ipynb` notebook in Google Colab or Jupyter.
    * Ensure the dataset is organized in folders named by disease category.
    * Execute the cells to train the models and visualize the area measurement results.

---

## 📈 Results and Outputs

### 1. Multi-Model Prediction Results
The system classifies the disease and provides instant remedies for farmers.


| **SVM Prediction** | **KNN Prediction** |
| :---: |  :---: | 
| <img width="295" height="163" alt="Screenshot 2026-01-03 203430" src="https://github.com/user-attachments/assets/67968fa4-a9ee-404e-a88b-5d944c8fb79a" /> | <img width="634" height="190" alt="Screenshot 2026-01-03 203439" src="https://github.com/user-attachments/assets/9e3e1036-76f0-4a2f-babb-777ca45d7f53" /> |



| **Healthy Leaf Detection** | **Random Forest Prediction** |
| :---: | :---: |
| <img width="203" height="118" alt="Screenshot 2026-01-03 203916" src="https://github.com/user-attachments/assets/0a5ce738-407a-4706-897f-12e3c7fc819d" /> |<img width="748" height="193" alt="Screenshot 2026-01-03 203458" src="https://github.com/user-attachments/assets/7c21c3d1-1871-4b3c-b32b-e698640aa682" /> | 


| **Area Intensity Detection for Diseased Leaf** | **Area Intensity Detection for Healthy Leaf** |
| :---: | :---: |
| <img width="340" height="121" alt="Screenshot 2026-01-03 203508" src="https://github.com/user-attachments/assets/1687d9c2-2f41-43ca-aafe-fd02e7b70f08" /> | <img width="294" height="184" alt="Screenshot 2026-01-03 204454" src="https://github.com/user-attachments/assets/ef05d95f-76cb-4ebc-a378-fef582d371ee" /> |


---

## 👥 Team Members (Team A1)

* **Narla Sathvika** (20WH1A0528)
* **Kashetty Deekshitha** (20WH1A0557)  
* **Peddi Vahnika** (20WH1A0554) 
* **Project Guide:** Dr. Nara Sreekanth (Associate Professor) 

---

## 📜 Acknowledgments
Developed at **BVRIT HYDERABAD College of Engineering for Women** as part of the B.Tech CSE.
