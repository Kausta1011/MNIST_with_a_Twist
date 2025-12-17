# MNIST with a Twist: Binary Classification & Metric Analysis

## 📌 Project Overview
This project explores the famous **MNIST dataset** (70,000 images of handwritten digits) with a specific focus on **Binary Classification**. Instead of classifying all 10 digits, we isolate the digit **'5'** to create a "5-Detector".

The "Twist" in this project is the demonstration of the **Accuracy Paradox**—showing why "Accuracy" is a dangerous metric for imbalanced datasets and how to use advanced metrics like Precision, Recall, and ROC Curves to build a truly reliable model.

## 🔑 Key Insights ("The Twist")
* **The Accuracy Trap:**
    * A "Dummy Classifier" that blindly predicts "Not 5" for every image achieves **~90% Accuracy**.
    * This proves that accuracy is a misleading metric when the target class represents only 10% of the data.
* **Precision vs. Recall Trade-off:**
    * The project demonstrates how shifting the **Decision Threshold** impacts model performance.
    * **Insight:** Raising the threshold reduces False Positives (High Precision) but increases False Negatives (Lower Recall).
* **Optimization:**
    * We successfully identified a specific decision threshold (approx. **3370**) to guarantee **90% Precision**, making the model highly reliable for positive predictions.

## 🛠️ Technologies & Libraries
* **Python 3.x**
* **Scikit-Learn:** `SGDClassifier`, `DummyClassifier`, `cross_val_score`, `confusion_matrix`, `precision_recall_curve`, `roc_curve`
* **Matplotlib:** For visualizing decision boundaries and metric curves
* **NumPy:** For array manipulation

## 📊 Methodology
1.  **Data Preparation:** * Fetched MNIST data via `fetch_openml`.
    * Split data into Training (60k) and Test (10k) sets.
    * Created boolean target vectors (`y_train_5`) to convert the problem into binary classification.
2.  **Model Training:**
    * Trained a **Stochastic Gradient Descent (SGD)** classifier.
    * Implemented Stratified K-Fold Cross-Validation.
3.  **Evaluation:**
    * Compared SGD performance against a baseline `DummyClassifier`.
    * Generated a **Confusion Matrix** to visualize True Positives vs. False Negatives.
    * Calculated **F1 Score** (harmonic mean of precision and recall).
4.  **Visualizations:**
    * Plotted **Precision-Recall Curves** to visualize trade-offs.
    * Plotted **ROC Curves** to analyze True Positive Rate vs. False Positive Rate.

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy matplotlib scikit-learn
    ```
3.  **Run the Notebook:**
    Launch Jupyter Notebook and open `MNIST_with_a_Twist.ipynb`.
    ```bash
    jupyter notebook
    ```

## 📈 Results Snapshot
| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Baseline Accuracy** | ~90% | Achieved by "Dumb" classifier (useless) |
| **SGD Accuracy** | ~95% | Looks good, but needs context |
| **Optimized Precision** | **90%** | Achieved by tuning threshold to >3000 |

---
*This project serves as a practical guide to handling imbalanced classification problems in Machine Learning.*
