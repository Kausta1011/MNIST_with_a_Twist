MNIST with a Twist: The Accuracy Paradox & Advanced Evaluation

🎯 Project Goal
This project uses the MNIST dataset to demonstrate a critical lesson in Data Science: Why "Accuracy" is a dangerous metric for imbalanced classification.

Instead of simply building a model to recognize digits, this notebook constructs a narrative that exposes the flaws in standard evaluation metrics and implements a robust framework using Precision-Recall trade-offs and ROC Analysis to tune model performance for specific business constraints.

🧠 The "Twist" (Key Insights)
The core insight of this project is the Accuracy Paradox.

The Trap: The notebook demonstrates that a "Dummy Classifier" (which blindly predicts "False" for every image) achieves 90% Accuracy. This proves that on skewed datasets (where the target class is only 10%), accuracy is a misleading indicator of success.

The Solution: The project pivots from accuracy to Confusion Matrix analysis, proving that we must optimize for Precision (trustworthiness of positive predictions) or Recall (ability to find all positive instances) depending on the use case.

⚙️ Technical Workflow
Data Acquisition: Loading the MNIST dataset (70,000 images) via Scikit-Learn.

Binary Target Transformation: Converting the multi-class problem into a binary "5 vs Non-5" problem to create class imbalance.

Baseline Establishment: Training a Stochastic Gradient Descent (SGDClassifier) model.

The "Dumb" Benchmark: Implementing a DummyClassifier to statistically prove the irrelevance of the "Accuracy" metric.

Advanced Metric Evaluation:

Confusion Matrix: extracting True Positives/Negatives and False Positives/Negatives.

Precision & Recall: Calculating exact performance scores.

F1 Score: Computing the harmonic mean for a single performance metric.

Threshold Tuning (Decision Boundary Optimization):

Accessing raw decision scores via decision_function.

Visualizing the Precision-Recall Curve to identify the inflection point where performance degrades.

Implementing a dynamic threshold (e.g., threshold = 3000) to guarantee 90% Precision, prioritizing prediction confidence over coverage.

ROC Analysis: Plotting the Receiver Operating Characteristic (ROC) curve to visualize the True Positive Rate vs. False Positive Rate trade-off.

📊 Results & Performance
Baseline Accuracy: ~95% (Misleadingly high).

Dummy Accuracy: ~90% (Proves the baseline was mostly noise).

Optimized Performance:

Identified a decision threshold of ~3370 to achieve 90% Precision.

Demonstrated that raising the threshold reduces False Positives (higher precision) but increases False Negatives (lower recall), allowing the model to be tuned for "high-stakes" scenarios where false alarms are unacceptable.

🛠 Libraries Used
Scikit-Learn: SGDClassifier, DummyClassifier, cross_val_score, confusion_matrix, precision_recall_curve, roc_curve.

Matplotlib: Custom visualization of decision boundaries and metric curves.

NumPy: Efficient array manipulation for image data.
