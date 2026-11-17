# 📩 Spam Message Classification Using Naive Bayes Classifier

An end-to-end, beginner-friendly Machine Learning project that classifies SMS and text messages as **SPAM** or **NOT SPAM (HAM)** using **TF-IDF Vectorization**, the **Multinomial Naive Bayes Classifier**, and an interactive **Streamlit** web application.

---

## 📌 1. Project Abstract & Objectives

### Abstract
Mobile messaging is one of the most widely used forms of digital communication. However, the rise of unsolicited spam messages, phishing scams, and marketing promotions threatens user privacy and digital security. This project implements a lightweight, highly accurate Machine Learning model using the **Multinomial Naive Bayes** algorithm to automatically detect spam text messages. Text messages are vectorized using **TF-IDF (Term Frequency-Inverse Document Frequency)** and evaluated using standard metrics such as Accuracy, Precision, Recall, and Confusion Matrix. An interactive **Streamlit** web interface allows users to input any message, click sample messages, and receive real-time classification results with confidence scores.

### Objectives
1. **Accurate Classification:** Classify input messages into **Spam** (unsolicited/scam) and **Ham** (legitimate/not spam).
2. **High Precision:** Prioritize precision to ensure legitimate messages are not mistakenly flagged as spam (avoiding false alarms).
3. **Interactive UI:** Provide an attractive, clean, and intuitive web application built with Streamlit.
4. **Beginner-Friendly:** Keep the codebase modular, clean, well-commented, and easy to explain during academic viva or project presentations.

---

## 🗂️ 2. Project Folder Structure

```text
spam_classifier/
│
├── spam.csv            # SMS Spam Collection dataset (5,572 rows of ham & spam)
├── train_model.py      # Python script: loads data, trains Naive Bayes, saves artifacts
├── app.py              # Streamlit web application: interactive UI & sample testing
├── model.pkl           # Saved trained Multinomial Naive Bayes model
├── vectorizer.pkl      # Saved fitted TF-IDF Vectorizer
├── metrics.pkl         # Saved evaluation metrics (accuracy, confusion matrix, etc.)
├── requirements.txt    # List of required Python packages
└── README.md           # Comprehensive project documentation & Viva Q&A
```

---

## ⚙️ 3. Technologies & Libraries Used

| Technology / Library | Purpose in this Project |
| :--- | :--- |
| **Python 3.10+** | Core programming language |
| **Pandas** | Loading and structuring the SMS dataset (`spam.csv`) |
| **NumPy** | Numerical data operations and matrix manipulation |
| **Scikit-learn** | Splitting data, TF-IDF vectorization, MultinomialNB classifier, and metrics |
| **Streamlit** | Building the interactive frontend web dashboard |
| **Pickle** | Serializing (saving) and deserializing (loading) the model and vectorizer |

---

## 🚀 4. Step-by-Step Instructions to Run in VS Code

### Step 1: Open VS Code
1. Launch **Visual Studio Code**.
2. Go to **File -> Open Folder...** and select the project folder:
   ```text
   c:\Users\HP\OneDrive\Desktop\hackalgo\spam_classifier
   ```

### Step 2: Open Terminal
- Press `Ctrl + ~` (or go to **Terminal -> New Terminal** in the top menu).

### Step 3: (Recommended) Create & Activate Virtual Environment
```bash
# Create virtual environment named 'venv'
python -m venv venv

# Activate on Windows (Command Prompt):
venv\Scripts\activate.bat

# OR Activate on Windows (PowerShell):
venv\Scripts\Activate.ps1
```

### Step 4: Install Required Packages
Run the following command in the terminal to install all dependencies:
```bash
pip install -r requirements.txt
```
*(Alternative single command: `pip install pandas numpy scikit-learn streamlit matplotlib`)*

### Step 5: Train the Model
Run the model training script to train the classifier and generate `model.pkl` and `vectorizer.pkl`:
```bash
python train_model.py
```
**Expected Output:**
```text
[INFO] Loading dataset from: .../spam.csv
[INFO] Total rows in dataset: 5572
[INFO] Class Distribution: Ham (Not Spam) = 4825, Spam = 747
[INFO] Training samples: 4457, Testing samples: 1115
[INFO] Extracting features with TF-IDF Vectorizer...
[INFO] Training Multinomial Naive Bayes classifier...
==================================================
                MODEL EVALUATION RESULTS
==================================================
Accuracy  : 97.67%
Precision : 98.55%
Recall    : 84.47%
F1-Score  : 90.97%
==================================================
[INFO] Model successfully saved to: .../model.pkl
[INFO] Vectorizer successfully saved to: .../vectorizer.pkl
[SUCCESS] Model training and export completed successfully!
```

### Step 6: Launch the Streamlit Web Application
Run the Streamlit application:
```bash
streamlit run app.py
```
Your default browser will automatically open:
```text
Local URL: http://localhost:8501
Network URL: http://...:8501
```

---

## 💻 5. Working Process & Workflow Explanation

The spam classification pipeline follows 5 sequential stages:

```
[ Raw SMS Data (spam.csv) ]
             │
             ▼
[ Data Preprocessing (Drop NAs, Binary Label Encoding: Ham=0, Spam=1) ]
             │
             ▼
[ Feature Extraction via TF-IDF Vectorizer (Tokenize, Lowercase, Remove Stopwords) ]
             │
             ▼
[ Model Training via Multinomial Naive Bayes (Calculate Word Probabilities) ]
             │
             ▼
[ Model Evaluation (Accuracy ~97.7%, Precision ~98.5%, Confusion Matrix) ]
             │
             ▼
[ Streamlit Web UI (Interactive Message Input -> Real-time SPAM / HAM Prediction) ]
```

### 1. Data Ingestion & Preprocessing
- The dataset `spam.csv` consists of 5,572 messages labeled either `ham` (legitimate message) or `spam`.
- Labels are converted into binary numerical values: `ham` $\rightarrow 0$, `spam` $\rightarrow 1$.

### 2. Text Feature Extraction (TF-IDF Vectorizer)
Computers cannot interpret sentences directly. We use **TF-IDF**:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
- **TF (Term Frequency):** Frequency of word $t$ in message $d$.
- **IDF (Inverse Document Frequency):** Logarithmic measure of how rare word $t$ is across all messages $D$. Rare keywords like *"claim"*, *"winner"*, *"cash"* receive high weight, whereas common filler words (*"is"*, *"the"*) are eliminated via English stopwords.

### 3. Model Training (Multinomial Naive Bayes)
- Uses Bayes' Theorem with the assumption that features (words) are conditionally independent given the class:
$$P(C \mid X) = \frac{P(X \mid C) \cdot P(C)}{P(X)}$$
- Where:
  - $P(C \mid X)$ is the posterior probability of class $C$ (Spam or Ham) given message $X$.
  - $P(X \mid C)$ is the likelihood of message words appearing in class $C$.
  - $P(C)$ is the prior probability of class $C$.
  - $P(X)$ is the evidence probability of message $X$.

### 4. Classification & Prediction
- The model computes both $P(\text{Spam} \mid \text{Message})$ and $P(\text{Ham} \mid \text{Message})$.
- Whichever probability is higher determines the final prediction.

---

## 🧪 6. Sample Inputs and Expected Outputs

| # | Sample Input Message | Predicted Class | Confidence Score | UI Display |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341.` | **SPAM** | ~99.4% Spam | 🚨 Red Box ("SPAM DETECTED") |
| **2** | `URGENT! You have won a 1 week FREE membership in our £100,000 Prize Draw! Call 09058094597 now to claim.` | **SPAM** | ~98.9% Spam | 🚨 Red Box ("SPAM DETECTED") |
| **3** | `Hey, are we still meeting for lunch at 1 PM today? Let me know!` | **NOT SPAM (HAM)** | ~99.2% Ham | ✅ Green Box ("NOT SPAM (HAM)") |
| **4** | `Can you please send me the class notes from yesterday's lecture? Thanks a lot!` | **NOT SPAM (HAM)** | ~98.8% Ham | ✅ Green Box ("NOT SPAM (HAM)") |

---

## ⚖️ 7. Advantages, Limitations & Future Enhancements

### Advantages
1. **Blazing Fast Performance:** Naive Bayes has low computational complexity $O(n)$, allowing real-time prediction in milliseconds.
2. **High Precision:** Minimizes false positives, ensuring critical legitimate emails/SMS are not misrouted to spam.
3. **Small Memory Footprint:** The trained model and vectorizer files are only a few megabytes in size.
4. **Works Well on Small & Medium Datasets:** Does not require heavy GPU hardware or days of deep neural network training.

### Limitations
1. **Independence Assumption:** In human language, words often depend on surrounding words (e.g., *"not bad"* means *"good"*). Naive Bayes assumes all words are mutually independent.
2. **Zero Frequency Problem:** If a new word never appeared in the training dataset, its probability becomes zero (mitigated using Laplace smoothing `alpha=1.0`).
3. **Susceptible to Adversarial Evasion:** Spammers who intentionally misspell words (e.g., *"W!NNER"*, *"fr33"*) can bypass basic word-level tokenization.

### Future Enhancements
1. **N-Gram Support:** Incorporating Bi-grams and Tri-grams (e.g., *"click here"*, *"claim prize"*) into the vectorizer.
2. **Deep Learning / Transformer Models:** Comparing results with BERT / DistilBERT or LSTM networks.
3. **SMS Gateway / Webhook Integration:** Deploying the Streamlit app on Streamlit Cloud and connecting to Twilio API to filter live incoming SMS.
4. **Multi-language Support:** Extending stopwords and tokenization to languages beyond English.

---

## 🎓 8. Comprehensive Viva / Interview Questions & Answers

### Q1: What is the main objective of this project?
**Answer:** The objective is to build a machine learning pipeline and interactive web interface that classifies SMS messages as either Spam (promotional/fraudulent) or Ham (legitimate) with high accuracy and precision using Multinomial Naive Bayes.

### Q2: Why did you choose Multinomial Naive Bayes instead of Gaussian Naive Bayes?
**Answer:** Gaussian Naive Bayes assumes features follow a continuous normal (bell-shaped) distribution. However, text data vectorized with word counts or TF-IDF values represents discrete frequency-based features. Multinomial Naive Bayes is mathematically designed for discrete count/frequency distributions, making it the standard choice for text classification.

### Q3: What is the formula for Bayes' Theorem?
**Answer:**
$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$
- $P(A \mid B)$: Posterior probability of hypothesis $A$ given evidence $B$.
- $P(B \mid A)$: Likelihood of evidence $B$ given hypothesis $A$.
- $P(A)$: Prior probability of hypothesis $A$.
- $P(B)$: Marginal probability of evidence $B$.

### Q4: Why is it called "Naive"?
**Answer:** It is called "naive" because it makes the strong (and naive) simplifying assumption that all input features (words) are strictly independent of each other given the class label. Even though words in natural language have dependencies, this assumption drastically reduces computation while maintaining high practical accuracy.

### Q5: What is TF-IDF and why do we use it?
**Answer:** TF-IDF stands for **Term Frequency - Inverse Document Frequency**.
- **TF** measures how frequently a term occurs in a message.
- **IDF** measures how rare or unique the term is across the entire dataset.
Common words like *"the"*, *"is"*, *"and"* appear everywhere and get low TF-IDF scores, while domain-specific terms like *"urgent"*, *"lottery"*, *"claim"* get high TF-IDF scores, making them strong indicators for classification.

### Q6: What is Laplace Smoothing (the `alpha` parameter)?
**Answer:** Laplace smoothing (default `alpha=1.0` in `MultinomialNB`) adds a small constant to word count probabilities. This prevents zero-probability errors when the classifier encounters a word during testing that was never seen during training.

### Q7: Why is Precision more critical than Recall in spam classification?
**Answer:** In spam classification:
- A **False Positive** means a genuine, important message (like an interview invite, OTP, or doctor's note) is wrongly classified as Spam and missed.
- A **False Negative** simply means a junk message reaches the inbox.
Because missing a crucial message is far worse than seeing an occasional spam SMS, we prioritize **high Precision** to minimize False Positives.

### Q8: What dataset was used and what is the class ratio?
**Answer:** The dataset is the UCI SMS Spam Collection dataset containing 5,572 English messages. It consists of roughly 86.6% Ham (4,825 messages) and 13.4% Spam (747 messages), representing a realistic imbalanced distribution.

### Q9: How did you handle class imbalance in the train-test split?
**Answer:** We used `stratify=y` inside `train_test_split()`. This ensures both the training and test sets have the exact same proportion of Ham and Spam (~86.6% to ~13.4%), preventing split bias.

### Q10: What is a Confusion Matrix?
**Answer:** A $2 \times 2$ table that compares actual labels with predicted labels:
- **True Positive (TP):** Spam correctly predicted as Spam.
- **True Negative (TN):** Ham correctly predicted as Ham.
- **False Positive (FP):** Ham mistakenly predicted as Spam (Type I error).
- **False Negative (FN):** Spam missed and predicted as Ham (Type II error).

### Q11: What is the F1-Score?
**Answer:** The F1-score is the harmonic mean of Precision and Recall:
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
It provides a single balanced score, especially useful on imbalanced datasets.

### Q12: How are the trained model and vectorizer saved and reused?
**Answer:** We use Python's built-in `pickle` library to serialize the trained `MultinomialNB` model into `model.pkl` and the fitted `TfidfVectorizer` into `vectorizer.pkl`. When `app.py` runs, it deserializes both files instantly without needing to retrain every time.

### Q13: What happens if an input message is empty?
**Answer:** The Streamlit app includes input validation (`clean_text = user_text.strip()`). If empty, it displays an alert warning: *"Please enter a message before clicking Predict!"* to prevent unnecessary model evaluation.

### Q14: How does Streamlit connect the model to the web interface?
**Answer:** When the user types a message and clicks **Predict**:
1. The raw text is passed to `vectorizer.transform([text])` to generate a numerical vector.
2. The vector is passed to `model.predict()` to obtain the class (0 or 1).
3. Probabilities are obtained using `model.predict_proba()` to compute percentage confidence.
4. Streamlit renders the outcome dynamically in custom HTML/CSS (Green for Ham, Red for Spam).

---

## 👨‍💻 9. Author & License
- Project: SMS Spam Message Classifier
- Technology: Python, Scikit-learn, Streamlit
- License: Open Source (MIT)
