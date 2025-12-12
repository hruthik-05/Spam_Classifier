# 📘 **Spam Classifier Using Custom Transformer Architecture**

 I implemented a **Transformer architecture from scratch** using PyTorch and trained a **binary classification head** to detect spam vs. non-spam messages.

We also built a simple **HTML UI** and connected it with a **FastAPI backend** to get real-time predictions locally.

---

## 🚀 **Project Highlights**

* Implemented Transformer components from scratch:

  * Multi-Head Attention
  * Layer Normalization
  * Feed Forward Network
  * Positional Embeddings
  * Transformer Blocks

* Trained a **custom binary classifier** on top of the transformer outputs.

* Built a **FastAPI** inference server to serve predictions.

* Frontend UI created using **HTML + CSS + JavaScript**.

* Local end-to-end pipeline working with real-time message classification.

---

## 🧠 **Architecture Overview**

The model follows a GPT-style architecture, including:

* Token Embedding Layer
* Positional Embedding Layer
* Dropout + Normalization
* Multi-layer Transformer stack
* Final Linear Head → Outputs **spam** / **not spam**

The spam classification decision is made using the **last token’s logits**.

---

## 🖥️ **Project Structure**

```
project/
│── llmfromscratch.py        # Transformer architecture + model code
│── index.html               # Frontend UI to test messages
│── requirements.txt         # Dependencies
│── README.md                # Project documentation
```

---

## 📦 **Download Trained Model (600 MB)**

Due to GitHub’s file size limit, the trained model is hosted externally.

👉 **Download model weights:**
🔗  https://drive.google.com/file/d/1Lv-B3PL_5zi_Nvtl3jSrWHRjQEyZZ_FA/view?usp=sharing

Place the downloaded file in your project folder:

```
review_classifier.pth
```

---

## ▶️ **How to Run Locally**

### **1️⃣ Install dependencies**

```
pip install -r requirements.txt
```

### **2️⃣ Start FastAPI backend**

```
uvicorn llmfromscratch:app --reload
```

### **3️⃣ Open `index.html`**

Simply open the file in your browser.

The UI will send requests to:

```
http://127.0.0.1:8000/predict
```

Enter any text → See the model’s prediction.



---

## ✨ **Key Learnings**

* Deep understanding of Transformer internals
* Masked self-attention implementation
* How embeddings + attention + MLP layers work together
* Building real-time inference APIs
* Connecting ML models to frontend interfaces

---

## 🛠️ **Tech Stack**

* **Python**, **PyTorch**
* **FastAPI**
* **HTML**, **CSS**, **JavaScript**
* **tiktoken** (GPT-2 tokenizer)


## ⭐ **If you like this project**

Feel free to star ⭐ the repo and connect on LinkedIn!

---
