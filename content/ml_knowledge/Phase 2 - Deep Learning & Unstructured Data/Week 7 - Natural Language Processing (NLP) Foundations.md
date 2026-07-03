---
title: Week 7 - Natural Language Processing (NLP) Foundations
phase: Phase 2 - Deep Learning & Unstructured Data
tags: #NLP #PyTorch #WordEmbeddings #SentimentAnalysis #DeepLearning #MachineLearning
date: 2026-07-03
---

# 🧠 Week 7 - Natural Language Processing (NLP) Foundations

> [!NOTE] 📋 ภาพรวมเนื้อหา (Overview)
> **ทฤษฎี (2 ชม.):** Text Preprocessing, Tokenization, Word Embeddings, RNNs พื้นฐาน  
> **เวิร์กชอป (Workshops):**
> - **Workshop 1:** การทำ Text Tokenization และ Text Cleaning
> - **Workshop 2:** การใช้งาน Word Embeddings และดูความหมายของคำในรูปแบบ Vector
> - **Workshop 3:** สร้างโมเดล Sentiment Analysis วิเคราะห์อารมณ์ข้อความ
> - **Workshop 4:** การทำ Sequence Classification แบบง่ายด้วย PyTorch

---

## 📖 ส่วนที่ 1: สรุปทฤษฎีสำคัญ (Key Theory Concepts)

### 1. Text Preprocessing & Cleaning
ข้อความ (Text) ในโลกจริงมักมีความไม่เรียบร้อย เช่น ตัวอักษรพิมพ์เล็ก-ใหญ่ เครื่องหมายวรรคตอน หรือคำสร้อยที่ไม่สื่อความหมาย ขั้นตอนการทำความสะอาดข้อมูลก่อนนำเข้าโมเดลจึงสำคัญมาก:
- **Lowercasing:** แปลงตัวอักษรทั้งหมดเป็นพิมพ์เล็ก (เช่น `"Apple"` $\rightarrow$ `"apple"`)
- **Removing Punctuation & Numbers:** ลบเครื่องหมายวรรคตอนและตัวเลขที่ไม่จำเป็นออก
- **Stopword Removal:** ลบคำเชื่อมหรือคำสร้อยที่มีความถี่สูงแต่สื่อความหมายน้อย (เช่น *is, am, are, the, in, ของ, ที่, ซึ่ง, อัน*)
- **Stemming & Lemmatization:** การตัดรากคำให้กลับสู่รูปมาตรฐาน (เช่น *running, ran* $\rightarrow$ *run*)

### 2. Tokenization (การตัดคำ)
โมเดลคอมพิวเตอร์ไม่สามารถประมวลผลประโยคยาว ๆ ได้โดยตรง จึงต้องตัดแบ่งข้อความออกเป็นหน่วยย่อยเรียกว่า **"Token"**
- **Word Tokenization:** ตัดแบ่งเป็นคำ ๆ (ภาษาไทยมีความท้าทายสูงเพราะเขียนติดกัน มักใช้ไลบรารีอย่าง `PyThaiNLP` หรือ `NewMM`)
- **Subword Tokenization (BPE, WordPiece):** ตัดแบ่งระดับกึ่งคำ (ใช้ในโมเดลสมัยใหม่อย่าง BERT หรือ GPT) ช่วยแก้ปัญหาคำที่ไม่รู้จักในระบบ (Out-of-Vocabulary / OOV)
- **Character Tokenization:** ตัดแบ่งระดับตัวอักษร

### 3. Word Embeddings (เวกเตอร์ของคำ)
การแปลง Token ให้กลายเป็นตัวเลขที่คอมพิวเตอร์เข้าใจ โดยมิติของตัวเลข (Vector Space) จะสื่อถึง **"ความหมาย (Semantic)"** ของคำนั้น ๆ
- **One-Hot Encoding:** วิธีดั้งเดิม แปลงคำเป็นเวกเตอร์ที่มีเลข 1 ตำแหน่งเดียวและ 0 ที่เหลือ (ข้อเสีย: ไม่สื่อความหมาย และเวกเตอร์มีขนาดใหญ่เกินไปตามจำนวนคำในพจนานุกรม)
- **Dense Word Embeddings (Word2Vec, FastText, GloVe):** เวกเตอร์ความหนาแน่นสูง (เช่น ขนาด 100-300 มิติ) คำที่มีความหมายใกล้เคียงกันจะมีระยะห่างในพื้นที่เวกเตอร์ใกล้เคียงกัน เช่น:
  $$\text{Vector}(\text{"King"}) - \text{Vector}(\text{"Man"}) + \text{Vector}(\text{"Woman"}) \approx \text{Vector}(\text{"Queen"})$$

### 4. Recurrent Neural Networks (RNNs) พื้นฐาน
ภาษาและข้อความมีลักษณะเป็น **ข้อมูลอนุกรมเวลา (Sequence Data)** ที่ลำดับคำมีความสำคัญ (เช่น *"ฉันกินข้าว"* ความหมายไม่เหมือนกับ *"ข้าวกินฉัน"*)
- **RNN (Recurrent Neural Network):** มีโครงสร้างวนซ้ำ (Loop) ที่สามารถส่งผ่าน "ความจำ (Hidden State)" จากคำก่อนหน้าไปสู่คำถัดไปได้
- **ข้อจำกัดของ RNN ดั้งเดิม:** เมื่อประโยคยาวมาก ๆ จะเกิดปัญหา **Vanishing Gradient (ความจำสั้น)** ทำให้ลืมข้อมูลต้นประโยค จึงมีการพัฒนาเป็นสถาปัตยกรรม **LSTM (Long Short-Term Memory)** และ **GRU (Gated Recurrent Unit)** มารองรับแทน

---

## 🛠️ ส่วนที่ 2: เวิร์กชอปเชิงปฏิบัติการ (Hands-on Workshops)

### Workshop 1: การทำ Text Tokenization และ Text Cleaning
ทำความสะอาดข้อความและตัดคำทั้งภาษาอังกฤษและภาษาไทยด้วย Python

```python
import re
import string

# ----------------------------------------------------
# 1. การทำความสะอาดข้อความภาษาอังกฤษ (English Cleaning)
# ----------------------------------------------------
def clean_english_text(text):
    # 1. แปลงเป็นตัวพิมพ์เล็ก
    text = text.lower()
    # 2. ลบ URL และแท็ก HTML
    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)
    # 3. ลบเครื่องหมายวรรคตอนและตัวเลข
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    # 4. ลบช่องว่างส่วนเกิน
    text = re.sub(r'\s+', ' ', text).strip()
    return text

sample_en = "Hello!! Welcome to Machine Learning Week 7... NLP is AMAZING in 2026! Check http://ai.com"
cleaned_en = clean_english_text(sample_en)
print("Original (EN):", sample_en)
print("Cleaned  (EN):", cleaned_en)
print("Tokens   (EN):", cleaned_en.split())
print("-" * 50)

# ----------------------------------------------------
# 2. การตัดคำภาษาไทยด้วย PyThaiNLP (Thai Tokenization)
# ----------------------------------------------------
# หมายเหตุ: หากยังไม่ได้ติดตั้ง ให้รันคำสั่ง -> pip install pythainlp
try:
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus import thai_stopwords

    sample_th = "การประมวลผลภาษาธรรมชาติในโลกปัญญาประดิษฐ์ปี 2026 สนุกมากๆครับ"
    
    # ตัดคำด้วยเครื่องมือ newmm (Dictionary-based + Maximal Matching)
    tokens_th = word_tokenize(sample_th, engine="newmm")
    print("Original (TH):", sample_th)
    print("Tokens   (TH):", tokens_th)

    # ลบคำฟุ่มเฟือย (Stopwords) ออก
    stopwords = thai_stopwords()
    filtered_th = [word for word in tokens_th if word not in stopwords and word.strip() != ""]
    print("Filtered (TH):", filtered_th)

except ImportError:
    print("กรุณาติดตั้ง pythainlp โดยรัน: pip install pythainlp")
```

---

### Workshop 2: การใช้งาน Word Embeddings และดูความหมายของคำในรูปแบบ Vector
จำลองเวกเตอร์ของคำและการคำนวณความเหมือน (Cosine Similarity) ระหว่างคำใน Vector Space

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------
# จำลองตาราง Word Embeddings ขนาด 4 มิติของคำศัพท์ต่าง ๆ
# ----------------------------------------------------
vocab_embeddings = {
    "king":   np.array([ 0.90,  0.85,  0.10, -0.20]),
    "queen":  np.array([ 0.88, -0.80,  0.15, -0.18]),
    "man":    np.array([ 0.50,  0.80,  0.05,  0.10]),
    "woman":  np.array([ 0.48, -0.75,  0.08,  0.12]),
    "apple":  np.array([-0.80,  0.10,  0.95,  0.50]),
    "banana": np.array([-0.75,  0.05,  0.90,  0.55])
}

def get_similarity(word1, word2):
    vec1 = vocab_embeddings[word1].reshape(1, -1)
    vec2 = vocab_embeddings[word2].reshape(1, -1)
    sim = cosine_similarity(vec1, vec2)[0][0]
    return sim

# 1. เปรียบเทียบความคล้ายคลึงกัน (Cosine Similarity)
print(f"Similarity (apple <-> banana): {get_similarity('apple', 'banana'):.4f}") # ควรใกล้เคียง 1 (ผลไม้เหมือนกัน)
print(f"Similarity (king  <-> queen):  {get_similarity('king', 'queen'):.4f}")   # คล้ายกันในเชิงสถานะกษัตริย์
print(f"Similarity (apple <-> king):   {get_similarity('apple', 'king'):.4f}")   # ควรติดลบหรือเข้าใกล้ 0 (ไม่เกี่ยวข้องกัน)
print("-" * 50)

# 2. ทดสอบสมการราชาคำศัพท์ (King - Man + Woman ~ Queen)
result_vec = vocab_embeddings["king"] - vocab_embeddings["man"] + vocab_embeddings["woman"]
result_vec = result_vec.reshape(1, -1)

# หาคำศัพท์ในคลังที่มีเวกเตอร์ใกล้เคียงกับผลลัพธ์ที่สุด
best_match = None
max_sim = -1.0

for word, vec in vocab_embeddings.items():
    if word in ["king", "man", "woman"]:
        continue # ข้ามคำที่เป็นตัวตั้ง
    sim = cosine_similarity(result_vec, vec.reshape(1, -1))[0][0]
    print(f"ความคล้ายของผลลัพธ์กับ '{word}': {sim:.4f}")
    if sim > max_sim:
        max_sim = sim
        best_match = word

print(f"\n👉 ผลลัพธ์ของสมการ (King - Man + Woman) คือคำว่า: ** {best_match.upper()} ** (Similarity = {max_sim:.4f})")
```

---

### Workshop 3: สร้างโมเดล Sentiment Analysis วิเคราะห์อารมณ์ข้อความ
สร้างโมเดลจำแนกข้อความรีวิวภาพยนตร์ว่ามีอารมณ์เชิงบวก (Positive) หรือเชิงลบ (Negative) ด้วย Scikit-Learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. ชุดข้อมูลตัวอย่างรีวิวภาพยนตร์ (Dataset)
reviews = [
    "This movie is absolutely fantastic! I loved every second of it.",
    "Great acting and wonderful story. A must watch!",
    "One of the best films I have ever seen in my life.",
    "I enjoyed the plot and the characters were very charming.",
    "What a brilliant masterpiece! Highly recommended.",
    "Terrible movie. A complete waste of time and money.",
    "The acting was horrible and the plot made no sense.",
    "I hated this film so much. Easily the worst movie of the year.",
    "So boring and predictable. I fell asleep halfway through.",
    "Awful directing and terrible script. Do not watch it."
]

# เลเบล: 1 = Positive (บวก), 0 = Negative (ลบ)
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# 2. แบ่งข้อมูล Train / Test
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.3, random_state=42)

# 3. แปลงข้อความให้เป็นตัวเลขด้วย TF-IDF (Term Frequency-Inverse Document Frequency)
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. ฝึกสอนโมเดล Logistic Regression
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 5. ทำนายผลและประเมินความแม่นยำ
y_pred = model.predict(X_test_vec)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["Negative (0)", "Positive (1)"]))

# 6. ลองนำโมเดลไปทดสอบกับประโยคใหม่ๆ
new_reviews = [
    "The story was amazing and I really enjoyed the characters!",
    "This was a horrible film, absolutely waste of time."
]
new_vecs = vectorizer.transform(new_reviews)
preds = model.predict(new_vecs)

print("\n--- ผลการทำนายประโยคใหม่ ---")
for text, pred in zip(new_reviews, preds):
    sentiment = "😊 Positive (เชิงบวก)" if pred == 1 else "😡 Negative (เชิงลบ)"
    print(f"Review: '{text}' --> {sentiment}")
```

---

### Workshop 4: การทำ Sequence Classification แบบง่ายด้วย PyTorch
สร้างโครงข่ายประสาทเทียมแบบ Deep Learning ที่ใช้ชั้น **Embedding Layer** และ **RNN/LSTM Layer** ใน PyTorch เพื่อจัดหมวดหมู่ข้อความ

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------------------------------
# 1. เตรียมคำศัพท์และข้อมูลจำลอง (Vocabulary & Data)
# ----------------------------------------------------
# ประโยคตัวอย่าง: หมวดหมู่ กีฬา (1) vs หมวดหมู่ เทคโนโลยี (0)
corpus = [
    ("i love playing football", 1),
    ("soccer is a great sport", 1),
    ("basketball game was exciting", 1),
    ("computer processor is super fast", 0),
    ("new smart phone battery life", 0),
    ("artificial intelligence algorithm", 0)
]

# สร้างคำศัพท์ (Vocabulary Dictionary)
word2idx = {"<PAD>": 0}
for sentence, _ in corpus:
    for word in sentence.split():
        if word not in word2idx:
            word2idx[word] = len(word2idx)

vocab_size = len(word2idx)
print(f"Vocabulary Size: {vocab_size} words")

# แปลงประโยคเป็น Indice Tensors และ Padding ให้ยาวเท่ากัน (5 คำ)
max_len = 5
X_data, y_data = [], []

for sentence, label in corpus:
    tokens = [word2idx[w] for w in sentence.split()]
    # Pad ด้วย 0 ให้ครบ max_len
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    X_data.append(tokens[:max_len])
    y_data.append(label)

X_tensor = torch.tensor(X_data, dtype=torch.long)
y_tensor = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1) # เป็น Float สำหรับ BCELoss

# ----------------------------------------------------
# 2. สร้างสถาปัตยกรรมโมเดล RNN / LSTM ด้วย PyTorch
# ----------------------------------------------------
class SimpleLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleLSTMClassifier, self).__init__()
        # 1. Embedding Layer: แปลง Index ของคำให้เป็นเวกเตอร์
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # 2. LSTM Layer: ประมวลผลข้อมูลตามลำดับเวลา (Sequence)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 3. Fully Connected (Linear) Layer: แปลง Hidden State สุดท้ายเป็นผลลัพธ์
        self.fc = nn.Linear(hidden_dim, output_dim)
        # 4. Sigmoid: แปลงผลลัพธ์เป็นความน่าจะเป็นระหว่าง 0 ถึง 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embeds = self.embedding(x)          # (batch_size, seq_len, embedding_dim)
        lstm_out, (hn, cn) = self.lstm(embeds) # hn shape: (1, batch_size, hidden_dim)
        
        # ดึง Hidden State ของคำสุดท้ายมาใช้ในการตัดสินใจ
        last_hidden = hn.squeeze(0)         # (batch_size, hidden_dim)
        out = self.fc(last_hidden)          # (batch_size, output_dim)
        return self.sigmoid(out)

# กำหนด Hyperparameters
EMBEDDING_DIM = 16
HIDDEN_DIM = 32
model = SimpleLSTMClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim=1)

# กำหนด Loss Function และ Optimizer
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ----------------------------------------------------
# 3. ฝึกสอนโมเดล (Training Loop)
# ----------------------------------------------------
print("\n--- เริ่มต้นฝึกสอนโมเดล LSTM ---")
for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()
    
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/50 | Loss: {loss.item():.4f}")

# ----------------------------------------------------
# 4. ทดสอบพยากรณ์ประโยคใหม่
# ----------------------------------------------------
model.eval()
test_sentences = [
    "great football sport",
    "fast computer algorithm"
]

print("\n--- ผลการทดสอบประโยคใหม่ ---")
with torch.no_grad():
    for sent in test_sentences:
        tokens = [word2idx.get(w, 0) for w in sent.split()]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        test_tensor = torch.tensor([tokens[:max_len]], dtype=torch.long)
        
        prob = model(test_tensor).item()
        category = "⚽ กีฬา (Sport)" if prob > 0.5 else "💻 เทคโนโลยี (Tech)"
        print(f"Sentence: '{sent}' --> ทำนาย: {category} (Probability of Sport: {prob:.4f})")
```

---

## 🔗 อ้างอิงและจุดเชื่อมโยง (Wiki-Links & Next Steps)
- **เนื้อหาก่อนหน้า:** [[Week 6 - Computer Vision Advanced]]
- **เนื้อหาถัดไป:** [[Week 8 - The Era of Transformers]]
- **ความรู้ที่เกี่ยวข้อง:** [[PyTorch Fundamentals]], [[Deep Learning Architecture]], [[Attention Mechanism]]
