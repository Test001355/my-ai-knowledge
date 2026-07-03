---
title: Week 8 - The Era of Transformers
phase: Phase 2 - Deep Learning & Unstructured Data
tags: #Transformers #Attention #BERT #GPT #HuggingFace #FineTuning #TextGeneration #NLP
date: 2026-07-03
---

# 🟡 Week 8 - The Era of Transformers

> [!NOTE] 📋 ภาพรวมเนื้อหา (Overview)
> **ทฤษฎี (2 ชม.):** Attention Mechanism, สถาปัตยกรรม Transformer, แนะนำ Hugging Face  
> **เวิร์กชอป (Workshops):**
> - **Workshop 1:** สร้างบล็อก Self-Attention แบบง่ายเพื่อให้เข้าใจการทำงาน
> - **Workshop 2:** การดึงโมเดล pre-trained จาก Hugging Face (Transformers library) มาใช้งาน
> - **Workshop 3:** Fine-tuning โมเดล BERT สำหรับงาน Text Classification
> - **Workshop 4:** การทำ Text Generation เบื้องต้นโดยใช้โมเดลกลุ่ม GPT-2

---

## 📖 ส่วนที่ 1: สรุปทฤษฎีสำคัญ (Key Theory Concepts)

### 1. ทำไม Transformer ถึงปฏิวัติโลก AI?
ก่อนหน้าปี 2017 โมเดล RNN และ LSTM ต้องประมวลผลข้อความทีละคำจากซ้ายไปขวา ทำให้ช้าและลืมคำต้นประโยคเมื่อเจอประโยคยาว สถาปัตยกรรม **Transformer (จากเปเปอร์ "Attention Is All You Need")** แก้ปัญหานี้โดยประมวลผลทุกคำพร้อมกัน (Parallel Processing) และใช้กลไก **Self-Attention**

### 2. กลไก Self-Attention (Q, K, V)
คำแต่ละคำในประโยคจะมองหา "ความสัมพันธ์" กับคำอื่น ๆ ในประโยคเดียวกันผ่านเมทริกซ์ 3 ตัว:
- **Query (Q):** สิ่งที่คำนั้นกำลังค้นหา
- **Key (K):** ป้ายระบุคุณสมบัติของคำอื่น ๆ
- **Value (V):** ข้อมูลความหมายจริงของคำนั้น

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 3. ตระกูลโมเดลหลักในยุค Transformer
- **Encoder-only (เช่น BERT, RoBERTa):** เก่งงานทำความเข้าใจภาษา (NLU) เช่น วิเคราะห์อารมณ์ แยกหมวดหมู่ ค้นหาคำตอบในบทความ
- **Decoder-only (เช่น GPT-2, GPT-4, Llama):** เก่งงานสร้างข้อความ (NLG) โดยทำนายคำถัดไปทีละคำ (Auto-regressive)
- **Encoder-Decoder (เช่น T5, BART):** เก่งงานแปลภาษา และย่อความ

---

## 🛠️ ส่วนที่ 2: เวิร์กชอปเชิงปฏิบัติการ (Hands-on Workshops)

### Workshop 1: สร้างบล็อก Self-Attention แบบง่ายเพื่อให้เข้าใจการทำงาน
เขียนฟังก์ชันคำนวณ Self-Attention จากคณิตศาสตร์พื้นฐานด้วย PyTorch เพื่อดูเมทริกซ์ความสัมพันธ์ระหว่างคำ

```python
import torch
import torch.nn.functional as F

def simple_self_attention(x):
    """
    x shape: (batch_size, seq_len, embed_dim)
    สมมติให้ Q, K, V เท่ากับ x เพื่อความเรียบง่าย (Scaled Dot-Product Attention)
    """
    Q = K = V = x
    d_k = Q.size(-1)
    
    # 1. คำนวณคะแนนความสัมพันธ์ (Attention Scores) = Q * K^T
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # 2. แปลงเป็นความน่าจะเป็นระหว่าง 0 ถึง 1 ด้วย Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # 3. คูณด้วย Value (V) เพื่อสกัดข้อมูลที่ต้องสนใจ
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

# --- ทดสอบจำลองประโยค 3 คำ คำละ 4 มิติ ---
x_sample = torch.tensor([[
    [1.0, 0.0, 1.0, 0.0],  # คำที่ 1
    [0.0, 2.0, 0.0, 1.0],  # คำที่ 2
    [1.0, 1.0, 1.0, 1.0]   # คำที่ 3
]])

out, weights = simple_self_attention(x_sample)
print("Attention Weights Matrix (3x3):\n", np.round(weights.numpy(), 3))
print("\nOutput Embeddings:\n", np.round(out.numpy(), 3))
```

---

### Workshop 2: การดึงโมเดล pre-trained จาก Hugging Face มาใช้งาน
ใช้ไลบรารี `transformers` และคำสั่ง `pipeline` เพื่อเรียกใช้ AI ระดับโลกสำหรับการวิเคราะห์ข้อความภายใน 3 บรรทัด

```python
# หมายเหตุ: หากยังไม่ได้ติดตั้ง ให้รัน -> pip install transformers torch
from transformers import pipeline

# 1. Pipeline สำหรับ Sentiment Analysis (ใช้โมเดล Pre-trained)
classifier = pipeline("sentiment-analysis")
result = classifier("Transformers are absolutely shaping the future of AI engineering!")
print("Sentiment Result:", result)

# 2. Pipeline สำหรับ Named Entity Recognition (NER - ดึงชื่อคน สถานที่ องค์กร)
ner = pipeline("ner", grouped_entities=True)
ner_result = ner("Google DeepMind team is developing Advanced Agentic Coding in London.")
print("\nNER Result:")
for entity in ner_result:
    print(f"- {entity['word']} ({entity['entity_group']}) | Confidence: {entity['score']:.2f}")
```

---

### Workshop 3: Fine-tuning โมเดล BERT สำหรับงาน Text Classification
จำลองขั้นตอนการเตรียมโมเดลและ Dataset เพื่อ Fine-tune โมเดล BERT ให้รู้จักการจำแนกข้อความเฉพาะทางของเรา

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. โหลด Tokenizer และโมเดล BERT (ขนาดเล็กเช่น DistilBERT เพื่อความเร็ว)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. เตรียมข้อความตัวอย่าง
texts = ["This library is so fast and easy to use.", "I encountered too many bugs and crashes."]
labels = torch.tensor([1, 0]) # 1 = Positive, 0 = Negative

# 3. แปลงข้อความให้เป็น Input IDs และ Attention Masks
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 4. คำนวณค่า Loss ในโหมด Training
model.train()
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits

print(f"Initial Training Loss: {loss.item():.4f}")
print("Logits (คะแนนดิบก่อนแปลงเป็น Probability):\n", logits.detach().numpy())
print("\n👉 พร้อมนำเข้าสู่ลูป Optimizer.step() หรือใช้ Hugging Face Trainer เพื่อทำการ Fine-tune ต่อไป!")
```

---

### Workshop 4: การทำ Text Generation เบื้องต้นโดยใช้โมเดลกลุ่ม GPT-2
ใช้โมเดล Decoder-only (GPT-2) ในการแต่งประโยคหรือเขียนโค้ดต่อจากข้อความเริ่มต้น (Prompt)

```python
from transformers import pipeline

# 1. โหลด Pipeline สำหรับ Text Generation (ใช้ GPT-2)
generator = pipeline("text-generation", model="gpt2")

# 2. กำหนดข้อความเริ่มต้น (Prompt)
prompt = "The future of Artificial Intelligence and Machine Learning engineering will be"

# 3. ให้โมเดลสร้างข้อความต่อ (Generate)
output = generator(
    prompt, 
    max_length=40,       # ความยาวสูงสุดของประโยครวม
    num_return_sequences=1, 
    temperature=0.7,     # ความคิดสร้างสรรค์ (0.1 = ตอบตรงไปตรงมา, 0.9 = สร้างสรรค์สูง)
    top_p=0.9,
    pad_token_id=50256
)

print("--- ผลลัพธ์การสร้างข้อความจาก GPT-2 ---")
print(output[0]['generated_text'])
```

---

## 🔗 อ้างอิงและจุดเชื่อมโยง (Wiki-Links & Next Steps)
- **บทเรียนก่อนหน้า:** [[Week 7 - Natural Language Processing (NLP) Foundations]]
- **ภาพรวมเฟส:** [[Phase 2 - Overview|Phase 2: Deep Learning & Unstructured Data]]
- **ก้าวสู่เฟสที่ 3:** [[Phase 3 - Overview|Phase 3: Generative AI & Application Building]]
- **ความรู้ที่เกี่ยวข้อง:** [[Attention Is All You Need Paper]], [[HuggingFace Transformers Guide]], [[BERT vs GPT Architecture]]
