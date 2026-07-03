---
title: Week 9 - Generative AI & Prompt Engineering Masterclass
phase: Phase 3 - Generative AI & Application Building
tags: #GenerativeAI #PromptEngineering #LLM #OpenAI #Gemini #Streamlit #ZeroShot #FewShot #CoT #JSON
date: 2026-07-03
---

# 🟢 Week 9 - Generative AI & Prompt Engineering Masterclass

> [!NOTE] 📋 ภาพรวมเนื้อหา (Overview)
> **ทฤษฎี (2 ชม.):** หลักการทำงานของ LLMs, รูปแบบ Prompt Engineering, ค่า API พารามิเตอร์ (Temperature, Top-P)  
> **เวิร์กชอป (Workshops):**
> - **Workshop 1:** การเรียกใช้ LLM ผ่าน API (OpenAI / Anthropic / Gemini API)
> - **Workshop 2:** ฝึกทักษะ Zero-Shot, Few-Shot และ Chain-of-Thought (CoT) Prompting
> - **Workshop 3:** การบังคับให้ LLM ตอบกลับเป็น Structured Data (JSON format)
> - **Workshop 4:** สร้าง Chatbot UI ส่วนตัวด้วย Streamlit

---

## 📖 ส่วนที่ 1: สรุปทฤษฎีสำคัญ (Key Theory Concepts)

### 1. Large Language Models (LLMs) และ พารามิเตอร์สำคัญ
LLM คือโมเดล Transformer ขนาดใหญ่ที่ถูกฝึกให้พยากรณ์คำถัดไป (Next-token prediction) การควบคุมพฤติกรรมของ LLM ทำได้ผ่านพารามิเตอร์:
- **Temperature:** ควบคุมความสุ่ม (Randomness) ของคำตอบ
  - `Temperature = 0.0` $\rightarrow$ ตอบตรงไปตรงมา แม่นยำ เหมาะกับงานคณิตศาสตร์หรือเขียนโค้ด
  - `Temperature = 0.7 - 1.0` $\rightarrow$ มีความคิดสร้างสรรค์ หลากหลาย เหมาะกับการแต่งนิยายหรือระดมสมอง
- **Top-P (Nucleus Sampling):** สุ่มเลือกเฉพาะคำศัพท์ที่มีความน่าจะเป็นสะสมรวมกันครบ $P\%$ (เช่น Top-P = 0.9 คือตัดคำแปลกประหลาด 10% ท้ายทิ้งไป)

### 2. รูปแบบ Prompt Engineering (เทคนิคสั่งการ AI)
- **Zero-Shot Prompting:** สั่งงานตรง ๆ โดยไม่ยกตัวอย่าง
- **Few-Shot Prompting:** ยกตัวอย่างอินพุตและเอาต์พุตให้ AI ดู 2-3 ตัวอย่าง เพื่อให้จับแพทเทิร์นได้แม่นยำขึ้น
- **Chain-of-Thought (CoT):** สั่งให้ AI "คิดทีละขั้นตอน (Let's think step by step)" ก่อนสรุปคำตอบ ช่วยเพิ่มความถูกต้องในโจทย์ตรรกศาสตร์และคณิตศาสตร์อย่างมหาศาล

---

## 🛠️ ส่วนที่ 2: เวิร์กชอปเชิงปฏิบัติการ (Hands-on Workshops)

### Workshop 1: การเรียกใช้ LLM ผ่าน API (OpenAI / Gemini API)
เขียน Python เชื่อมต่อกับ LLM API โดยตรง พร้อมตั้งค่า Temperature และ System Prompt

```python
# หมายเหตุ: หากยังไม่ได้ติดตั้ง ให้รัน -> pip install openai google-generativeai
import os

# ----------------------------------------------------
# 1. ตัวอย่างการใช้ OpenAI API (GPT-4o / GPT-3.5)
# ----------------------------------------------------
try:
    from openai import OpenAI
    # กำหนด API Key (ควรตั้งเป็น Environment Variable: os.environ["OPENAI_API_KEY"])
    client = OpenAI(api_key="sk-your-mock-api-key-here")

    def ask_gpt(prompt, temp=0.7):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "คุณคือผู้เชี่ยวชาญด้าน AI Engineering ตอบคำถามกระชับและเป็นมืออาชีพ"},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=200
        )
        return response.choices[0].message.content

    # print("GPT Response:", ask_gpt("LLM คืออะไร อธิบายใน 2 บรรทัด"))
except Exception as e:
    print("OpenAI API ซ้อมทำงาน (ตรวจสอบ API Key ก่อนใช้งานจริง)")

# ----------------------------------------------------
# 2. ตัวอย่างการใช้ Google Gemini API
# ----------------------------------------------------
try:
    import google.generativeai as genai
    genai.configure(api_key="AIzaSy-your-mock-gemini-api-key-here")
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    # response = model.generate_content("ข้อดีของ Prompt Engineering มีอะไรบ้าง?")
    # print("Gemini Response:", response.text)
except Exception as e:
    print("Google Gemini API ซ้อมทำงาน (ตรวจสอบ API Key ก่อนใช้งานจริง)")
```

---

### Workshop 2: ฝึกทักษะ Zero-Shot, Few-Shot และ Chain-of-Thought (CoT)
เปรียบเทียบผลลัพธ์ของพรอมต์ทั้ง 3 รูปแบบในการแก้โจทย์ปัญหาเดียวกัน

```python
# ----------------------------------------------------
# 1. Zero-Shot Prompting (ถามตรงๆ)
# ----------------------------------------------------
zero_shot_prompt = """
จำแนกอารมณ์ของรีวิวนี้ว่าเป็น บวก หรือ ลบ:
"ระบบใช้งานยากมาก โหลดช้าสุดๆ ไม่ประทับใจเลย"
อารมณ์:
"""

# ----------------------------------------------------
# 2. Few-Shot Prompting (ให้ตัวอย่างประกอบ)
# ----------------------------------------------------
few_shot_prompt = """
จำแนกอารมณ์ของรีวิวด้านล่างนี้:

รีวิว: "สินค้าจัดส่งไวมาก ประทับใจการบริการครับ"
อารมณ์: บวก

รีวิว: "ของชำรุด ติดต่อฝ่ายซัพพอร์ตก็ไม่มีใครตอบ"
อารมณ์: ลบ

รีวิว: "ระบบใช้งานยากมาก โหลดช้าสุดๆ ไม่ประทับใจเลย"
อารมณ์:
"""

# ----------------------------------------------------
# 3. Chain-of-Thought (CoT) Prompting (คิดทีละขั้นตอน)
# ----------------------------------------------------
cot_prompt = """
โจทย์: ร้านค้ามีแอปเปิ้ล 50 ลูก ขายไปตอนเช้า 15 ลูก ตอนบ่ายมีคนเอามาส่งเพิ่มอีก 20 ลูก จากนั้นขายไปอีกครึ่งหนึ่งที่มีอยู่ ตอนนี้ร้านค้าเหลือแอปเปิ้ลกี่ลูก?

กรุณาคิดวิเคราะห์ทีละขั้นตอน (Let's think step by step) แล้วสรุปคำตอบสุดท้าย
"""

print("--- ตัวอย่าง Chain-of-Thought Prompt ---")
print(cot_prompt)
```

---

### Workshop 3: การบังคับให้ LLM ตอบกลับเป็น Structured Data (JSON format)
ใช้เทคนิค **Structured Outputs (JSON Mode)** เพื่อให้มั่นใจว่า LLM ตอบกลับเป็น JSON ที่โปรแกรมเมอร์นำไป `json.loads()` ต่อได้ไม่พัง

```python
import json

# ตัวอย่าง Prompt บังคับ Output เป็น JSON Schema
json_prompt = """
ดึงข้อมูลสำคัญจากประโยคต่อไปนี้และให้ออกมาในรูปแบบ JSON เท่านั้น ห้ามมีข้อความอื่นปน:
ประโยค: "คุณสมชาย มุ่งมั่น อายุ 35 ปี ทำงานเป็น Data Engineer ที่บริษัท AI Tech จำกัด เงินเดือน 85,000 บาท"

โครงสร้าง JSON ที่ต้องการ:
{
    "first_name": "...",
    "last_name": "...",
    "age": number,
    "job_title": "...",
    "company": "...",
    "salary": number
}
"""

# จำลองผลลัพธ์ที่ได้จาก LLM (Valid JSON String)
mock_llm_output = """
{
    "first_name": "สมชาย",
    "last_name": "มุ่งมั่น",
    "age": 35,
    "job_title": "Data Engineer",
    "company": "บริษัท AI Tech จำกัด",
    "salary": 85000
}
"""

# แปลง JSON String เป็น Python Dictionary
user_data = json.loads(mock_llm_output)
print("--- ดึงข้อมูลสำเร็จด้วย JSON Mode ---")
print(f"ชื่อพนักงาน: {user_data['first_name']} {user_data['last_name']}")
print(f"ตำแหน่ง:    {user_data['job_title']} ({user_data['company']})")
print(f"เงินเดือน:    {user_data['salary']:,.2f} บาท")
```

---

### Workshop 4: สร้าง Chatbot UI ส่วนตัวด้วย Streamlit
ใช้ไลบรารี **Streamlit** สร้างหน้าเว็บแชตบอต UI อย่างรวดเร็ว พร้อมระบบบันทึกประวัติการสนทนา (Chat History)

```python
# หมายเหตุ: หากยังไม่ได้ติดตั้ง ให้รัน -> pip install streamlit
import streamlit as st
import time

def main():
    st.set_page_config(page_title="🤖 My AI Engineer Chatbot", page_icon="💡")
    st.title("💡 AI & Machine Learning Assistant")
    st.caption("แชตบอตถาม-ตอบความรู้ AI สร้างด้วย Streamlit และ Python")

    # 1. เริ่มต้นบันทึกประวัติแชตใน Session State
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "สวัสดีครับ! ผมคือผู้ช่วย AI ด้าน Machine Learning Engineering มีข้อสงสัยเรื่องไหนสอบถามได้เลยครับ!"}
        ]

    # 2. แสดงประวัติข้อความบนหน้าเว็บ
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 3. กล่องรับข้อความจากผู้ใช้งาน (Chat Input)
    if user_input := st.chat_input("พิมพ์คำถามของคุณที่นี่... (เช่น RAG คืออะไร?)"):
        # แสดงข้อความฝั่ง User
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 4. จำลองการคิดและสร้างคำตอบจาก AI
        with st.chat_message("assistant"):
            with st.spinner("กำลังประมวลผล..."):
                time.sleep(1) # จำลองเวลาหน่วง API
                
                # จำลองคำตอบ (ในการใช้งานจริง ให้เรียก client.chat.completions.create ตรงนี้)
                response_text = f"💡 สำหรับคำถามเกี่ยวกับ **'{user_input}'**: ในฐานะ ML Engineer เราแนะนำให้ใช้โครงสร้างสถาปัตยกรรมที่เหมาะสม ปรับ Temperature ให้พอดี และใช้ Vector Database ในการค้นหาครับ!"
                st.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    # วิธีรันแอป: บันทึกโค้ดนี้ไฟล์ชื่อ app.py แล้วพิมพ์ใน Terminal -> streamlit run app.py
    print("โค้ด Streamlit พร้อมทำงาน! รันคำสั่ง: streamlit run filename.py ใน Terminal")
```

---

## 🔗 อ้างอิงและจุดเชื่อมโยง (Wiki-Links & Next Steps)
- **ภาพรวมเฟส:** [[Phase 3 - Overview|Phase 3: Generative AI & Application Building]]
- **บทเรียนถัดไป:** [[Week 10 - Retrieval-Augmented Generation (RAG)]]
- **ความรู้ที่เกี่ยวข้อง:** [[OpenAI API Reference]], [[Streamlit Chat UI Guide]], [[JSON Schema Validation]]
