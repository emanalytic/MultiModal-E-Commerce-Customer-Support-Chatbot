# Multimodal E-Commerce Customer Service Chatbot 🛍️

## Problem:
Customers interact with brands via text, images, and voice—but most chatbots only handle text. This creates a fragmented experience and limits accessibility.

## 🌐 Solution:
A state-of-the-art chatbot designed to handle multimodal inputs—including text, voice and images to provide personalized shopping recommendations and support across your e-commerce platform. It delivers personalized product recommendations, answers FAQs, and support across your e-commerce platform. 

## 🛠️ Tools & Technologies
- **LangChain:** To orchestrate various language processing tasks
- **Groq API:** For generating responses (Groq / Llama3)
- **Chainlit:** To build interactive, user-friendly chatbot interface
- **OpenAI-CLIP:** For image embeddings and similarity search
- **Speech Recognition:**  Google Speech-to-Text for voice input
- **Hugging Face Model:** For Text embeddings and similarity search

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Chainlit](https://img.shields.io/badge/Chainlit-4B8BBE?style=for-the-badge&logo=react&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-FF6F00?style=for-the-badge&logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-00C853?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenAI CLIP](https://img.shields.io/badge/OpenAI%20CLIP-000000?style=for-the-badge&logo=openai&logoColor=white)
![SpeechRecognition](https://img.shields.io/badge/SpeechRecognition-FFCA28?style=for-the-badge&logo=python&logoColor=white)

## 🔗 Workflow Diagram

1. **Text Input:** Users type queries, and the chatbot responds with product suggestions or FAQ answers.
2. **Image Input:** Using AI models like CLIP, the chatbot analyzes uploaded images (like product photos) to find similar items.
3. **Voice Input:** Integrated speech-to-text library lets users speak their queries which makes interactions smooth and accessible.

```mermaid
---
config:
  theme: forest
  look: handDrawn
---
flowchart LR
    subgraph Data_Sources[Data Sources]
        A[Product Data:<br/>Titles, Prices, FAQs,<br/>Product URLs]
        B[Images]
    end

    subgraph Embeddings[Embedding Models]
        A--> C1[Product Data<br/>Embedding Model]
        B--> C2[Image Embedding<br/>Model]
    end
    C1 --> D1((Product Data Embeds))
    C2 --> D2((Image Embeds))
    subgraph Storage[Vector Store & URLs]
        D1 --> E[Vector Store]
        D2 --> E
        E --> F[Pickle File:<br/>Stores Product URLs]
    end
    subgraph Interaction[User & LLM]
        G((User))
        G -->|Text/Voice/Image Query| H[LLM - Groq & Llama3]
        H -->|Vector Search| E
        E -->|Relevant Info<br/>Similar Image Search / Product Data| H
        H -->|Final Response| G
    end
```


## Installation
This project uses Poetry for dependency management and packaging. Follow these steps to set up your environment:
1. **Clone the repository:**

   ```bash
   git clone https://github.com/emanalytic/MultiModal-E-Commerce-Customer-Support-Chatbot.git
   cd MultiModal-E-Commerce-Customer-Support-Chatbot
   ```
2. **Install Poetry:** (if not already installed)
   ```bash
   pipx install poetry
   ```
3. **Install project dependencies:**
   ```bash
   poetry install
   ```
## Running the Project
4. **Set up your environment variables:**
   ```bash
   GROQ_API_KEY= your_api_key_here  
   ```
5. **Run the Chainlit App locally:**
   ```bash
   poetry run chainlit run backend/chainlit_ui.py -w
   ```
   *For voice mode, run the app via terminal with `app.py` using Poetry as well.*

