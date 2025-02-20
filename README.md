# MultiModal-E-Commerce-Customer-Support-Chatbot




## Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Chainlit](https://img.shields.io/badge/Chainlit-4B8BBE?style=for-the-badge&logo=react&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-FF6F00?style=for-the-badge&logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-00C853?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenAI CLIP](https://img.shields.io/badge/OpenAI%20CLIP-000000?style=for-the-badge&logo=openai&logoColor=white)
![SpeechRecognition](https://img.shields.io/badge/SpeechRecognition-FFCA28?style=for-the-badge&logo=python&logoColor=white)


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
