from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from backend.config.config import Config
from dotenv import load_dotenv
import os
load_dotenv()

class ChatbotHandler:
    def __init__(self, vectorDB_path):
        config = Config()
        self.api_key = os.getenv("API_KEY")

        self.llm = config.get('settings', 'llm')
        self.model = config.get('settings', 'model')
        self.vectorDB = LangchainFAISS.load_local(
            vectorDB_path,
            HuggingFaceEmbeddings(model_name=self.model),
            allow_dangerous_deserialization=True
        )
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key='chat_history',
            output_key='result',
            input_key='input'
        )
        self.conversation_chain = self.setup_conversation_chain()


    def setup_conversation_chain(self):
        llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.llm,
            temperature=0
        )
        with open('prompt.txt', 'r') as file:
            prompt_text = file.read().strip()

        prompt_chatbot = ChatPromptTemplate.from_messages([
            ('system', prompt_text),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ])

        context_retriever = create_history_aware_retriever(
            llm, self.vectorDB.as_retriever(), prompt_chatbot
        )
        document_chain = create_stuff_documents_chain(llm, prompt_chatbot)
        return create_retrieval_chain(context_retriever, document_chain)

    def process_text_query(self, user_input):
        documents_retrieved = self.vectorDB.similarity_search(user_input, threshold=0.9)
        if not documents_retrieved:
            return "I don't have information on that. Can you specify another product or feature?"

        context = " ".join([doc.page_content for doc in documents_retrieved])
        result = self.conversation_chain.invoke({
            'input': user_input,
            'chat_history': self.memory.chat_memory.messages,
            'context': context
        })
        self.memory.save_context({'input': user_input}, {'result': result['answer']})
        return result['answer']

