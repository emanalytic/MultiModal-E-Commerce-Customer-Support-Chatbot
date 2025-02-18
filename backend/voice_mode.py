import speech_recognition as sr
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
import time

class VoiceShoppingAssistant:
    def __init__(self, vectorDB_path):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Load vector database
        self.vectorDB = FAISS.load_local(
            vectorDB_path,
            HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'),
            allow_dangerous_deserialization=True
        )
        
        # Initialize conversation chain and memory
        self.conversation_chain, self.memory = self.get_conversation_chain()
        
        # Calibrate microphone 
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)


    def get_conversation_chain(self):
        llm = ChatGroq(
            api_key='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',
            model_name='llama3-70b-8192',
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
            llm,
            self.vectorDB.as_retriever(),
            prompt_chatbot
        )

        document_chain = create_stuff_documents_chain(llm, prompt_chatbot)
        conversation_chain = create_retrieval_chain(context_retriever, document_chain)
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key='chat_history',
            output_key='result',
            input_key='input'
        )
        
        return conversation_chain, memory

    def listen_to_user(self):
        try:
            with self.microphone as source:
                print("\nListening... (Speak now)")
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=10)
                
            print("Processing speech...")
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
            
        except sr.WaitTimeoutError:
            print("No speech detected within timeout period.")
            return None
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand what you said.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

    def process_query(self, user_input):
        if not user_input:
            return "I'm sorry, I couldn't catch that. Could you please try again?"

        documents_retrieved = self.vectorDB.similarity_search(user_input, threshold=0.9)
        
        if not documents_retrieved:
            return "I apologize, but I don't have any information on that particular product. Could you please specify another product or feature you are interested in?"
        
        context = " ".join([doc.page_content for doc in documents_retrieved if hasattr(doc, 'page_content')])
        chain_input = {
            'input': user_input,
            'chat_history': self.memory.chat_memory.messages,
            'context': context
        }

        result = self.conversation_chain.invoke(chain_input)
        response = result['answer']
        
        # Save conversation context
        self.memory.save_context({'input': user_input}, {'result': response})
        
        return response

    def run(self):
        print("Voice Shopping Assistant is ready! Say 'exit' to quit.")
        try:
            while True:
                user_input = self.listen_to_user()
                if user_input:
                    # Check for exit command
                    if user_input.lower() in ['exit', 'quit']:
                        print("Exiting assistant.")
                        break

                    print('-' * 80)
                    response = self.process_query(user_input)
                    print('Assistant:', response)
                    print('-' * 80)
                time.sleep(0.5) 
                
        except KeyboardInterrupt:
            print("\nThank you for using Voice Shopping Assistant. Goodbye!")



def main():
    input_dict = {'vectorDB_path': 'faiss_index'}
    assistant = VoiceShoppingAssistant(input_dict['vectorDB_path'])
    assistant.run()

if __name__ == "__main__":
    main()