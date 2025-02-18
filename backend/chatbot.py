#%%
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
# from langchain import PromptTemplate
# from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.memory import ConversationBufferMemory
import dotenv
dotenv.load_dotenv()

def get_conversation_chain(text_vectorDB, image_vectorDB):
    llm = ChatGroq(api_key=API_KEY,
                   model_name='llama3-70b-8192', temperature=0)

    prompt_shopping_assistant = """
    *Welcome Message:*
    "Hi there! I'm your Shopping Assistant. I'll be happy to help in your shopping journey. How may I assist you today?"
    *Your role* ".

    *User Interaction:*
    - *Query Received*: {input}
    - *Your Database*: {context}
        Use this as your primary knowledge base.

    *Guidelines for Usage of Database:*

    1. I operate strictly within the bounds of our product database. If an inquiry is about topics outside our catalog, I will suggest alternatives or ask for more details about our products.
       For instance: "I can only assist with questions about our products. Can you specify which one you’re interested in?"
    2. If the context is irrelevant based on the user query or if the required product information is missing, don't refer to the mismatched context or make assumptions.
       Instead, directly ask the user to provide more specific details or shift their inquiry to a different product that is available in our database.
    3. When the context you get does not align with the user's query or if the required product information is not in our database, clearly state:
       "I apologize, but I don't have any information on that particular product. Could you please specify another product or feature you are interested in?"
       This keeps the conversation focused and encourages users to ask about other products we might have information on.

    *Strict Restrictions*:
        - I can only use the provided knowledge base to answer queries.
        - I am not allowed to provide any support except for product-related queries.
        - I will refuse any non-product-related queries without explanation.
        - I will not attempt to answer questions outside my product-related scope.
        - I will provide concise and direct responses.
        - I will strictly adhere to all the above guidelines.

    *Execution Standards*:
        - My responses will be succinct, precise, and tailored to your specific inquiries.
        - I will maintain a professional tone throughout our interaction.
        - I will use proper text formatting for better readability."""

    prompt_chatbot = ChatPromptTemplate.from_messages([('system', prompt_shopping_assistant),
                                                       MessagesPlaceholder('chat_history'), ('human', '{input}')])

    context_retriever = create_history_aware_retriever(llm,
                                                       vectorDB.as_retriever(), prompt_chatbot)

    document_chain = create_stuff_documents_chain(llm, prompt_chatbot)
    conversation_chain = create_retrieval_chain(context_retriever, document_chain)
    memory = ConversationBufferMemory(return_messages=True, memory_key='chat_history',
                                      output_key='result', input_key='input')
    return conversation_chain, memory


def main(input_dict):
    vectorDB_path = input_dict['vectorDB_path']
    vectorDB = FAISS.load_local(vectorDB_path,
                                HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'),
                                allow_dangerous_deserialization=True)

    conversation_chain, memory = get_conversation_chain(vectorDB)
    while True:
        user_input = input('User: ')

        documents_retrieved = vectorDB.similarity_search(user_input, threshold=0.9)
        if not documents_retrieved:
            response = "I apologize, but I don't have any information on that particular product. Could you please specify another product or feature you are interested in?"
        else:
            context = " ".join([doc.page_content for doc in documents_retrieved if 'page_content' in doc])
        chain_input = {'input': user_input, 'chat_history': memory.chat_memory.messages, 'context': context}

        result = conversation_chain.invoke(chain_input)
        response = result['answer']
        print('-' * 80)
        print('Assistant: ', response)
        print('-' * 80)
        memory.save_context({'input': user_input}, {'result': response})


input_dict = {'vectorDB_path': 'faiss_index'}
main(input_dict) 