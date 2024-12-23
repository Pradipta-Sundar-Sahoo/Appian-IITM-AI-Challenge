import streamlit as st
import json
import os
import nest_asyncio
from llama_parse import LlamaParse
from openai import OpenAI
import sqlite3
from datetime import datetime
import base64
from pathlib import Path
import redis
from functools import lru_cache
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import pickle
import os
import base64
import io
from docx import Document


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()


OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY']=OPENAI_API_KEY

LLAMA_CLOUD_API_KEY=os.getenv('LLAMA_CLOUD_API_KEY')
os.environ['LLAMA_CLOUD_API_KEY']=LLAMA_CLOUD_API_KEY

st.set_page_config(
        page_title="Document Categorizer",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
# Redis setup
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    result_type="markdown",
    num_workers=4,
    verbose=True,
    language="en"
)


st.markdown("""
<style>
    /* Main styling */
    .main {
        padding: 2rem;
        background-color: #121212;
        color: #E0E0E0;
    }
    
    /* Card styling */
    .stExpander {
        background-color: #1E1E1E;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
        border: none !important;
    }
    
    /* Status boxes */
    .status-box {
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #2E7D32;
        border-left: 4px solid #388E3C;
    }
    
    .info-box {
        background-color: #1565C0;
        border-left: 4px solid #1E88E5;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background-color: #D3D3D3;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2C2C2C;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 2.5rem;
        background-color: #2196F3;
        color: #E0E0E0;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #1976D2;
        transform: translateY(-2px);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #2196F3;
    }
    
    /* Document history cards */
    .document-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }
    
    /* PDF viewer */
    iframe {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(255, 255, 255, 0.1);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #2196F3;
        font-weight: 600;
    }
    
    /* Category filter styling */
    .stSelectbox {
        margin-bottom: 2rem;
        background-color: #2C2C2C;
        color: #E0E0E0;
    }
    
    /* Loading spinner */
    .stSpinner {
        text-align: center;
        color: #2196F3;
    }
</style>
""", unsafe_allow_html=True)




# Database setup with modified schema
def init_db():
    conn = sqlite3.connect('documents_new.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents_new
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        category TEXT NOT NULL,
        subcategory TEXT NOT NULL,
        important_info TEXT,
        pdf_data BLOB,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        all_extracted TEXT,
        summary TEXT
        )
    ''')
    conn.commit()
    conn.close()


def get_cache_key(function_name, *args):
    """Generate a cache key for Redis"""
    return f"{function_name}:{':'.join(str(arg) for arg in args)}"

def cache_result(key, data, expire_time=3600):
    """Cache data in Redis"""
    try:
        redis_client.setex(key, expire_time, json.dumps(data))
    except Exception as e:
        st.warning(f"Caching failed: {str(e)}")

def get_cached_result(key):
    """Retrieve cached data from Redis"""
    try:
        data = redis_client.get(key)
        return json.loads(data) if data else None
    except Exception as e:
        st.warning(f"Cache retrieval failed: {str(e)}")
        return None

def save_to_db(filename, category, subcategory, important_info, pdf_data,all_extracted):
    conn = sqlite3.connect('documents_new.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO documents_new (filename, category, subcategory, important_info, pdf_data, upload_date, all_extracted)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (filename, category, subcategory, important_info, pdf_data, datetime.now(),all_extracted))
    conn.commit()
    conn.close()

    # Invalidate relevant cache
    redis_client.delete(get_cache_key("get_all_documents"))
    redis_client.delete(get_cache_key("get_documents_by_category", category))
def get_documents_by_category(category):
    """Get documents for a specific main category"""
    cache_key = get_cache_key("get_documents_by_category", category)
    cached_result = get_cached_result(cache_key)
    
    if cached_result:
        return cached_result
    
    conn = sqlite3.connect('documents_new.db')
    c = conn.cursor()
    c.execute('''
        SELECT id, filename, category, subcategory, upload_date, all_extracted 
        FROM documents_new
        WHERE category = ?
        ORDER BY upload_date DESC
    ''', (category,))
    docs = c.fetchall()
    conn.close()
    
    cache_result(cache_key, docs)
    return docs

def get_all_documents():
    """Get all documents ordered by upload date (most recent first)"""
    cache_key = get_cache_key("get_all_documents")
    cached_result = get_cached_result(cache_key)
    
    if cached_result:
        return cached_result
    
    conn = sqlite3.connect('documents_new.db')
    c = conn.cursor()
    c.execute('''
        SELECT id, filename, category, subcategory, upload_date, all_extracted
        FROM documents_new 
        ORDER BY upload_date DESC
    ''')
    docs = c.fetchall()
    conn.close()
    
    cache_result(cache_key, docs)
    return docs

def display_pdf(file_data):
    """Display PDF or DOCX files in the sidebar based on file data"""
    
    if file_data[:4] == b'%PDF':
        # For PDF files, display it using iframe
        b64_pdf = base64.b64encode(file_data).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.sidebar.markdown(pdf_display, unsafe_allow_html=True)
    
    elif file_data[:2] == b'PK': 
        doc = Document(io.BytesIO(file_data))
        doc_text = "\n".join([para.text for para in doc.paragraphs])
        st.sidebar.markdown(f"### Document Content\n{doc_text}")
    
    else:
        st.sidebar.error("Unsupported file format")

@lru_cache(maxsize=100)
def get_document_by_id(doc_id):
    """Get specific document with local Python caching"""
    conn = sqlite3.connect('documents_new.db')
    c = conn.cursor()
    c.execute('SELECT * FROM documents_new WHERE id = ?', (doc_id,))
    doc = c.fetchone()
    conn.close()
    return doc



def cache_viewed_document(doc_id, pdf_data, important_info):
    """Cache PDF data and important info for viewed documents"""
    try:
        # Cache PDF data (convert to base64 for storage)
        pdf_cache_key = f"pdf_data:{doc_id}"
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        redis_client.setex(pdf_cache_key, 3600, pdf_base64)
        
        # Cache important info
        info_cache_key = f"important_info:{doc_id}"
        redis_client.setex(info_cache_key, 3600, important_info)
        
    except Exception as e:
        st.warning(f"Failed to cache document data: {str(e)}")

def get_cached_document_data(doc_id):
    """Retrieve cached PDF data and important info"""
    try:
        # Get cached PDF data
        pdf_cache_key = f"pdf_data:{doc_id}"
        cached_pdf = redis_client.get(pdf_cache_key)
        
        # Get cached important info
        info_cache_key = f"important_info:{doc_id}"
        cached_info = redis_client.get(info_cache_key)
        
        if cached_pdf and cached_info:
            # Convert PDF data back from base64
            pdf_data = base64.b64decode(cached_pdf)
            return pdf_data, cached_info
            
        return None, None
        
    except Exception as e:
        st.warning(f"Failed to retrieve cached document data: {str(e)}")
        return None, None

# Modify the get_document_by_id function
@lru_cache(maxsize=100)
def get_document_by_id(doc_id):
    """Get specific document with caching"""
    # Try to get from cache first
    cached_pdf, cached_info = get_cached_document_data(doc_id)
    
    if cached_pdf and cached_info:
        # Return in same format as database query
        # (id, filename, category, subcategory, important_info, pdf_data, upload_date)
        # Note: Some fields will be None since we only cache essential data
        return (doc_id, None, None, None, cached_info, cached_pdf, None)
    
    # If not in cache, get from database
    conn = sqlite3.connect('documents_new.db')
    c = conn.cursor()
    c.execute('SELECT * FROM documents_new WHERE id = ?', (doc_id,))
    doc = c.fetchone()
    conn.close()
    
    if doc:
        # Cache the document data for future use
        cache_viewed_document(doc_id, doc[5], doc[4])  # doc[5] is pdf_data, doc[4] is important_info
    
    return doc

# Modify the relevant part in your main() function where documents are viewed:


def classify_document(text, categories):
    """Classify document with main category focus"""
    categories_prompt = json.dumps(categories, indent=2)
    
    prompt = f"""Given the following categories and their subcategories:
    {categories_prompt}
    
    And the following document text:
    {text[:1500]}...
    
    Please determine the main category and subcategory this document belongs to.
    The main category is the top-level category (e.g., IdentityDocuments, FinancialDocuments, etc.).
    Return the response in the following format exactly:
    Main Category: [main category name]
    Subcategory: [specific subcategory name]"""

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a document classification expert. Respond only with the Main Category and Subcategory in the exact format specified."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    response_text = response.choices[0].message.content.strip()
    
    try:
        lines = response_text.split('\n')
        category = lines[0].split('Main Category: ')[1].strip()
        subcategory = lines[1].split('Subcategory: ')[1].strip()
        return category, subcategory
    except Exception as e:
        st.error(f"Error parsing classification response: {str(e)}")
        return None, None


def extract_important_info(text, category):
    """Extract important information based on document category"""
    prompt = f"""From the following document text, extract only the most important identifying information based on the document category ({category}).
    Focus only on:
    1. Personal identifiers (name, ID numbers if present)
    2. Contact information (email, phone if present)
    3. Key events (dates, locations, causes, etc.) if present. For example, in a legal document, you might want to extract
    4. Company information if any
    5. Bank name information if any
    4. Document-specific key dates
    5. Document reference numbers if any

    Document text:
    {text[:1500]}...

    Return only the found information in a simple list format with labels. If a piece of information is not found, skip it."""

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a document information extraction expert. Extract only the key identifying information requested."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content.strip()


def load_categories():
    with open('categories.json', 'r') as f:
        return json.load(f)


def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF using LlamaParse"""
    try:
        extra_info = {"file_name": uploaded_file.name}
        file_bytes = uploaded_file.read()
        documents = parser.load_data(file_bytes, extra_info=extra_info)
        
        if documents and len(documents) > 0:
            return documents[0].text
        else:
            st.error("No text could be extracted from the document")
            return None
            
    except Exception as e:
        st.error(f"Error parsing PDF: {str(e)}")
        return None


def save_summary_to_db(doc_id, summary):
    conn = sqlite3.connect('documents_new.db')
    c = conn.cursor()
    c.execute('''
        UPDATE documents_new 
        SET summary = ? 
        WHERE id = ?
    ''', (summary, doc_id))
    conn.commit()
    conn.close()
    
    # Invalidate relevant cache keys
    redis_client.delete(f"summary:{doc_id}")


def get_cached_summary(doc_id):
    """Get summary from cache or database"""
    # Try Redis cache first
    cache_key = f"summary:{doc_id}"
    cached_summary = redis_client.get(cache_key)
    
    if cached_summary:
        return cached_summary
        
    # If not in Redis, check database
    conn = sqlite3.connect('documents_new.db')
    c = conn.cursor()
    c.execute('SELECT summary FROM documents_new WHERE id = ?', (doc_id,))
    result = c.fetchone()
    conn.close()
    
    if result and result[0]:
        # Cache the summary in Redis
        redis_client.setex(cache_key, 3600, result[0])  # Cache for 1 hour
        return result[0]
        
    return None


def summarize_document(doc_id,text):
    cached_summary = get_cached_summary(doc_id)
    if cached_summary:
        st.success("🚀 Retrieved document from cache summary")
        return cached_summary
    
    """Summarize document content using GPT-4"""
    prompt = f"""Please provide a comprehensive summary of the following document. 
    Focus on the main points, key findings, and important details. 
    Keep the summary clear and concise while capturing the essential information.
    Provide in Structured Way with headings and bullet points wherever possible.
    Document text:
    {text}...
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert document summarizer. Create clear, concise, yet comprehensive summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        summary= response.choices[0].message.content.strip()
        save_summary_to_db(doc_id, summary)
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None
    


class VectorStoreManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
    
    def create_vectorstore(self, text, doc_id):
        """Create and save vector store for a document"""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Create vector store
        vectorstore = FAISS.from_texts(chunks, self.embeddings)
        
        # Save the index and documents separately
        index_path = f'faiss_index_{doc_id}'
        if not os.path.exists('document_stores'):
            os.makedirs('document_stores')
            
        # Save FAISS index
        vectorstore.save_local(f"document_stores/{index_path}")
        
        return vectorstore
    
    def get_vectorstore(self, doc_id):
        """Get vector store for a document"""
        try:
            index_path = f'faiss_index_{doc_id}'
            # Load the index if it exists
            if os.path.exists(f"document_stores/{index_path}"):
                vectorstore = FAISS.load_local(
                    f"document_stores/{index_path}", 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return vectorstore
            return None
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return None

def get_conversation_chain(vectorstore):
    """Create conversation chain for RAG"""
    llm = ChatOpenAI(temperature=0.7)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True
    )
    return conversation_chain

def save_chat_history(doc_id, chat_history):
    """Save chat history to Redis"""
    cache_key = f"chat_history:{doc_id}"
    try:
        redis_client.setex(cache_key, 3600, json.dumps(chat_history))  # Cache for 1 hour
    except Exception as e:
        st.warning(f"Failed to save chat history: {str(e)}")

def get_chat_history(doc_id):
    """Get chat history from Redis"""
    cache_key = f"chat_history:{doc_id}"
    try:
        chat_history = redis_client.get(cache_key)
        return json.loads(chat_history) if chat_history else []
    except Exception as e:
        st.warning(f"Failed to retrieve chat history: {str(e)}")
        return []


def init_chat_system():
    """Initialize the chat system with LangGraph"""
    # Initialize the chat model
    model = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-4o",
        api_key=OPENAI_API_KEY
    )
    
    # Create the graph
    workflow = StateGraph(state_schema=MessagesState)
    
    # Define the function that calls the model with summarization
    def call_model(state: MessagesState):
        system_prompt = (
            "You are a helpful assistant. "
            "Answer all questions about the document to the best of your ability. "
            "The provided chat history includes a summary of the earlier conversation."
        )
        system_message = SystemMessage(content=system_prompt)
        message_history = state["messages"][:-1]  # exclude the most recent user input
        
        # Summarize the messages if the chat history reaches a certain size
        if len(message_history) >= 4:
            last_human_message = state["messages"][-1]
            
            # Generate conversation summary
            summary_prompt = (
                "Distill the above chat messages into a single summary message. "
                "Include as many specific details as you can."
            )
            summary_message = model.invoke(
                message_history + [HumanMessage(content=summary_prompt)]
            )
            
            # Delete old messages
            delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
            # Re-add user message
            human_message = HumanMessage(content=last_human_message.content)
            # Call model with summary & response
            response = model.invoke([system_message, summary_message, human_message])
            message_updates = [summary_message, human_message, response] + delete_messages
        else:
            message_updates = model.invoke([system_message] + state["messages"])
            
        return {"messages": message_updates}
    
    # Define the node and edge
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    
    # Add simple in-memory checkpointer
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app
def save_chat_history_to_redis(doc_id, messages):
    """Save chat history to Redis"""
    try:
        cache_key = f"doc_chat_history:{doc_id}"
        # Convert messages to a format that can be JSON serialized
        serializable_messages = []
        for msg in messages:
            serializable_messages.append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": datetime.now().isoformat()
            })
        redis_client.setex(cache_key, 86400, json.dumps(serializable_messages))  # Store for 24 hours
    except Exception as e:
        st.warning(f"Failed to save chat history: {str(e)}")

def get_chat_history_from_redis(doc_id):
    """Retrieve chat history from Redis"""
    try:
        cache_key = f"doc_chat_history:{doc_id}"
        chat_history = redis_client.get(cache_key)
        if chat_history:
            return json.loads(chat_history)
        return []
    except Exception as e:
        st.warning(f"Failed to retrieve chat history: {str(e)}")
        return []


    

# Modified handle_chat function to incorporate chat history
def handle_chat(doc_id, extracted_text):
    """Handle document chat functionality with chat history"""
    # Initialize chat system if not already done
    chat_key = f"chat_state_{doc_id}"
    thread_key = f"thread_state_{doc_id}"
    messages_key = f"messages_state_{doc_id}"
    
    if chat_key not in st.session_state:
        st.session_state[chat_key] = init_chat_system()
    if thread_key not in st.session_state:
        st.session_state[thread_key] = f"thread_{doc_id}"
    if messages_key not in st.session_state:
        # Load chat history from Redis when initializing
        st.session_state[messages_key] = get_chat_history_from_redis(doc_id)
    
    chat_container = st.container()
    
    with chat_container:
        st.markdown("""
            <div class="status-box info-box">
                <h3>💬 Chat with Document</h3>
                <p>Ask questions about the document content</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state[messages_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        user_question = st.chat_input("Ask a question about the document...", key=f"chat_input_{doc_id}")
        
        if user_question:
            # Add user message to chat
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Add to message history
            st.session_state[messages_key].append({
                "role": "user",
                "content": user_question
            })
            
            # Save updated chat history to Redis
            save_chat_history_to_redis(doc_id, st.session_state[messages_key])
            
            # Prepare context and get response
            context_message = HumanMessage(
                content=f"Context from the document:\n{extracted_text}\n\nQuestion: {user_question}"
            )
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state[chat_key].invoke(
                            {"messages": [context_message]},
                            config={"configurable": {"thread_id": st.session_state[thread_key]}}
                        )
                        
                        if response and "messages" in response:
                            ai_messages = [msg for msg in response["messages"] if isinstance(msg, AIMessage)]
                            if ai_messages:
                                answer = ai_messages[-1].content
                                st.markdown(answer)
                                
                                # Add assistant response to history
                                st.session_state[messages_key].append({
                                    "role": "assistant",
                                    "content": answer
                                })
                                
                                # Save updated chat history to Redis
                                save_chat_history_to_redis(doc_id, st.session_state[messages_key])
                            else:
                                st.error("No response generated")
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        # Remove the failed message from history
                        st.session_state[messages_key].pop()
                        save_chat_history_to_redis(doc_id, st.session_state[messages_key])
        
        # Clear chat history button
        if st.button("Clear Chat History", key=f"clear_chat_{doc_id}"):
            st.session_state[messages_key] = []
            st.session_state[thread_key] = f"thread_{doc_id}"
            # Clear chat history from Redis
            redis_client.delete(f"doc_chat_history:{doc_id}")
            st.rerun()


def regenerate_summary(doc_id, text):
    """Force regenerate summary regardless of cache"""
    prompt = f"""Please provide a comprehensive summary of the following document. 
    Focus on the main points, key findings, and important details. 
    Keep the summary clear and concise while capturing the essential information.
    Provide in Structured Way with headings and bullet points wherever possible.
    Document text:
    {text}...
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert document summarizer. Create clear, concise, yet comprehensive summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        summary = response.choices[0].message.content.strip()
        
        # Clear existing cache
        redis_client.delete(f"summary:{doc_id}")
        
        # Save new summary to database
        save_summary_to_db(doc_id, summary)
        return summary
    except Exception as e:
        st.error(f"Error regenerating summary: {str(e)}")
        return None

def main():
    init_db()
    col1, col2, col3 = st.columns([1, 3, 1])

    # Create an empty space in the middle column and add the image
    with col2:
        placeholder = st.empty()
        placeholder.image("appian.jpg", use_container_width=True)

    # Custom sidebar header
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color: #F8F9FA;'>📑 Document Viewer</h1>
        </div>
    """, unsafe_allow_html=True)
    
    categories = load_categories()
    main_categories = list(categories.keys())
    
    # Enhanced category filter
    selected_category = st.sidebar.selectbox(
        "📂 Filter by Category",
        ["All"] + main_categories,
        help="Select a category to filter documents"
    )
    
    # Main content with enhanced layout
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("""
    <div style='background-color: #F8F9FA; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);'>
        <h2 style='color:#1976D2; margin-bottom: 0.5rem; font-size: 1.4rem;'>📤 Upload New Document</h2>
    </div>
""", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose a PDF file")
        
        if uploaded_file:
            file_details = {
                "📄 Filename": uploaded_file.name,
                "📋 File Type": uploaded_file.type,
                "📦 File Size": f"{uploaded_file.size / 1024:.2f} KB"
            }
            
            st.markdown("""
                <div class='uploadedFile'>
                    <h3 style='color: #1976D2; margin-bottom: 0 rem;'>File Details</h3>
                </div>
            """, unsafe_allow_html=True)
            
            for key, value in file_details.items():
                st.markdown(f"**{key}:** {value}")
            
            if st.button("🔍 Process Document"):
                with st.spinner('Analyzing document...'):
                    text = extract_text_from_pdf(uploaded_file)
                    if text:
                        category, subcategory = classify_document(text, categories)
                        important_info = extract_important_info(text, category)
                        pdf_data = uploaded_file.getvalue()
                        all_extracted= text
                        print(all_extracted)
                        save_to_db(uploaded_file.name, category, subcategory, important_info, pdf_data, all_extracted)
                        
                        st.markdown("""
                            <div class="status-box success-box">
                                <h3>✅ Document Processed Successfully!</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"**📂 Main Category:** {category}")
                        st.markdown(f"**📑 Subcategory:** {subcategory}")
                        
                        display_pdf(pdf_data)
                        st.balloons()
    with col2:
        st.markdown("""
        <div style='background-color: #F8F9FA; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2); margin-bottom: 2rem;'>
            <h2 style='color:#1976D2; margin-bottom: 0.5rem; font-size: 1.4rem;'>📚 Document History</h2>
        </div>
        """, unsafe_allow_html=True)

        if selected_category == "All":
            docs = get_all_documents()
        else:
            docs = get_documents_by_category(selected_category)

        if docs:
            for doc in docs:
                doc_id, filename, main_cat, subcat, upload_date, extracted_text = doc
                
                # Initialize session state keys
                summary_key = f"summary_{doc_id}"
                show_summary_key = f"show_summary_{doc_id}"
                
                if summary_key not in st.session_state:
                    st.session_state[summary_key] = None
                if show_summary_key not in st.session_state:
                    st.session_state[show_summary_key] = False

                with st.expander(f"📄 {filename}", expanded=False):
                    st.markdown("""
                        <style>
                            .document-card {
                                background-color: #424242;
                                padding: 1rem;
                                border-radius: 8px;
                                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                                margin-bottom: 1rem;
                            }
                        </style>
                    """, unsafe_allow_html=True)
                    st.markdown(f"**📅 Upload Date:** {upload_date}")
                    st.markdown(f"**📂 Main Category:** {main_cat}")
                    st.markdown(f"**📑 Subcategory:** {subcat}")
                    st.markdown("</div>", unsafe_allow_html=True)

                    btn_col1, btn_col2, btn_col3 = st.columns(3)

                    # View Document button
                    if btn_col1.button(f"👁️ View Document", key=f"view_{doc_id}"):
                        with st.spinner("Loading document..."):
                            full_doc = get_document_by_id(doc_id)
                            if full_doc:
                                cached_pdf, cached_info = get_cached_document_data(doc_id)
                                if cached_pdf:
                                    st.success("🚀 Retrieved document from cache")
                                display_pdf(full_doc[5])
                                st.markdown("""
                                    <div class="status-box info-box">
                                        <h3>📌 Important Information</h3>
                                    </div>
                                """, unsafe_allow_html=True)
                                st.info(full_doc[4])
                                st.download_button(
                                    label="📥 Download Document",
                                    data=full_doc[5],
                                    file_name=filename,
                                    mime="application/pdf"
                                )

                    # Summarize button
                    if btn_col2.button(f"📝 Summarize", key=f"summarize_{doc_id}"):
                        st.session_state[show_summary_key] = not st.session_state[show_summary_key]
                        if st.session_state[show_summary_key] and st.session_state[summary_key] is None:
                            with st.spinner("Generating summary..."):
                                summary = summarize_document(doc_id, extracted_text)
                                st.session_state[summary_key] = summary

                    # Display summary section if active
                    if st.session_state[show_summary_key]:
                        if st.session_state[summary_key]:
                            st.markdown("""
                                <div class="status-box info-box">
                                    <h3>📋 Document Summary</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            st.success(st.session_state[summary_key])

                            action_col1, action_col2 = st.columns(2)

                            # Copy summary button
                            if action_col1.button("📋 Copy Summary", key=f"copy_summary_{doc_id}"):
                                st.write(st.markdown(
                                    f'<textarea style="position: absolute; left: -9999px;">{st.session_state[summary_key]}</textarea>',
                                    unsafe_allow_html=True
                                ))
                                st.success("Summary copied to clipboard!")

                            # Regenerate summary button with container
                            with action_col2:
                                if st.button("🔄 Regenerate", key=f"regenerate_{doc_id}"):
                                    with st.spinner("Regenerating summary..."):
                                        new_summary = regenerate_summary(doc_id, extracted_text)
                                        if new_summary:
                                            st.session_state[summary_key] = new_summary
                                            st.success("Summary regenerated! 🔃 Reload to see the new summary!")

                    # Chat functionality
                    chat_key = f"chat_active_{doc_id}"
                    if chat_key not in st.session_state:
                        st.session_state[chat_key] = False
                        
                    if btn_col3.button(f"💬 Chat", key=f"chat_{doc_id}"):
                        st.session_state[chat_key] = not st.session_state[chat_key]

                    if st.session_state[chat_key]:
                        handle_chat(doc_id, extracted_text)

        else:
            st.info("📂 No documents found in this category.")

if __name__ == "__main__":
    main()