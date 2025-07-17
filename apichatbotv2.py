import os
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Document Chatbot API",
    description="API for a document-based chatbot with conversation memory",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

class ClearMemoryRequest(BaseModel):
    session_id: str

class DocumentChatbotWithMemory:
    def __init__(self):
        # Initialize Gemini LLM with more human-like settings
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,  # Higher temperature for more varied responses
            top_p=0.9,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            model_kwargs={
                "system_instruction": "Anda adalah asisten AI yang ramah dan membantu. Berbicaralah dengan gaya santai namun profesional dalam Bahasa Indonesia. Gunakan kalimat lengkap dan alami seperti manusia."
            }
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Dictionary to store multiple conversation memories
        self.memories = {}
        
        # Load and process documents
        self.vectorstore = self.load_documents()
    
    def load_documents(self):
        """Load and process documents from the 'data/documents' folder (markdown only)"""
        try:
            loader = DirectoryLoader(
                'data/documents',
                glob="**/*.md",  # Only load markdown files
                loader_cls=TextLoader
            )
            documents = loader.load()
            
            if not documents:
                raise ValueError("No markdown documents found in 'data/documents' folder")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            return FAISS.from_documents(text_splitter.split_documents(documents), self.embeddings)
            
        except Exception as e:
            print(f"Error loading documents: {e}")
            exit(1)
    
    def get_memory(self, session_id: str):
        """Get or create memory for a session with personalized settings"""
        if session_id not in self.memories:
            self.memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key='answer',
                human_prefix="Pengguna",
                ai_prefix="Asisten",
                memory_chat_template=(
                    "Percakapan sebelumnya dengan {session_id}:\n"
                    "{history}\n"
                    "Terakhir pengguna berkata: {input}\n"
                    "Ingatlah untuk merespon dengan ramah dan alami dalam Bahasa Indonesia"
                )
            )
        return self.memories[session_id]
    
    def humanize_response(self, response: str) -> str:
        """Make the response more polite and professional in Indonesian"""
        phrases = [
            "Berdasarkan informasi yang tersedia,",
            "Izinkan saya menjelaskan,",
            "Berikut penjelasan yang dapat saya sampaikan,",
            "Merujuk pada data yang ada,",
            "Berikut informasi yang dapat saya berikan,"
        ]
        
        unknown_phrases = [
            "Mohon maaf, saat ini saya belum memiliki informasi terkait hal tersebut.",
            "Maaf, saya belum menemukan jawaban yang sesuai. Silakan hubungi pihak terkait untuk informasi lebih lanjut.",
            "Mohon maaf, pengetahuan saya mengenai hal ini masih terbatas.",
            "Maaf, saya tidak mengerti pertanyaan Anda. Bisakah Anda menjelaskannya lebih detail?"
        ]
        
        # Check if it's an "I don't know" response
        if (
            "tidak memiliki informasi" in response
            or "tidak ditemukan" in response
            or "tidak mengerti" in response
            or "tidak paham" in response
        ):
            # Return only the unknown phrase, without opening phrase
            return random.choice(unknown_phrases)
        
        # Make language more polite and professional
        replacements = {
            "berdasarkan": "berdasarkan informasi yang tersedia",
            "tersebut": "yang dimaksud",
            "adalah": "merupakan",
            "apabila": "jika",
            "oleh karena itu": "oleh sebab itu"
        }
        
        for formal, polite in replacements.items():
            response = response.replace(formal, polite)
        
        # Only add opening phrase if not already present
        if random.random() > 0.5:
            if not any(response.strip().lower().startswith(p.lower()) for p in phrases):
                response = f"{random.choice(phrases)} {response[0].lower() + response[1:]}"

        return response
    
    def setup_qa_chain(self, session_id: str):
        """Set up conversational QA chain with human-like responses"""
        memory = self.get_memory(session_id)
        
        custom_prompt = PromptTemplate(
            input_variables=["chat_history", "question", "context"],
            template="""Anda adalah asisten yang ramah dan membantu. Gunakan informasi dari dokumen berikut untuk menjawab pertanyaan dengan bahasa yang natural dan manusiawi dalam Bahasa Indonesia. 
            Jika informasi tidak ditemukan dalam dokumen, jawab dengan sopan bahwa Anda tidak memiliki informasinya.

            Percakapan sebelumnya:
            {chat_history}
            
            Dokumen referensi:
            {context}
            
            Pertanyaan: {question}
            
            Jawaban yang alami dan membantu:"""
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            return_source_documents=True,
            output_key='answer'
        )

# Initialize the chatbot when the API starts
try:
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in .env file")
        
    print("Initializing Document Chatbot with Memory...")
    chatbot = DocumentChatbotWithMemory()
    print("Chatbot initialized successfully!")
except Exception as e:
    print(f"Failed to start chatbot: {str(e)}")
    exit(1)

@app.get("/")
def read_root():
    return {"message": "Document Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    try:
        if not chat_request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Handle clear memory command
        if chat_request.message.lower() == 'clear':
            chatbot.memories.pop(chat_request.session_id, None)
            return ChatResponse(
                response="Percakapan sudah direset!",
                session_id=chat_request.session_id
            )
        
        qa_chain = chatbot.setup_qa_chain(chat_request.session_id)
        result = qa_chain.invoke({"question": chat_request.message})
        
        # Humanize the response before returning
        humanized_response = chatbot.humanize_response(result['answer'])
        
        return ChatResponse(
            response=humanized_response,
            session_id=chat_request.session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_memory")
async def clear_memory(request: ClearMemoryRequest):
    try:
        chatbot.memories.pop(request.session_id, None)
        return {"message": f"Memori percakapan untuk sesi {request.session_id} telah dihapus"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)