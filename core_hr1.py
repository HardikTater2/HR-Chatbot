import os
import fitz
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, TypedDict, List, Any, Optional
from langchain_community.vectorstores import FAISS
from textblob import TextBlob
from supabase import create_client, Client
from datetime import datetime
import uuid
import logging
import re
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hr_chatbot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
TICKET_TYPES = ["leave", "lost_and_found", "reimbursement", "other"]
SUPPORT_PHONE = os.getenv("SUPPORT_PHONE", "7738085467")
EXIT_COMMANDS = [
    "exit", "quit", "cancel", "abort", "stop", "reset", "start over", 
    "new conversation", "clear", "restart", "nevermind", "forget it", "go back",
    "switch", "change agent", "different agent", "another agent"
]

# Initialize LLM and embeddings with error handling
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0.3,
        max_retries=3
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        max_retries=3
    )
    logger.info("LLM and embeddings initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM/embeddings: {e}")
    raise

# Text splitters for parent/child chunks
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, 
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def init_supabase() -> Client:
    """Initialize Supabase client with comprehensive error handling"""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials missing in environment variables")
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")
        
        client = create_client(supabase_url, supabase_key)
        
        # Test connection
        try:
            client.table("feedback").select("*").limit(1).execute()
            logger.info("Supabase client initialized and tested successfully")
        except Exception as test_error:
            logger.warning(f"Supabase connection test failed: {test_error}")
            logger.info("Supabase client initialized but connection not verified")
            
        return client
        
    except Exception as e:
        logger.error(f"Supabase initialization error: {e}")
        raise

# Initialize Supabase with error handling
try:
    supabase = init_supabase()
except Exception as e:
    logger.error(f"Failed to initialize Supabase: {e}")
    supabase = None

class DocumentStore:
    """Enhanced unified document store with parent-child retrieval and error handling"""
    
    def __init__(self):
        try:
            # Initialize with a meaningful dummy text instead of empty string
            dummy_text = "HR Policy Document Initialization"
            self.vector_store = FAISS.from_texts([dummy_text], embeddings)
            self.parent_docs = {}
            self.child_to_parent = {}
            self.is_initialized = False
            logger.info("DocumentStore initialized successfully")
        except Exception as e:
            logger.error(f"DocumentStore initialization failed: {e}")
            raise
        
    def add_document(self, policy_name: str, content: str) -> bool:
        """Process document with parent-child relationship and return success status"""
        try:
            if not content or not content.strip():
                logger.warning(f"Empty content for policy: {policy_name}")
                return False
                
            # Split into parent chunks
            parent_chunks = parent_splitter.split_text(content)
            if not parent_chunks:
                logger.warning(f"No parent chunks created for policy: {policy_name}")
                return False
                
            child_chunks = []
            
            for parent_id, parent_chunk in enumerate(parent_chunks):
                parent_key = f"{policy_name}_{parent_id}"
                self.parent_docs[parent_key] = parent_chunk
                
                # Create child chunks
                children = child_splitter.split_text(parent_chunk)
                child_chunks.extend(children)
                
                # Map children to parent
                for child in children:
                    self.child_to_parent[child] = parent_key
            
            # Add children to vector store
            if child_chunks:
                if not self.is_initialized:
                    # Replace dummy initialization with real content
                    self.vector_store = FAISS.from_texts(child_chunks, embeddings)
                    self.is_initialized = True
                else:
                    self.vector_store.add_texts(child_chunks)
                
                logger.info(f"Added {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks for {policy_name}")
                return True
            else:
                logger.warning(f"No child chunks created for policy: {policy_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding document {policy_name}: {e}")
            return False
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Retrieve parent documents via child chunks with enhanced error handling"""
        try:
            if not self.is_initialized:
                logger.warning("DocumentStore not properly initialized with documents")
                return []
                
            if not query or not query.strip():
                logger.warning("Empty query provided to search")
                return []
            
            # Search child chunks
            child_docs = self.vector_store.similarity_search(query, k=k*2)  # Get more results
            if not child_docs:
                logger.info(f"No child documents found for query: {query}")
                return []
            
            # Get unique parent IDs
            parent_ids = set()
            for doc in child_docs:
                parent_id = self.child_to_parent.get(doc.page_content, "")
                if parent_id:
                    parent_ids.add(parent_id)
            
            # Retrieve parent documents
            results = []
            for pid in list(parent_ids)[:k]:  # Limit to k results
                if pid in self.parent_docs:
                    results.append(self.parent_docs[pid])
            
            logger.info(f"Search returned {len(results)} parent documents for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

# Initialize unified document store
policy_store = DocumentStore()

def get_policy_paths() -> Dict[str, str]:
    """Get policy document paths from environment or use defaults"""
    base_path = os.getenv("HR_DOCS_PATH", os.path.join(os.getcwd(), "hr_docs"))
    
    return {
        "attendance": os.path.join(base_path, "Attendance Policy.pdf"),
        "leave": os.path.join(base_path, "Leave Policy.pdf"),
        "disciplinary": os.path.join(base_path, "DISCIPLINARY ACTION POLICY.pdf"),
        "medical_insurance": os.path.join(base_path, "Group Medical Insurance Policy.pdf"),
        "medical_parents": os.path.join(base_path, "Group Medical Insurance Policy Parents Coverage - An Optional Scheme.pdf"),
        "reimbursement": os.path.join(base_path, "GUIDELINE FOR CLAIMS & REIMBURSEMENTS.pdf"),
        "relocation": os.path.join(base_path, "RELOCATION POLICY.pdf"),
        "return_office": os.path.join(base_path, "Return to Office (RTO) 2.pdf")
    }


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF with enhanced error handling"""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return ""
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.warning(f"Empty file: {file_path}")
            return ""
        
        doc = fitz.open(file_path)
        full_text = ""
        page_count = 0
        
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():  # Only add non-empty pages
                full_text += page_text + "\n\n"
                page_count += 1
        
        doc.close()
        
        if not full_text.strip():
            logger.warning(f"No text extracted from {file_path}")
            return ""
        
        logger.info(f"Extracted text from {file_path}: {page_count} pages, {len(full_text)} characters")
        return full_text
        
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""

def load_policy_documents() -> bool:
    """Load all policy documents into unified store with comprehensive error handling"""
    policy_paths = get_policy_paths()
    loaded_count = 0
    total_count = len(policy_paths)
    
    logger.info(f"Starting to load {total_count} policy documents...")
    
    for policy_name, path in policy_paths.items():
        try:
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}")
                continue
            
            # Check file size
            file_size = os.path.getsize(path)
            if file_size == 0:
                logger.warning(f"Empty file: {path}")
                continue
                
            logger.info(f"Loading {policy_name} from {path} ({file_size} bytes)")
            
            # Open and extract text
            doc = fitz.open(path)
            full_text = ""
            page_count = 0
            
            for page in doc:
                page_text = page.get_text()
                if page_text.strip():  # Only add non-empty pages
                    full_text += page_text + "\n\n"
                    page_count += 1
            
            doc.close()
            
            if not full_text.strip():
                logger.warning(f"No text extracted from {policy_name}")
                continue
            
            # Add to document store
            if policy_store.add_document(policy_name, full_text):
                loaded_count += 1
                logger.info(f"Successfully loaded {policy_name} ({page_count} pages, {len(full_text)} characters)")
            else:
                logger.error(f"Failed to add {policy_name} to document store")
                
        except Exception as e:
            logger.error(f"Error loading {policy_name}: {str(e)}")
            continue
    
    success_rate = (loaded_count / total_count) * 100 if total_count > 0 else 0
    logger.info(f"Document loading complete: {loaded_count}/{total_count} files loaded ({success_rate:.1f}% success rate)")
    
    return loaded_count > 0

# Load all policies at startup
documents_loaded = load_policy_documents()
if not documents_loaded:
    logger.warning("No policy documents were loaded successfully")

class AgentState(TypedDict):
    input: str
    sender: str
    chat_history: List[Dict[str, str]]
    current_agent: str
    agent_output: str
    metadata: Dict[str, Any]

def generate_ticket_id() -> str:
    """Generate a unique ticket ID with timestamp and random component"""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    random_id = str(uuid.uuid4())[:8].upper()
    return f"TKT-{timestamp}-{random_id}"

def validate_email(email: str) -> bool:
    """Enhanced email validation with comprehensive regex"""
    if not email or not isinstance(email, str):
        return False
    
    # Clean the email
    email = email.strip().lower()
    
    # Comprehensive email regex
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Basic format check
    if not re.match(pattern, email):
        return False
    
    # Additional checks
    if email.count('@') != 1:
        return False
    
    local, domain = email.split('@')
    if len(local) == 0 or len(domain) == 0:
        return False
    
    return True

def validate_phone(phone: str) -> bool:
    """Enhanced phone number validation with international support"""
    if not phone or not isinstance(phone, str):
        return False
    
    # Clean phone number
    phone_clean = re.sub(r'[^\d+]', '', phone)
    
    # Remove leading + if present
    if phone_clean.startswith('+'):
        phone_clean = phone_clean[1:]
    
    # Check length (10-15 digits is standard for most countries)
    if len(phone_clean) < 10 or len(phone_clean) > 15:
        return False
    
    # Check if all remaining characters are digits
    return phone_clean.isdigit()

def analyze_sentiment(text: str) -> str:
    """Analyze sentiment with TextBlob and enhanced error handling"""
    try:
        if not text or not isinstance(text, str):
            return "neutral"
        
        # Clean text
        text = text.strip()
        if len(text) == 0:
            return "neutral"
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Enhanced sentiment classification
        if polarity > 0.3:
            return "positive"
        elif polarity < -0.3:
            return "negative"
        elif polarity > 0.1:
            return "slightly_positive"
        elif polarity < -0.1:
            return "slightly_negative"
        else:
            return "neutral"
            
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return "neutral"

def should_exit_context(user_input: str) -> bool:
    """ENHANCED: More responsive exit context detection to break sticky agents"""
    if not user_input or not isinstance(user_input, str):
        return False
    
    user_input_clean = user_input.lower().strip()
    
    # Direct matches - expanded list
    if user_input_clean in EXIT_COMMANDS:
        logger.info(f"Direct exit command detected: {user_input_clean}")
        return True
    
    # Enhanced exit patterns - more flexible matching
    exit_patterns = [
        r'\bexit\b', r'\bquit\b', r'\bcancel\b', r'\bstop\b',
        r'\breset\b', r'\brestart\b', r'\bstart over\b', 
        r'\bnew conversation\b', r'\bclear\b', r'\bgo back\b',
        r'\bswitch\b', r'\bchange agent\b', r'\bdifferent agent\b',
        r'\banother agent\b', r'\bback to main\b', r'\bmain menu\b',
        r'\bstart fresh\b', r'\bbegin again\b'
    ]
    
    for pattern in exit_patterns:
        if re.search(pattern, user_input_clean):
            logger.info(f"Exit pattern matched: {pattern}")
            return True
    
    return False

def should_switch_agent(user_input: str) -> bool:
    """ENHANCED: More comprehensive agent switching detection"""
    if not user_input or not isinstance(user_input, str):
        return False
    
    user_input_lower = user_input.lower()
    
    # Enhanced switching patterns - more comprehensive
    switch_patterns = [
        r'\bswitch to\b', r'\bchange to\b', r'\btalk to\b', r'\bconnect me to\b',
        r'\bi want to talk to\b', r'\bi want to speak with\b', 
        r'\bi need help with\b', r'\bcan you help with\b',
        r'\bgo to\b', r'\bmove to\b', r'\btake me to\b',
        r'\bi want to\s+(ask|create|give|provide|submit|report)\b',
        r'\bswitch agent\b', r'\bchange agent\b', r'\bdifferent agent\b',
        r'\banother agent\b', r'\bother agent\b'
    ]
    
    for pattern in switch_patterns:
        if re.search(pattern, user_input_lower):
            logger.info(f"Agent switch pattern detected: {pattern}")
            return True
    
    return False

def detect_explicit_agent_request(user_input: str) -> str:
    """NEW: Detect explicit agent requests to break sticky states"""
    if not user_input or not isinstance(user_input, str):
        return ""
    
    user_input_lower = user_input.lower()
    
    # Explicit agent requests
    agent_patterns = {
        "feedback": [
            r'\bi want to give feedback\b', r'\bi want to provide feedback\b',
            r'\bfeedback\b', r'\bgive feedback\b', r'\bprovide feedback\b',
            r'\bshare feedback\b', r'\bleave feedback\b'
        ],
        "ticket": [
            r'\bcreate ticket\b', r'\bnew ticket\b', r'\bsubmit ticket\b',
            r'\breport issue\b', r'\bi have an issue\b', r'\bproblem\b',
            r'\bhelp me with\b', r'\bneed help\b'
        ],
        "faq": [
            r'\bask question\b', r'\bwhat is\b', r'\bhow to\b', r'\bpolicy\b',
            r'\binformation\b', r'\bexplain\b', r'\btell me about\b'
        ],
        "live": [
            r'\bspeak to human\b', r'\btalk to person\b', r'\bhuman support\b',
            r'\blive agent\b', r'\brepresentative\b'
        ]
    }
    
    for agent, patterns in agent_patterns.items():
        for pattern in patterns:
            if re.search(pattern, user_input_lower):
                logger.info(f"Explicit {agent} agent request detected: {pattern}")
                return agent
    
    return ""

def detect_feedback_intent(user_input: str) -> bool:
    """ENHANCED: Comprehensive feedback intent detection"""
    if not user_input or not isinstance(user_input, str):
        return False
    
    user_input_lower = user_input.lower().strip()
    
    # HIGHEST PRIORITY: Direct feedback expressions
    direct_feedback_patterns = [
        r'\bi want to give feedback\b',
        r'\bi want to provide feedback\b',
        r'\bi would like to give feedback\b',
        r'\bi would like to provide feedback\b',
        r'\bi need to give feedback\b',
        r'\bcan i give feedback\b',
        r'\bcan i provide feedback\b',
        r'\bgive feedback\b',
        r'\bprovide feedback\b',
        r'\bshare feedback\b',
        r'\bleave feedback\b',
        r'\bsubmit feedback\b',
        r'\bfeedback about\b',
        r'\bfeedback on\b',
        r'\bmy feedback\b',
        r'\bsome feedback\b',
        r'\bhave feedback\b'
    ]
    
    for pattern in direct_feedback_patterns:
        if re.search(pattern, user_input_lower):
            logger.info(f"FEEDBACK INTENT DETECTED: Pattern matched - {pattern}")
            return True
    
    # Secondary: Opinion/sentiment expressions without action requests
    sentiment_words = [
        "good", "great", "excellent", "amazing", "wonderful", "fantastic", "awesome",
        "bad", "terrible", "awful", "horrible", "disappointing", "poor", "worst",
        "love", "hate", "like", "dislike", "satisfied", "unsatisfied", "happy", "angry"
    ]
    
    # Action words that suggest tickets instead of feedback
    action_words = [
        "fix", "solve", "resolve", "help me", "need help", "can't", "unable", 
        "issue", "problem", "error", "bug", "broken", "not working"
    ]
    
    has_sentiment = any(word in user_input_lower for word in sentiment_words)
    has_action = any(word in user_input_lower for word in action_words)
    
    # If sentiment without action request, it's likely feedback
    if has_sentiment and not has_action:
        logger.info("FEEDBACK INTENT DETECTED: Sentiment-based detection")
        return True
    
    return False

def clear_agent_context(state: AgentState) -> None:
    """NEW: Clear agent-specific context while preserving session info"""
    session_id = state["metadata"].get("session_id", str(uuid.uuid4()))
    logger.info(f"Clearing agent context for session: {session_id}")
    
    # Preserve only essential session information
    state["metadata"] = {
        "session_id": session_id,
        "context_cleared": True,
        "clear_time": datetime.now().isoformat()
    }
    state["current_agent"] = ""

def router(state: AgentState) -> str:
    try:
        meta = state.get("metadata", {})
        user_input = state["input"].lower().strip() if state.get("input") else ""
        
        if not user_input:
            return "faq"

        logger.info(f"Router processing: '{user_input}'")

        # PRIORITY 1: Handle exit commands (highest priority to break sticky states)
        if should_exit_context(user_input):
            logger.info("Exit command detected - clearing context and routing to FAQ")
            clear_agent_context(state)
            return "faq"

        # PRIORITY 2: Handle explicit agent switching requests
        if should_switch_agent(user_input):
            logger.info("Agent switch request detected - clearing context")
            clear_agent_context(state)
            
            # Detect target agent
            explicit_agent = detect_explicit_agent_request(user_input)
            if explicit_agent:
                logger.info(f"Switching to explicitly requested agent: {explicit_agent}")
                return explicit_agent

        # PRIORITY 3: Check for explicit agent requests (can override sticky states)
        explicit_agent = detect_explicit_agent_request(user_input)
        if explicit_agent:
            logger.info(f"Explicit agent request detected: {explicit_agent}")
            # Clear context if switching to different agent
            if state.get("current_agent") and state.get("current_agent") != explicit_agent:
                clear_agent_context(state)
            return explicit_agent

        # PRIORITY 4: Handle pending states (but allow breaking out)
        # FIXED: Made pending states less sticky
        if meta.get("awaiting_ticket_confirmation") and not (
            should_exit_context(user_input) or 
            should_switch_agent(user_input) or 
            detect_feedback_intent(user_input)
        ):
            # Only stay in ticket if user is responding to confirmation
            confirmation_keywords = ["yes", "y", "confirm", "create", "proceed", "ok", "sure", "no", "n", "cancel"]
            if any(keyword in user_input for keyword in confirmation_keywords):
                return "ticket"
            else:
                # User said something else, break out of confirmation
                logger.info("Breaking out of ticket confirmation due to unrelated input")
                clear_agent_context(state)
        
        elif meta.get("collecting_feedback") and not (
            should_exit_context(user_input) or 
            should_switch_agent(user_input) or 
            detect_explicit_agent_request(user_input)
        ):
            # Less sticky feedback collection
            if len(user_input.split()) > 2:  # Likely providing feedback
                return "feedback"
            else:
                # Short response might be switching intent
                logger.info("Breaking out of feedback collection due to short response")
                clear_agent_context(state)

        # PRIORITY 5: Feedback intent detection
        if detect_feedback_intent(user_input):
            logger.info("🎯 FEEDBACK INTENT DETECTED - routing to feedback agent")
            return "feedback"

        # PRIORITY 6: Enhanced keyword scoring with balanced weights
        keyword_scores = {"faq": 0, "ticket": 0, "feedback": 0, "live": 0}
        
        # FAQ keywords
        faq_keywords = [
            "information", "what is", "how to", "policy", "rule", "procedure",
            "when is", "where is", "explain", "define", "tell me about",
            "details", "requirements", "guidelines", "handbook", "attendance",
            "leave", "medical", "insurance", "reimbursement", "relocation",
            "what", "how", "when", "where", "why", "who"
        ]
        
        # Ticket keywords (balanced weight)
        ticket_keywords = [
            "issue", "problem", "ticket", "help me with", "bug", "technical",
            "system", "portal", "login", "access", "error", "wrong", "missing",
            "broken", "not working", "can't access", "unable to", "fix", "solve",
            "resolve", "urgent", "emergency", "submit request", "need help"
        ]
        
        # EXPANDED feedback keywords (higher weight)
        feedback_keywords = [
            "suggest", "improve", "compliment", "complain", "opinion", "review", 
            "rate", "experience", "service", "quality", "satisfied", "disappointed", 
            "recommend", "think about", "good", "bad", "great", "terrible", 
            "excellent", "poor", "amazing", "awful", "wonderful", "horrible", 
            "fantastic", "disappointing", "love", "hate", "like", "dislike", 
            "happy", "angry", "pleased", "frustrated", "impressed", "concerned", 
            "suggestion", "comment", "praise", "criticism", "thoughts", "feelings", 
            "food", "canteen", "service", "staff", "environment", "facilities"
        ]
        
        # Live agent keywords
        live_keywords = [
            "speak to someone", "talk to human", "human support", "representative",
            "call me", "phone call", "urgent", "emergency", "escalate"
        ]
        
        # Score based on keyword matches
        for keyword in faq_keywords:
            if keyword in user_input:
                keyword_scores["faq"] += 1
                
        for keyword in ticket_keywords:
            if keyword in user_input:
                keyword_scores["ticket"] += 1
                
        for keyword in feedback_keywords:
            if keyword in user_input:
                keyword_scores["feedback"] += 1.5  # Higher weight for feedback
        
        for keyword in live_keywords:
            if keyword in user_input:
                keyword_scores["live"] += 2  # Higher weight for live agent
        
        # Get highest scoring agent
        best_agent = max(keyword_scores, key=keyword_scores.get)
        if keyword_scores[best_agent] > 0:
            logger.info(f"Keyword-based routing to {best_agent}: scores={keyword_scores}")
            return best_agent

        # PRIORITY 7: LLM-based classification for ambiguous queries
        try:
            classification_prompt = f"""
            Analyze this user message and classify it into the most appropriate category.
            
            User Message: "{state['input']}"
            
            Categories and Guidelines:
            
            1. **feedback** - Choose this for:
               - ANY mention of "feedback" or "give feedback"
               - Opinions about food, services, workplace, policies
               - Positive/negative comments (good, bad, great, terrible, etc.)
               - Suggestions for improvement
               - Compliments or complaints
               - Reviews or ratings
               - Expressions of satisfaction/dissatisfaction
               - Comments about experiences
            
            2. **ticket** - Choose this for:
               - Specific problems needing resolution
               - Technical issues or errors
               - Requests for administrative action
               - Help with access or system problems
               - Lost items or missing resources
               - Urgent matters requiring intervention
            
            3. **faq** - Choose this for:
               - Questions about policies or procedures
               - Requests for information
               - "What is", "How to", "When is" questions
               - General inquiries about company rules
            
            4. **live** - Choose this for:
               - Complex issues requiring human intervention
               - Requests to speak with someone
               - Escalation requests
            
            CRITICAL RULE: If the message contains "I want to give feedback" or any variation of wanting to provide feedback, classify as "feedback".
            
            Examples:
            - "I want to give feedback" → feedback
            - "I want to provide feedback about service" → feedback
            - "The food is very good" → feedback
            - "I love the new office space" → feedback  
            - "The system is not working" → ticket
            - "What is the leave policy?" → faq
            
            Respond with ONLY the category name (feedback/ticket/faq/live).
            """
            
            response = llm.invoke(classification_prompt)
            agent = response.content.strip().lower()
            
            if agent in ["faq", "ticket", "feedback", "live"]:
                logger.info(f"LLM classified query as: {agent}")
                return agent
            else:
                logger.warning(f"LLM returned invalid classification: {agent}")
                return "faq"
                
        except Exception as e:
            logger.error(f"LLM routing error: {e}")
            return "faq"
            
    except Exception as e:
        logger.error(f"Router error: {e}")
        return "faq"

def search_knowledge_base(query: str, k: int = 3) -> str:
    """Search unified document store with enhanced error handling and context formatting"""
    try:
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided to search_knowledge_base")
            return ""
        
        query = query.strip()
        if not query:
            return ""
        
        logger.info(f"Searching knowledge base for: {query}")
        results = policy_store.search(query, k=k)
        
        if not results:
            logger.info(f"No results found for query: {query}")
            return ""
        
        # Format results with separators
        formatted_results = "\n\n" + "="*50 + "\n\n"
        formatted_results += formatted_results.join(results)
        
        logger.info(f"Knowledge base search returned {len(results)} results")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Knowledge base search error: {e}")
        return ""

def faq_agent(state: AgentState) -> Dict[str, Any]:
    """Enhanced FAQ handler with improved policy classification and response generation"""
    try:
        user_query = state.get("input", "").strip()
        if not user_query:
            return {
                "agent_output": "I didn't receive a question. Please ask me about HR policies or company information.",
                "metadata": {}
            }

        logger.info(f"FAQ agent processing query: {user_query}")

        # Enhanced policy classification
        classification_prompt = f"""
        You are an HR representative for Yash Technology. Analyze this query and classify it into the most relevant HR policy category.
        
        Query: "{user_query}"
        
        Available policy categories:
        - "attendance" - Work schedules, time tracking, tardiness, absences
        - "leave" - Vacation, sick leave, personal time, leave requests
        - "disciplinary" - Code of conduct, violations, corrective actions
        - "medical_insurance" - Health benefits, coverage, claims
        - "medical_parents" - Parental health insurance coverage
        - "reimbursement" - Expense claims, travel, medical reimbursements
        - "relocation" - Office moves, work location changes
        - "return_office" - Return to office policies, hybrid work
        - "general" - If query doesn't fit specific categories
        
        Respond with ONLY the most appropriate category keyword in quotes.
        """
        
        try:
            classification_response = llm.invoke(classification_prompt)
            policy_type = classification_response.content.strip().strip('"').lower()
            logger.info(f"Query classified as: {policy_type}")
        except Exception as e:
            logger.error(f"Policy classification error: {e}")
            policy_type = "general"
        
        # Retrieve context using enhanced search
        context = search_knowledge_base(user_query, k=4)
        
        if not context:
            return {
                "agent_output": (
                    "I couldn't find specific information about that topic in our HR policies. "
                    "This might be because:\n"
                    "• The policy documents aren't loaded yet\n"
                    "• Your question needs rephrasing\n"
                    "• It's a topic not covered in our current policies\n\n"
                    f"For immediate assistance, please contact HR at {SUPPORT_PHONE} or create a support ticket."
                ),
                "metadata": {"policy_type": policy_type, "context_found": False}
            }

        # Generate comprehensive response
        response_prompt = f"""
        You are a helpful HR AI Assistant for Yash Technology. Use the provided context to answer the employee's question accurately and helpfully.
        
        CONTEXT FROM HR POLICIES:
        {context}
        
        EMPLOYEE QUESTION: {user_query}
        
        IDENTIFIED POLICY AREA: {policy_type}
        
        INSTRUCTIONS:
        1. Answer directly and concisely (4-6 sentences)
        2. Reference specific policy names or sections when available
        3. Use bullet points for multiple items or steps
        4. If the context doesn't fully answer the question, acknowledge this
        5. Maintain a professional but friendly tone
        6. Never make up information not in the context
        7. Suggest contacting HR for complex cases
        
        RESPONSE:
        """
        
        try:
            response = llm.invoke(response_prompt)
            agent_output = response.content.strip()
            
            # Add helpful footer for complex queries
            if len(user_query.split()) > 10 or "complex" in user_query.lower():
                agent_output += f"\n\n💡 For detailed assistance with complex cases, contact HR at {SUPPORT_PHONE}"
            
            # Add agent switching help
            agent_output += "\n\n🔄 Type 'switch' or 'change agent' to switch to a different service."
            
            logger.info("FAQ response generated successfully")
            return {
                "agent_output": agent_output,
                "metadata": {
                    "policy_type": policy_type,
                    "context_found": True,
                    "response_length": len(agent_output)
                }
            }
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return {
                "agent_output": (
                    "I found relevant information but had trouble generating a response. "
                    f"Please contact HR directly at {SUPPORT_PHONE} for assistance with your question."
                ),
                "metadata": {"policy_type": policy_type, "error": "response_generation"}
            }
        
    except Exception as e:
        logger.error(f"FAQ agent error: {e}")
        return {
            "agent_output": (
                "I'm experiencing technical difficulties. Please try rephrasing your question "
                f"or contact HR directly at {SUPPORT_PHONE}."
            ),
            "metadata": {"error": str(e)}
        }

def ticket_agent(state: AgentState) -> Dict[str, Any]:
    """Enhanced ticket creation with robust state management and non-sticky behavior"""
    try:
        meta = state.get("metadata", {})
        user_input = state["input"].strip().lower() if state.get("input") else ""
        
        logger.info(f"Ticket agent processing: {user_input}")

        # ENHANCED: Handle exit/switch commands even during confirmation
        if should_exit_context(user_input) or should_switch_agent(user_input):
            logger.info("Exit/switch command detected in ticket agent")
            clear_agent_context(state)
            return {
                "agent_output": "✅ Ticket process cancelled. How else can I help you?",
                "metadata": {}
            }

        # Handle confirmation flow
        if meta.get("awaiting_ticket_confirmation"):
            confirmation_keywords = ["yes", "y", "confirm", "create", "proceed", "ok", "sure"]
            cancellation_keywords = ["no", "n", "cancel", "abort", "nevermind"]
            
            if any(keyword in user_input for keyword in confirmation_keywords):
                try:
                    ticket_data = meta.get("pending_ticket", {})
                    
                    # Enhance ticket data with required fields
                    ticket_data.update({
                        "created_at": datetime.now().isoformat(),
                        "sender": state.get("sender", "anonymous"),
                        "original_query": meta.get("original_query", state["input"]),
                        "session_id": str(uuid.uuid4()),
                        "priority": meta.get("priority", "normal"),
                        "status": "open"
                    })

                    if supabase:
                        # Insert ticket data and handle response
                        result = supabase.table("tickets").insert(ticket_data).execute()
                        
                        # Properly extract ticket ID from response
                        if hasattr(result, 'data') and result.data:
                            db_ticket_id = result.data[0].get('id') or generate_ticket_id()
                        else:
                            db_ticket_id = generate_ticket_id()

                        success_message = (
                            f"🎫 **Ticket Created Successfully!**\n\n"
                            f"• **Ticket ID:** {db_ticket_id}\n"
                            f"• **Type:** {ticket_data['ticket_type'].replace('_', ' ').title()}\n"
                            f"• **Status:** Open\n"
                            f"• **Priority:** {ticket_data.get('priority', 'normal').title()}\n\n"
                            f"📞 **Next Steps:**\n"
                            f"• Our team will review your request\n"
                            f"• You'll receive updates within 24-48 hours\n"
                            f"• For urgent matters, call {SUPPORT_PHONE}\n\n"
                            f"Your ticket ID for reference: **{db_ticket_id}**\n\n"
                            f"🔄 Type 'switch' to use a different service or ask another question!"
                        )
                        
                        logger.info(f"Ticket created successfully: {db_ticket_id}")
                    else:
                        # Fallback without database
                        fallback_id = generate_ticket_id()
                        success_message = (
                            f"🎫 **Ticket Request Noted!**\n\n"
                            f"• **Reference ID:** {fallback_id}\n"
                            f"• **Type:** {ticket_data['ticket_type'].replace('_', ' ').title()}\n\n"
                            f"⚠️ Note: Database unavailable. Please contact HR directly:\n"
                            f"📞 Phone: {SUPPORT_PHONE}\n"
                            f"Reference this ID: {fallback_id}\n\n"
                            f"🔄 Type 'switch' to use a different service!"
                        )
                        logger.warning("Ticket created without database storage")

                    return {
                        "agent_output": success_message,
                        "metadata": {}  # Clear metadata to make non-sticky
                    }
                    
                except Exception as e:
                    logger.error(f"Ticket creation error: {e}")
                    return {
                        "agent_output": (
                            "❌ **Ticket Creation Failed**\n\n"
                            "There was a technical issue creating your ticket. Please:\n"
                            f"• Try again in a few minutes\n"
                            f"• Contact HR directly at {SUPPORT_PHONE}\n"
                            f"• Email: hr@yashtechnologies.com\n\n"
                            "Would you like to try creating the ticket again? (yes/no)\n"
                            "Or type 'switch' to use a different service."
                        ),
                        "metadata": {}  # Clear metadata on error
                    }
                    
            elif any(keyword in user_input for keyword in cancellation_keywords):
                logger.info("Ticket creation declined by user")
                return {
                    "agent_output": (
                        "✅ No problem! Ticket creation cancelled.\n\n"
                        "What else can I help you with?\n"
                        "🔄 Type 'switch' to use a different service!"
                    ),
                    "metadata": {}
                }
            else:
                # ENHANCED: Check if user is trying to switch agents
                if detect_feedback_intent(user_input) or detect_explicit_agent_request(user_input):
                    logger.info("User switching agents during ticket confirmation")
                    clear_agent_context(state)
                    return {
                        "agent_output": (
                            "✅ Ticket creation cancelled. Switching to your requested service...\n"
                            "Please repeat your request."
                        ),
                        "metadata": {}
                    }
                
                return {
                    "agent_output": (
                        "Please confirm your ticket creation:\n"
                        "• Type **'yes'** to create the ticket\n"
                        "• Type **'no'** to cancel\n"
                        "• Type **'switch'** to use a different service\n\n"
                        "What would you like to do?"
                    ),
                    "metadata": meta
                }

        # New ticket creation flow
        try:
            # Enhanced ticket classification
            classification_prompt = f"""
            Classify this support request into the most appropriate category:
            
            Categories: {", ".join(TICKET_TYPES)}
            
            Support Request: "{state['input']}"
            
            Guidelines:
            - "leave" - Leave requests, vacation, sick days, time off
            - "lost_and_found" - Missing items, lost property, found objects
            - "reimbursement" - Expense claims, travel costs, medical reimbursements
            - "other" - General issues, technical problems, miscellaneous requests
            
            Respond with ONLY the most appropriate category name.
            """

            response = llm.invoke(classification_prompt)
            ticket_type = response.content.strip().lower()
            
            # Validate ticket type
            if ticket_type not in TICKET_TYPES:
                ticket_type = "other"
                
            logger.info(f"Ticket classified as: {ticket_type}")
            
        except Exception as e:
            logger.error(f"Ticket classification error: {e}")
            ticket_type = "other"

        # Enhanced priority detection
        high_priority_keywords = ["urgent", "emergency", "asap", "immediately", "critical", "important"]
        priority = "high" if any(
            re.search(rf"\b{keyword}\b", state["input"].lower()) 
            for keyword in high_priority_keywords
        ) else "normal"

        # Prepare comprehensive ticket data with required fields
        ticket_data = {
            "sender": state.get("sender", "anonymous"),
            "query": state["input"],
            "ticket_type": ticket_type,
            "status": "open",
            "priority": priority
        }

        # Create detailed confirmation message
        confirmation_message = (
            f"🎫 **Ready to Create Your Ticket**\n\n"
            f"**Details:**\n"
            f"• **Type:** {ticket_type.replace('_', ' ').title()}\n"
            f"• **Priority:** {priority.title()}\n"
            f"• **Issue:** {state['input']}\n"
            f"• **Estimated Resolution:** 24-48 hours\n\n"
            f"**Next Steps:**\n"
            f"• Our team will review your request\n"
            f"• You'll receive status updates\n"
            f"• For urgent matters, call {SUPPORT_PHONE}\n\n"
            f"**Confirm ticket creation:**\n"
            f"• Type **'yes'** to create\n"
            f"• Type **'no'** to cancel\n"
            f"• Type **'switch'** to use a different service"
        )

        return {
            "agent_output": confirmation_message,
            "metadata": {
                "awaiting_ticket_confirmation": True,
                "pending_ticket": ticket_data,
                "original_query": state["input"],
                "priority": priority
            }
        }
        
    except Exception as e:
        logger.error(f"Ticket agent error: {e}")
        return {
            "agent_output": (
                f"❌ Sorry, I'm having technical difficulties with ticket creation.\n\n"
                f"Please contact HR directly:\n"
                f"📞 Phone: {SUPPORT_PHONE}\n"
                f"📧 Email: hr@yashtechnologies.com\n\n"
                f"Reference your issue: {state.get('input', 'Technical support needed')}\n\n"
                f"🔄 Type 'switch' to try a different service!"
            ),
            "metadata": {}
        }

def feedback_agent(state: AgentState) -> Dict[str, Any]:
    """Enhanced feedback collection with sentiment analysis and non-sticky behavior"""
    try:
        user_input = state["input"].strip() if state.get("input") else ""
        meta = state.get("metadata", {})
        
        logger.info(f"Feedback agent processing: {user_input}")
        
        # ENHANCED: Handle exit/switch commands even during feedback collection
        if should_exit_context(user_input) or should_switch_agent(user_input):
            logger.info("Exit/switch command detected in feedback agent")
            clear_agent_context(state)
            return {
                "agent_output": "✅ Feedback collection cancelled. What would you like to do instead?",
                "metadata": {}
            }

        # Check if user expressed direct feedback intent
        if detect_feedback_intent(user_input) and not meta.get("collecting_feedback"):
            return {
                "agent_output": (
                    "💭 **Perfect! I'm ready to collect your feedback.**\n\n"
                    "Please share your thoughts about:\n"
                    "• HR services and processes\n"
                    "• Company policies\n"
                    "• Workplace experience\n"
                    "• Food and facilities\n"
                    "• Any suggestions for improvement\n\n"
                    "What specific feedback would you like to share?\n"
                    "🔄 Type 'switch' anytime to use a different service."
                ),
                "metadata": {
                    "collecting_feedback": True,
                    "original_query": state["input"],
                    "feedback_start_time": datetime.now().isoformat()
                }
            }

        # Initial feedback collection prompt
        if not meta.get("collecting_feedback"):
            return {
                "agent_output": (
                    "💭 **We Value Your Feedback!**\n\n"
                    "Please share your thoughts about:\n"
                    "• HR services and processes\n"
                    "• Company policies\n"
                    "• Workplace experience\n"
                    "• Food and facilities\n"
                    "• Any suggestions for improvement\n\n"
                    "Your feedback helps us improve our services. What would you like to share?\n"
                    "🔄 Type 'switch' anytime to use a different service."
                ),
                "metadata": {
                    "collecting_feedback": True,
                    "original_query": state["input"],
                    "feedback_start_time": datetime.now().isoformat()
                }
            }

        # Process the feedback
        feedback_text = state["input"]
        
        # ENHANCED: Check if feedback is too short (might be switching intent)
        if len(feedback_text.split()) < 3:
            return {
                "agent_output": (
                    "Please provide more detailed feedback to help us improve our services.\n\n"
                    "Or if you'd like to switch to a different service, just type 'switch'.\n"
                    "What would you like to share?"
                ),
                "metadata": meta
            }
        
        try:
            # Analyze sentiment
            sentiment = analyze_sentiment(feedback_text)
            
            # Categorize feedback
            categorization_prompt = f"""
            Categorize this feedback into the most relevant category:
            
            Feedback: "{feedback_text}"
            
            Categories:
            - "hr_services" - HR processes, support, responsiveness
            - "policies" - Company policies, procedures, rules
            - "workplace" - Office environment, culture, facilities
            - "food_services" - Canteen, cafeteria, meals, food quality
            - "benefits" - Insurance, leave, compensation, perks
            - "technology" - IT systems, tools, software
            - "management" - Leadership, communication, decisions
            - "general" - Other feedback not fitting specific categories
            
            Respond with ONLY the category name.
            """
            
            try:
                cat_response = llm.invoke(categorization_prompt)
                category = cat_response.content.strip().lower()
                valid_categories = ["hr_services", "policies", "workplace", "food_services", 
                                  "benefits", "technology", "management", "general"]
                if category not in valid_categories:
                    category = "general"
            except Exception as e:
                logger.error(f"Feedback categorization error: {e}")
                category = "general"
            
            # Prepare comprehensive feedback data
            feedback_data = {
                "sender": state.get("sender", "anonymous"),
                "feedback_text": feedback_text,
                "sentiment": sentiment,
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "session_id": str(uuid.uuid4()),
                "word_count": len(feedback_text.split()),
                "character_count": len(feedback_text)
            }
            
            # Store feedback if database is available
            storage_success = False
            if supabase:
                try:
                    supabase.table("feedback").insert(feedback_data).execute()
                    storage_success = True
                    logger.info(f"Feedback stored successfully: {sentiment} sentiment, {category} category")
                except Exception as e:
                    logger.error(f"Feedback storage error: {e}")
            
            # Generate response based on sentiment and category
            if sentiment in ["positive", "slightly_positive"]:
                response_base = "🌟 **Thank you for the positive feedback!**\n\n"
                if category == "food_services":
                    response_base += "We're delighted you're enjoying our food services!"
                elif category == "hr_services":
                    response_base += "We're pleased our HR services are meeting your expectations!"
                elif category == "workplace":
                    response_base += "It's wonderful to hear you're enjoying the workplace environment!"
                else:
                    response_base += "Your positive comments really motivate our team!"
                    
            elif sentiment in ["negative", "slightly_negative"]:
                response_base = "🔧 **Thank you for bringing this to our attention.**\n\n"
                response_base += "We take all feedback seriously and will review your concerns. "
                if category == "food_services":
                    response_base += "Your feedback will be shared with our catering team."
                elif category in ["hr_services", "policies"]:
                    response_base += "Our team will analyze this feedback to improve our processes."
                    
            else:  # neutral
                response_base = "📝 **Thank you for your feedback.**\n\n"
                response_base += "We appreciate you taking the time to share your thoughts with us."
            
            # Add category-specific information
            category_messages = {
                "hr_services": "\n\n💡 For immediate HR assistance, contact us at " + SUPPORT_PHONE,
                "policies": "\n\n📋 Policy suggestions are reviewed quarterly by our leadership team.",
                "food_services": "\n\n🍽️ Food service feedback is shared with our catering team.",
                "technology": "\n\n💻 Technical feedback is forwarded to our IT department.",
                "benefits": "\n\n🎯 Benefits feedback helps us enhance our employee programs."
            }
            
            if category in category_messages:
                response_base += category_messages[category]
            
            # Add storage status
            if storage_success:
                response_base += "\n\n✅ Your feedback has been recorded and will be reviewed by the appropriate team."
            else:
                response_base += "\n\n⚠️ Note: Feedback noted but not stored due to technical issues."
            
            # Add switching help
            response_base += "\n\n🔄 Type 'switch' to use a different service or ask another question!"
            
            return {
                "agent_output": response_base,
                "metadata": {}  # Clear metadata to make non-sticky
            }
            
        except Exception as e:
            logger.error(f"Feedback processing error: {e}")
            return {
                "agent_output": (
                    "❌ **Feedback Processing Error**\n\n"
                    "There was a technical issue processing your feedback, but we've noted your input:\n\n"
                    f"*\"{feedback_text}\"*\n\n"
                    f"Please consider contacting HR directly at {SUPPORT_PHONE} to ensure your feedback is properly recorded.\n\n"
                    f"🔄 Type 'switch' to try a different service!"
                ),
                "metadata": {}
            }
            
    except Exception as e:
        logger.error(f"Feedback agent error: {e}")
        return {
            "agent_output": (
                f"❌ Sorry, I'm experiencing technical difficulties with feedback collection.\n\n"
                f"Please share your feedback directly with HR:\n"
                f"📞 Phone: {SUPPORT_PHONE}\n"
                f"📧 Email: hr@yashtechnologies.com\n\n"
                f"🔄 Type 'switch' to try a different service!"
            ),
            "metadata": {}
        }

def live_agent(state: AgentState) -> Dict[str, Any]:
    """Enhanced live agent routing with multiple contact options"""
    try:
        contact_info = (
            "👨‍💼 **Connect with Human Support**\n\n"
            "**Immediate Assistance:**\n"
            f"📞 **Phone:** {SUPPORT_PHONE}\n"
            f"📧 **Email:** hr@yashtechnologies.com\n\n"
            "**Office Hours:**\n"
            "🕘 Monday - Friday: 9:00 AM - 6:00 PM\n"
            "🕘 Saturday: 10:00 AM - 2:00 PM\n"
            "🕘 Sunday: Closed\n\n"
            "**For Urgent Issues:**\n"
            "• Call the main number above\n"
            "• Email with 'URGENT' in subject line\n"
            "• Visit HR office directly\n\n"
            "**What to Prepare:**\n"
            "• Employee ID or details\n"
            "• Description of your issue\n"
            "• Any relevant documents\n\n"
            "Our team is ready to help with complex issues that need personal attention!\n\n"
            "🔄 Type 'switch' to use a different service anytime."
        )
        
        logger.info("Live agent contact information provided")
        
        return {
            "agent_output": contact_info,
            "metadata": {}  # Non-sticky
        }
        
    except Exception as e:
        logger.error(f"Live agent error: {e}")
        return {
            "agent_output": f"📞 For human assistance, please call {SUPPORT_PHONE}\n\n🔄 Type 'switch' for other services.",
            "metadata": {}
        }

def workflow(state: AgentState) -> Dict[str, Any]:
    """FIXED: Non-sticky workflow with enhanced agent switching"""
    try:
        # Initialize state with comprehensive defaults
        state.setdefault("chat_history", [])
        state.setdefault("metadata", {})
        state.setdefault("current_agent", "")
        state.setdefault("sender", "anonymous")
        state.setdefault("input", "")
        
        # Validate input
        if not state["input"] or not isinstance(state["input"], str):
            return {
                "agent_output": "Please provide a message or question for me to help you with.",
                "metadata": state["metadata"],
                "chat_history": state["chat_history"]
            }

        # Add user message to history with metadata
        user_message = {
            "role": "user", 
            "content": state["input"],
            "timestamp": datetime.now().isoformat(),
            "session_id": state["metadata"].get("session_id", str(uuid.uuid4()))
        }
        state["chat_history"].append(user_message)

        user_input = state["input"].lower().strip()
        
        logger.info(f"Workflow processing: {user_input[:100]}..." if len(user_input) > 100 else user_input)
        
        # Handle reset commands with confirmation
        if user_input in ["reset", "start over", "restart"]:
            return {
                "agent_output": (
                    "✨ **Session Reset Complete!**\n\n"
                    "Starting fresh! I'm here to help with:\n"
                    "• 📋 HR policy questions (FAQ)\n"
                    "• 🎫 Support tickets and issues\n"
                    "• 💭 Feedback and suggestions\n"
                    "• 👨‍💼 Connect with human support\n\n"
                    "What can I help you with today?\n"
                    "🔄 You can always type 'switch' to change services!"
                ),
                "metadata": {
                    "session_id": str(uuid.uuid4()),
                    "reset_time": datetime.now().isoformat()
                },
                "chat_history": []
            }

        # ENHANCED: More flexible routing logic - less sticky behavior
        should_route = (
            not state["current_agent"] or  # No current agent
            should_exit_context(user_input) or  # Exit command
            should_switch_agent(user_input) or  # Switch command
            detect_explicit_agent_request(user_input) or  # Explicit agent request
            # FIXED: Allow breaking out of sticky states more easily
            (
                state["metadata"].get("awaiting_ticket_confirmation") and 
                not any(keyword in user_input for keyword in ["yes", "y", "confirm", "create", "proceed", "ok", "sure", "no", "n", "cancel"])
            ) or
            (
                state["metadata"].get("collecting_feedback") and 
                len(user_input.split()) < 3  # Short responses might be switching intent
            )
        )

        # Route to appropriate agent
        if should_route:
            new_agent = router(state)
            
            # Log agent changes
            if state["current_agent"] and state["current_agent"] != new_agent:
                logger.info(f"Agent switch: {state['current_agent']} -> {new_agent}")
            
            state["current_agent"] = new_agent
            
            # ENHANCED: More aggressive context clearing for better switching
            if (should_exit_context(user_input) or 
                should_switch_agent(user_input) or 
                detect_explicit_agent_request(user_input)):
                clear_agent_context(state)

        # Enhanced agent mapping with error handling
        agent_map = {
            "faq": faq_agent,
            "ticket": ticket_agent,
            "feedback": feedback_agent,
            "live": live_agent
        }
        
        # Execute agent with fallback
        current_agent = state["current_agent"]
        agent_func = agent_map.get(current_agent, faq_agent)
        
        logger.info(f"Executing {current_agent} agent")
        
        # Execute agent
        response = agent_func(state)
        
        # Validate agent response
        if not isinstance(response, dict):
            logger.error(f"Invalid response from {current_agent} agent")
            response = {
                "agent_output": "I encountered an error processing your request. Please try again.",
                "metadata": {}
            }
        
        # Update state with response
        state["agent_output"] = response.get("agent_output", "No response generated.")
        
        # ENHANCED: More careful metadata merging to avoid sticky states
        if "metadata" in response and response["metadata"]:
            state["metadata"].update(response["metadata"])
        elif "metadata" in response and not response["metadata"]:
            # Empty metadata means clear sticky state
            session_id = state["metadata"].get("session_id", str(uuid.uuid4()))
            state["metadata"] = {"session_id": session_id}
        
        # Add agent response to history
        agent_message = {
            "role": "agent",
            "agent_type": current_agent,
            "content": state["agent_output"],
            "timestamp": datetime.now().isoformat(),
            "metadata": response.get("metadata", {})
        }
        state["chat_history"].append(agent_message)

        # Log successful completion
        logger.info(f"Workflow completed successfully with {current_agent} agent")

        return {
            "agent_output": state["agent_output"],
            "metadata": state["metadata"],
            "chat_history": state["chat_history"],
            "current_agent": current_agent
        }
        
    except Exception as e:
        logger.error(f"Workflow error: {e}")
        
        # Ensure we have a valid chat history even on error
        if "chat_history" not in state:
            state["chat_history"] = []
        
        error_response = {
            "agent_output": (
                "❌ **System Error**\n\n"
                "I encountered a technical issue. Please:\n"
                "• Try rephrasing your request\n"
                "• Check your connection\n"
                f"• Contact support at {SUPPORT_PHONE} if the issue persists\n\n"
                "I'm here to help once the issue is resolved!\n"
                "🔄 Type 'switch' to try a different service."
            ),
            "metadata": state.get("metadata", {"error": str(e)}),
            "chat_history": state.get("chat_history", []),
            "current_agent": state.get("current_agent", "faq")
        }
        
        return error_response

def diagnose_feedback_system():
    """Comprehensive diagnostic tool for debugging the entire system"""
    print("🔍 **HR Chatbot System Diagnostics**\n")
    
    # 1. Environment Variables Check
    print("1. **Environment Variables:**")
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GOOGLE_API_KEY"]
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}: {value[:20]}...")
        else:
            print(f"   ❌ {var}: Missing")
    
    # 2. Document Loading Check
    print("\n2. **Document Loading:**")
    policy_paths = get_policy_paths()
    loaded_docs = 0
    for name, path in policy_paths.items():
        if os.path.exists(path):
            print(f"   ✅ {name}: Found")
            loaded_docs += 1
        else:
            print(f"   ❌ {name}: Missing ({path})")
    
    print(f"   📊 Total: {loaded_docs}/{len(policy_paths)} documents available")
    
    # 3. Vector Store Check
    print("\n3. **Vector Store:**")
    if policy_store.is_initialized:
        print(f"   ✅ Initialized with {len(policy_store.parent_docs)} parent documents")
        print(f"   📄 Child mappings: {len(policy_store.child_to_parent)}")
    else:
        print("   ❌ Not properly initialized")
    
    # 4. Database Connection
    print("\n4. **Database Connection:**")
    if supabase:
        try:
            result = supabase.table("feedback").select("*").limit(1).execute()
            print("   ✅ Supabase connection working")
        except Exception as e:
            print(f"   ⚠️ Connection issue: {e}")
    else:
        print("   ❌ Supabase not initialized")
    
    # 5. LLM Check
    print("\n5. **LLM Services:**")
    try:
        test_response = llm.invoke("Test message")
        print("   ✅ Google Generative AI working")
    except Exception as e:
        print(f"   ❌ LLM error: {e}")
    
    # 6. Test Search
    print("\n6. **Search Functionality:**")
    test_results = search_knowledge_base("leave policy")
    if test_results:
        print(f"   ✅ Search working ({len(test_results)} characters returned)")
    else:
        print("   ❌ Search not returning results")
    
    print("\n✅ **Diagnostic Complete!**")

def test_agent_switching():
    """NEW: Test agent switching and non-sticky behavior"""
    print("🔄 **Testing Agent Switching (Non-Sticky Behavior)**\n")
    
    test_scenarios = [
        {
            "scenario": "Feedback to FAQ switch",
            "steps": [
                ("I want to give feedback", "feedback"),
                ("What is the leave policy?", "faq")
            ]
        },
        {
            "scenario": "Ticket to Feedback switch",
            "steps": [
                ("I have an issue", "ticket"),
                ("I want to give feedback", "feedback")
            ]
        },
        {
            "scenario": "Exit from sticky state",
            "steps": [
                ("I have an issue", "ticket"),
                ("cancel", "faq")
            ]
        },
        {
            "scenario": "Switch command",
            "steps": [
                ("I want to give feedback", "feedback"),
                ("switch", "faq")
            ]
        }
    ]
    
    results = {"passed": 0, "failed": 0}
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"**Scenario {i}:** {scenario['scenario']}")
        
        # Initialize fresh state
        test_state = {
            "chat_history": [],
            "metadata": {"session_id": str(uuid.uuid4())},
            "current_agent": "",
            "sender": "test_user"
        }
        
        scenario_passed = True
        
        for step_num, (user_input, expected_agent) in enumerate(scenario["steps"], 1):
            print(f"   Step {step_num}: \"{user_input}\" → Expected: {expected_agent}")
            
            try:
                test_state["input"] = user_input
                result = workflow(test_state)
                actual_agent = test_state.get("current_agent", "unknown")
                
                if actual_agent == expected_agent:
                    print(f"      ✅ Got: {actual_agent}")
                else:
                    print(f"      ❌ Got: {actual_agent} (Expected: {expected_agent})")
                    scenario_passed = False
                    
            except Exception as e:
                print(f"      💥 Error: {e}")
                scenario_passed = False
        
        if scenario_passed:
            print(f"   🎉 **Scenario PASSED**")
            results["passed"] += 1
        else:
            print(f"   ❌ **Scenario FAILED**")
            results["failed"] += 1
        print()
    
    # Summary
    total_scenarios = len(test_scenarios)
    print("📊 **Agent Switching Test Summary:**")
    print(f"   ✅ Passed: {results['passed']}/{total_scenarios}")
    print(f"   ❌ Failed: {results['failed']}/{total_scenarios}")
    
    success_rate = (results['passed'] / total_scenarios) * 100 if total_scenarios > 0 else 0
    print(f"   📈 Success Rate: {success_rate:.1f}%")
    
    if results['passed'] == total_scenarios:
        print("\n🎉 **ALL SWITCHING TESTS PASSED! Sticky agent issue is FIXED!**")
    else:
        print(f"\n⚠️ **{results['failed']} scenarios failed. Review switching logic.**")

def test_feedback_routing_fix():
    """Enhanced test for feedback routing"""
    print("🧪 **Testing Feedback Routing Fix**\n")
    
    test_cases = [
        {
            "input": "I want to give feedback",
            "expected_agent": "feedback",
            "description": "Direct feedback request - PRIMARY FIX"
        },
        {
            "input": "I want to provide feedback about the service",
            "expected_agent": "feedback",
            "description": "Feedback with context"
        },
        {
            "input": "Can I give feedback?",
            "expected_agent": "feedback",
            "description": "Question form feedback request"
        },
        {
            "input": "I would like to share feedback",
            "expected_agent": "feedback",
            "description": "Polite feedback request"
        },
        {
            "input": "The food is really good",
            "expected_agent": "feedback",
            "description": "Sentiment-based feedback"
        },
        {
            "input": "I have an issue to report",
            "expected_agent": "ticket",
            "description": "Should be ticket (control test)"
        },
        {
            "input": "What is the leave policy?",
            "expected_agent": "faq",
            "description": "Should be FAQ (control test)"
        }
    ]
    
    results = {"passed": 0, "failed": 0}
    
    for i, test in enumerate(test_cases, 1):
        print(f"**Test {i}:** {test['description']}")
        print(f"   Input: \"{test['input']}\"")
        
        try:
            # Create test state
            state = {
                "input": test["input"],
                "sender": "test_user",
                "chat_history": [],
                "current_agent": "",
                "metadata": {}
            }
            
            # Test the router directly
            detected_agent = router(state)
            
            # Test feedback detection function
            is_feedback = detect_feedback_intent(test["input"])
            
            # Check results
            if detected_agent == test["expected_agent"]:
                print(f"   ✅ **PASSED** - Agent: {detected_agent}")
                results["passed"] += 1
            else:
                print(f"   ❌ **FAILED** - Expected: {test['expected_agent']}, Got: {detected_agent}")
                results["failed"] += 1
            
            print(f"   🎯 Feedback intent detected: {is_feedback}")
            
        except Exception as e:
            print(f"   💥 **ERROR** - {str(e)}")
            results["failed"] += 1
        
        print()
    
    # Summary
    total_tests = len(test_cases)
    print("📊 **Test Summary:**")
    print(f"   ✅ Passed: {results['passed']}/{total_tests}")
    print(f"   ❌ Failed: {results['failed']}/{total_tests}")
    
    success_rate = (results['passed'] / total_tests) * 100 if total_tests > 0 else 0
    print(f"   📈 Success Rate: {success_rate:.1f}%")
    
    if results['passed'] == total_tests:
        print("\n🎉 **ALL TESTS PASSED! Feedback routing is FIXED!**")
    else:
        print(f"\n⚠️ **{results['failed']} tests failed. Review the fixes needed.**")