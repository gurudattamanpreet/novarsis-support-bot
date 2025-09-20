from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import json
import time
import random
import google.generativeai as genai
from datetime import datetime, timedelta
import base64
import io
from PIL import Image
import math
import logging
from typing import Optional, List, Dict
import hashlib
import html
import uvicorn
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Novarsis Support Center", description="AI Support Assistant for Novarsis SEO Tool")

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDPAxeBhQ_OENApv3It8ccLbDeVUG0aVVA")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini Flash 2.0 model
try:
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    logger.info("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini model: {str(e)}")
    model = None

# Initialize embedding model - with error handling for quota issues
reference_embedding = None
embedding_model = None
try:
    embedding_model = 'models/embedding-001'
    reference_text = "Novarsis AIO SEO Tool support, SEO analysis, website analysis, meta tags, page structure, link analysis, SEO check, SEO report, subscription, account, billing, plan, premium, starter, error, bug, issue, problem, not working, failed, crash, login, password, analysis, report, dashboard, settings, integration, Google, API, website, URL, scan, audit, optimization, mobile, speed, performance, competitor, ranking, keywords, backlinks, technical SEO, canonical, schema, sitemap, robots.txt, crawl, index, search console, analytics, traffic, organic, SERP"
    reference_embedding = genai.embed_content(
        model=embedding_model,
        content=reference_text,
        task_type="retrieval_document",
    )['embedding']
    logger.info("Embedding model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize embedding model: {str(e)}")
    # Continue without embedding functionality
    reference_embedding = None

# Constants
WHATSAPP_NUMBER = "+91-9999999999"
SUPPORT_EMAIL = "support@novarsis.tech"

# Enhanced System Prompt
SYSTEM_PROMPT = """You are Nova, the official AI support assistant for Novarsis AIO SEO Tool.

PERSONALITY:
- Natural and conversational like a human
- Friendly and approachable
- Brief but complete responses
- Polite and professional

INTRO RESPONSES:
- Who are you? → "I'm Nova, your AI assistant for Novarsis SEO Tool. I help users with SEO analysis, reports, account issues, and technical support."
- How can you help? → "I can help you with SEO website analysis, generating reports, fixing errors, managing subscriptions, and troubleshooting any Novarsis tool issues."
- What can you do? → "I assist with all Novarsis features - SEO audits, competitor analysis, keyword tracking, technical issues, billing, and more."

SCOPE:
Answer ALL questions naturally, but stay within Novarsis context:
• Greetings → Respond naturally (Hello! How can I help you today?)
• About yourself → Explain your role as Novarsis assistant
• Capabilities → List what you can help with
• Tool features → Explain Novarsis features
• Technical help → Provide solutions
• Account/billing → Assist with subscriptions

ONLY REDIRECT for completely unrelated topics like:
- Cooking recipes, travel advice, general knowledge
- Non-SEO tools or competitors
- Personal advice unrelated to SEO

For unrelated queries, politely say:
"Sorry, I only help with Novarsis SEO Tool.
Please let me know if you have any SEO tool related questions?"

RESPONSE STYLE:
- Natural conversation flow
- Answer directly without overthinking
- 2-4 lines for simple queries
- Use simple, everyday language"""

# Context-based quick reply suggestions
QUICK_REPLY_SUGGESTIONS = {
    "initial": [
        "How do I analyze my website SEO?",
        "Check my subscription status",
        "I'm getting an error message",
        "Generate SEO report",
        "Compare pricing plans",
        "Check ticket status",
        "Connect with an Expert"
    ],
    "seo_analysis": [
        "How to improve my SEO score?",
        "What are meta tags?",
        "Check page load speed",
        "Analyze competitor websites",
        "Mobile optimization tips"
    ],
    "account": [
        "Upgrade my plan",
        "Reset my password",
        "View billing history",
        "Cancel subscription",
        "Update payment method"
    ],
    "technical": [
        "API integration help",
        "Report not generating",
        "Login issues",
        "Data sync problems",
        "Browser compatibility",
        "Connect with an Expert"
    ],
    "report": [
        "Schedule automatic reports",
        "Export to PDF",
        "Share report with team",
        "Customize report sections",
        "Historical data comparison"
    ],
    "error": [
        "Website not loading",
        "Analysis stuck at 0%",
        "404 error on dashboard",
        "Payment failed",
        "Can't access reports",
        "Connect with an Expert"
    ],
    "pricing": [
        "What's included in Premium?",
        "Student discount available?",
        "Annual vs monthly billing",
        "Team plans pricing",
        "Free trial details"
    ]
}


def get_context_suggestions(message: str) -> list:
    """Get relevant quick reply suggestions based on user's input context."""
    if not message or len(message.strip()) < 2:
        return QUICK_REPLY_SUGGESTIONS["initial"]

    message_lower = message.lower().strip()

    # Return empty if message is very short
    if len(message_lower) < 3:
        return []

    # Check for specific actions first
    if any(word in message_lower for word in ['ticket', 'status', 'track', 'nvs']):
        return ["Check ticket status", "Connect with an Expert", "View all tickets", "Create new ticket"]
    elif any(word in message_lower for word in ['expert', 'human', 'agent', 'support', 'help']):
        return ["Connect with an Expert", "Check ticket status", "Call support", "Email support"]
    # Check for keywords and return appropriate suggestions
    elif any(
            word in message_lower for word in ['seo', 'analysis', 'analyze', 'score', 'optimization', 'meta', 'crawl']):
        return QUICK_REPLY_SUGGESTIONS["seo_analysis"]
    elif any(word in message_lower for word in
             ['account', 'subscription', 'plan', 'billing', 'payment', 'upgrade', 'cancel']):
        return QUICK_REPLY_SUGGESTIONS["account"]
    elif any(word in message_lower for word in
             ['error', 'issue', 'problem', 'not working', 'failed', 'stuck', 'broken']):
        return QUICK_REPLY_SUGGESTIONS["error"]
    elif any(word in message_lower for word in ['report', 'export', 'pdf', 'schedule', 'download']):
        return QUICK_REPLY_SUGGESTIONS["report"]
    elif any(word in message_lower for word in ['api', 'integration', 'technical', 'login', 'password']):
        return QUICK_REPLY_SUGGESTIONS["technical"]
    elif any(word in message_lower for word in ['price', 'pricing', 'cost', 'plan', 'cheap', 'expensive', 'free']):
        return QUICK_REPLY_SUGGESTIONS["pricing"]
    elif any(word in message_lower for word in ['how', 'what', 'why', 'when', 'where']):
        # For question words, show initial helpful suggestions
        return QUICK_REPLY_SUGGESTIONS["initial"]
    else:
        return []


# Novarsis Keywords - expanded for better detection
NOVARSIS_KEYWORDS = [
    'novarsis', 'seo', 'website analysis', 'meta tags', 'page structure', 'link analysis',
    'seo check', 'seo report', 'subscription', 'account', 'billing', 'plan', 'premium',
    'starter', 'error', 'bug', 'issue', 'problem', 'not working', 'failed', 'crash',
    'login', 'password', 'analysis', 'report', 'dashboard', 'settings', 'integration',
    'google', 'api', 'website', 'url', 'scan', 'audit', 'optimization', 'mobile', 'speed',
    'performance', 'competitor', 'ranking', 'keywords', 'backlinks', 'technical seo',
    'canonical', 'schema', 'sitemap', 'robots.txt', 'crawl', 'index', 'search console',
    'analytics', 'traffic', 'organic', 'serp'
]

# Casual/intro keywords that should be allowed
CAUSAL_ALLOWED = [
    'hello', 'hi', 'hey', 'who are you', 'what are you', 'what can you do', 
    'how can you help', 'help me', 'assist', 'support', 'thanks', 'thank you',
    'bye', 'goodbye', 'good morning', 'good afternoon', 'good evening',
    'yes', 'no', 'okay', 'ok', 'sure', 'please', 'sorry'
]

# Clearly unrelated topics that should be filtered
UNRELATED_TOPICS = [
    'recipe', 'cooking', 'food', 'biryani', 'pizza', 'travel', 'vacation',
    'movie', 'song', 'music', 'game', 'sports', 'cricket', 'football',
    'weather', 'politics', 'news', 'stock', 'crypto', 'bitcoin',
    'medical', 'doctor', 'medicine', 'disease', 'health' 
]

# Greeting keywords
GREETING_KEYWORDS = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]

# Set up templates - with error handling
try:
    templates = Jinja2Templates(directory="templates")
    logger.info("Templates initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize templates: {str(e)}")


    # Create a simple fallback template renderer
    class SimpleTemplates:
        def __init__(self, directory):
            self.directory = directory

        def TemplateResponse(self, name, context):
            # Simple fallback - just return a basic HTML response
            return HTMLResponse(
                "<html><body><h1>Novarsis Support Center</h1><p>Template rendering failed. Please check server logs.</p></body></html>")


    templates = SimpleTemplates("templates")

# FAST MCP - Fast Adaptive Semantic Transfer with Memory Context Protocol
class FastMCP:
    def __init__(self):
        self.conversation_memory = []  # Full conversation memory
        self.context_window = []  # Recent context (last 10 messages)
        self.user_intent = None  # Current user intent
        self.topic_stack = []  # Stack of conversation topics
        self.entities = {}  # Named entities extracted
        self.user_profile = {
            "name": None,
            "plan": None,
            "issues_faced": [],
            "preferred_style": "concise",
            "interaction_count": 0
        }
        self.conversation_state = {
            "expecting_response": None,  # What type of response we're expecting
            "last_question": None,  # Last question asked by bot
            "pending_action": None,  # Any pending action
            "emotional_tone": "neutral"  # User's emotional state
        }
    
    def update_context(self, role, message):
        """Update conversation context with new message"""
        entry = {
            "role": role,
            "content": message,
            "timestamp": datetime.now(),
            "intent": self.extract_intent(message) if role == "user" else None
        }
        
        self.conversation_memory.append(entry)
        self.context_window.append(entry)
        
        # Keep context window to last 10 messages
        if len(self.context_window) > 10:
            self.context_window.pop(0)
        
        if role == "user":
            self.analyze_user_message(message)
        else:
            self.analyze_bot_response(message)
    
    def extract_intent(self, message):
        """Extract user intent from message"""
        message_lower = message.lower()
        
        # Intent patterns
        if any(word in message_lower for word in ['how', 'what', 'where', 'when', 'why']):
            return "question"
        elif any(word in message_lower for word in ['yes', 'yeah', 'sure', 'okay', 'ok', 'yep', 'yup']):
            return "confirmation"
        elif any(word in message_lower for word in ['no', 'nope', 'nah', 'not']):
            return "denial"
        elif any(word in message_lower for word in ['help', 'assist', 'support']):
            return "help_request"
        elif any(word in message_lower for word in ['error', 'issue', 'problem', 'broken', 'not working']):
            return "problem_report"
        elif any(word in message_lower for word in ['thanks', 'thank you', 'appreciate']):
            return "gratitude"
        elif any(word in message_lower for word in ['more', 'elaborate', 'explain', 'detail']):
            return "elaboration_request"
        else:
            return "statement"
    
    def analyze_user_message(self, message):
        """Analyze user message for context and emotion"""
        message_lower = message.lower()
        
        # Update emotional tone
        if any(word in message_lower for word in ['urgent', 'asap', 'immediately', 'quickly']):
            self.conversation_state["emotional_tone"] = "urgent"
        elif any(word in message_lower for word in ['frustrated', 'annoyed', 'angry', 'upset']):
            self.conversation_state["emotional_tone"] = "frustrated"
        elif any(word in message_lower for word in ['please', 'thanks', 'appreciate']):
            self.conversation_state["emotional_tone"] = "polite"
        
        # Extract entities
        if 'website' in message_lower or 'site' in message_lower:
            self.entities['subject'] = 'website'
        if 'seo' in message_lower:
            self.entities['subject'] = 'seo'
        if 'report' in message_lower:
            self.entities['subject'] = 'report'
        
        self.user_profile["interaction_count"] += 1
    
    def analyze_bot_response(self, message):
        """Track what the bot asked or offered"""
        message_lower = message.lower()
        
        if '?' in message:
            self.conversation_state["last_question"] = message
            self.conversation_state["expecting_response"] = "answer"
        
        if 'need more help' in message_lower or 'need help' in message_lower:
            self.conversation_state["expecting_response"] = "help_confirmation"
        
        if 'try these steps' in message_lower or 'follow these' in message_lower:
            self.conversation_state["expecting_response"] = "feedback_on_solution"
    
    def get_context_prompt(self):
        """Generate context-aware prompt for AI"""
        context_parts = []
        
        # Add conversation history
        if self.context_window:
            context_parts.append("=== Conversation Context ===")
            for entry in self.context_window[-5:]:  # Last 5 messages
                role = "User" if entry["role"] == "user" else "Assistant"
                context_parts.append(f"{role}: {entry['content']}")
        
        # Add conversation state
        if self.conversation_state["expecting_response"]:
            context_parts.append(f"\n[Expecting: {self.conversation_state['expecting_response']}]")
        
        if self.conversation_state["emotional_tone"] != "neutral":
            context_parts.append(f"[User tone: {self.conversation_state['emotional_tone']}]")
        
        if self.entities:
            context_parts.append(f"[Current topic: {', '.join(self.entities.values())}]")
        
        return "\n".join(context_parts)
    
    def should_filter_novarsis(self, message):
        """Determine if Novarsis filter should be applied"""
        # Don't filter if we're expecting a response to our question
        if self.conversation_state["expecting_response"] in ["help_confirmation", "answer", "feedback_on_solution"]:
            return False
        
        # Don't filter for contextual responses
        intent = self.extract_intent(message)
        if intent in ["confirmation", "denial", "elaboration_request"]:
            return False
        
        return True

# Initialize FAST MCP
fast_mcp = FastMCP()

# Global session state (in a real app, you'd use Redis or a database)
session_state = {
    "chat_history": [],
    "unresolved_queries": [],
    "support_tickets": {},
    "current_plan": None,
    "current_query": {},
    "typing": False,
    "user_name": "User",
    "session_start": datetime.now(),
    "resolved_count": 0,
    "pending_input": None,
    "uploaded_file": None,
    "checking_ticket_status": False,
    "intro_given": False,
    "last_user_query": "",
    "fast_mcp": fast_mcp  # Add FAST MCP to session
}

# Initialize current plan
plans = [
    {"name": "STARTER", "price": "$100/Year", "validity": "Valid till: Dec 31, 2025",
     "features": ["5 Websites", "Monthly Reports", "Email Support"]},
    {"name": "PREMIUM", "price": "$150/Year", "validity": "Valid till: Dec 31, 2025",
     "features": ["Unlimited Websites", "Real-time Reports", "Priority Support", "API Access"]}
]
session_state["current_plan"] = random.choice(plans)


# Pydantic models for API
class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime
    show_feedback: bool = True  # Changed default to True


class ChatRequest(BaseModel):
    message: str
    image_data: Optional[str] = None


class TicketStatusRequest(BaseModel):
    ticket_id: str


class FeedbackRequest(BaseModel):
    feedback: str
    message_index: int


# Helper Functions
def generate_avatar_initial(name):
    return name[0].upper()


def format_time(timestamp):
    return timestamp.strftime("%I:%M %p")


def cosine_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def is_greeting(query: str) -> bool:
    query_lower = query.lower().strip()
    return any(greeting in query_lower for greeting in GREETING_KEYWORDS)

def is_casual_allowed(query: str) -> bool:
    """Check if it's a casual/intro question that should be allowed"""
    query_lower = query.lower().strip()
    return any(word in query_lower for word in CAUSAL_ALLOWED)

def is_clearly_unrelated(query: str) -> bool:
    """Check if query is clearly unrelated to our tool"""
    query_lower = query.lower().strip()
    return any(topic in query_lower for topic in UNRELATED_TOPICS)

def is_novarsis_related(query: str) -> bool:
    # First check if it's a casual/intro question - always allow these
    if is_casual_allowed(query):
        return True
    
    # Check if it's clearly unrelated - always filter these
    if is_clearly_unrelated(query):
        return False
    
    # Only use semantic filtering if embedding model was successfully initialized
    if reference_embedding is not None and embedding_model is not None:
        try:
            query_embedding = genai.embed_content(
                model=embedding_model,
                content=query,
                task_type="retrieval_query",
            )['embedding']
            similarity = cosine_similarity(reference_embedding, query_embedding)
            if similarity >= 0.7:
                return True
        except Exception as e:
            logger.error(f"Error in semantic filtering: {str(e)}")

    # Fall back to keyword-based filtering
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in NOVARSIS_KEYWORDS)


def get_intro_response() -> str:
    return """Hello! I'm Nova, the official AI support assistant for Novarsis AIO SEO Tool. 

I'm here to help you with any questions or issues you might have regarding our SEO tool.

How can I assist you today? Feel free to ask any questions about Novarsis!"""


def get_ai_response(user_input: str, image_data: Optional[str] = None, chat_history: list = None) -> str:
    if not model:
        return "I apologize, but I'm having trouble connecting to my AI service. Please try again in a moment, or click 'Connect to Human' for immediate assistance."

    try:
        # Get FAST MCP instance
        mcp = session_state.get("fast_mcp", FastMCP())
        
        # Update MCP with user input
        mcp.update_context("user", user_input)
        
        # Check if we should apply Novarsis filter
        should_filter = mcp.should_filter_novarsis(user_input)
        
        # Only filter if MCP says we should
        if should_filter and not is_novarsis_related(user_input):
            return """Sorry, I only help with Novarsis SEO Tool.
            
Please let me know if you have any SEO tool related questions?"""
        
        # Get context from MCP
        context = mcp.get_context_prompt()
        
        # Enhanced system prompt based on emotional tone
        enhanced_prompt = SYSTEM_PROMPT
        if mcp.conversation_state["emotional_tone"] == "urgent":
            enhanced_prompt += "\n[User is urgent - provide immediate, actionable solutions]"
        elif mcp.conversation_state["emotional_tone"] == "frustrated":
            enhanced_prompt += "\n[User is frustrated - be extra helpful and empathetic]"

        if image_data:
            prompt = f"{enhanced_prompt}\n\n{context}\n\nUser query with screenshot: {user_input}"
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            response = model.generate_content([prompt, image])
        else:
            prompt = f"{enhanced_prompt}\n\n{context}\n\nUser query: {user_input}"
            response = model.generate_content(prompt)

        # Remove ** symbols from the response
        response_text = response.text.replace("**", "")
        # Remove any repetitive intro lines if present
        response_text = re.sub(r'^(Hey there[!,. ]*I\'?m Nova.*?assistant[.!]?\s*)', '', response_text, flags=re.IGNORECASE).strip()
        # Keep alphanumeric, spaces, common punctuation, newlines, and bullet/section characters
        response_text = re.sub(r'[^a-zA-Z0-9 .,!?:;()\n•-]', '', response_text)

        # --- Formatting improvements for presentability ---
        # Normalize multiple spaces
        response_text = re.sub(r'\s+', ' ', response_text)
        # Ensure proper paragraph separation
        response_text = re.sub(r'([.!?])\s', r'\1\n\n', response_text)
        # Convert dashes to bullets if they appear at the start of a line
        response_text = re.sub(r'^\s*-\s+', '• ', response_text, flags=re.MULTILINE)
        # --- End formatting improvements ---

        # Format numbered lists: number stays with title, add a blank line after each block
        response_text = re.sub(r'\n?(\d+\.)\s*', r'\n\n\1 ', response_text)  # Ensure number+title on same line
        # Add spacing after list item sentences for readability
        response_text = re.sub(r'(\n\d+\. [^\n]+)(?=\n\d+\.)', r'\1\n', response_text)
        response_text = re.sub(r'(•)', r'\n\1', response_text)       # Bullets
        response_text = re.sub(r'(Step\s+\d+)', r'\n\1', response_text)  # Steps
        response_text = re.sub(r'(Tip:)', r'\n\1', response_text)        # Tips
        response_text = re.sub(r'(Solution:)', r'\n\1', response_text)   # Solutions
        response_text = re.sub(r'(Alternative:)', r'\n\1', response_text) # Alternatives

        # --- Final cleanup for unnecessary spaces and gaps ---
        # Remove spaces before punctuation
        response_text = re.sub(r'\s+([.,!?;:])', r'\1', response_text)
        # Remove extra spaces at line beginnings
        response_text = re.sub(r'^\s+', '', response_text, flags=re.MULTILINE)
        # Collapse multiple blank lines into max 2
        response_text = re.sub(r'\n{3,}', '\n\n', response_text)

        return response_text.strip()
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        return "I encountered an issue processing your request. Please try rephrasing your question or connect with our human support team for assistance."


def save_unresolved_query(query_data: Dict) -> str:
    query_data['timestamp'] = datetime.now()
    query_data['ticket_id'] = f"NVS{random.randint(10000, 99999)}"
    query_data['status'] = "In Progress"
    query_data['priority'] = "High" if "urgent" in query_data['query'].lower() else "Normal"
    session_state["unresolved_queries"].append(query_data)
    session_state["support_tickets"][query_data['ticket_id']] = query_data
    return query_data['ticket_id']


# API Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Add user message to chat history
    user_message = {
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now()
    }
    session_state["chat_history"].append(user_message)

    # Store current query for potential escalation
    session_state["current_query"] = {
        "query": request.message,
        "timestamp": datetime.now()
    }

    # Store last user query for "Connect with an Expert"
    session_state["last_user_query"] = request.message

    # Get AI response with chat history for context
    time.sleep(0.5)  # Simulate thinking time

    if session_state["checking_ticket_status"]:
        ticket_id = request.message.strip().upper()

        if ticket_id.startswith("NVS") and len(ticket_id) > 3:
            if ticket_id in session_state["support_tickets"]:
                ticket = session_state["support_tickets"][ticket_id]
                response = f"""🎫 Ticket Details:

Ticket ID: {ticket_id}
Status: {ticket['status']}
Priority: {ticket['priority']}
Created: {ticket['timestamp'].strftime('%Y-%m-%d %H:%M')}
Query: {ticket['query']}

Our team is working on your issue. You'll receive a notification when there's an update."""
            else:
                response = f"❌ Ticket ID '{ticket_id}' not found. Please check the ticket number and try again, or contact support at {SUPPORT_EMAIL}."
        else:
            response = "⚠️ Please enter a valid ticket ID (e.g., NVS12345)."

        session_state["checking_ticket_status"] = False
        show_feedback = True  # Changed to True
    elif is_greeting(request.message):
        response = get_intro_response()
        session_state["intro_given"] = True
        show_feedback = True  # Changed to True
    else:
        response = get_ai_response(request.message, request.image_data, session_state["chat_history"])
        show_feedback = True  # Already True

    # Update FAST MCP with bot response
    if "fast_mcp" in session_state:
        session_state["fast_mcp"].update_context("assistant", response)
    
    # Add bot response to chat history
    bot_message = {
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now(),
        "show_feedback": show_feedback
    }
    session_state["chat_history"].append(bot_message)

    # Don't send suggestions with response anymore since we're doing real-time
    return {"response": response, "show_feedback": show_feedback}


@app.post("/api/check-ticket-status")
async def check_ticket_status():
    session_state["checking_ticket_status"] = True
    response = "Please enter your ticket number (e.g., NVS12345):"

    bot_message = {
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now(),
        "show_feedback": True  # Changed to True
    }
    session_state["chat_history"].append(bot_message)

    return {"response": response}


@app.post("/api/connect-expert")
async def connect_expert():
    if session_state["last_user_query"]:
        ticket_id = save_unresolved_query({
            "query": session_state["last_user_query"],
            "timestamp": datetime.now()
        })
        response = f"""I've created a priority support ticket for you:

🎫 Ticket ID: {ticket_id}
📱 Status: Escalated to Human Support
⏱️ Response Time: Within 15 minutes

Our expert team has been notified and will reach out to you shortly via:
• In-app chat
• Email to your registered address
• WhatsApp: {WHATSAPP_NUMBER}

You can check your ticket status anytime by typing 'ticket {ticket_id}'"""
    else:
        response = "I'd be happy to connect you with an expert. Please first send your query so I can create a support ticket for you."

    bot_message = {
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now(),
        "show_feedback": True  # Changed to True
    }
    session_state["chat_history"].append(bot_message)

    return {"response": response}


@app.post("/api/feedback")
async def feedback(request: FeedbackRequest):
    if request.feedback == "no":
        ticket_id = save_unresolved_query(session_state["current_query"])
        response = f"""I understand this didn't fully resolve your issue. I've created a priority support ticket for you:

🎫 Ticket ID: {ticket_id}
📱 Status: Escalated to Human Support
⏱️ Response Time: Within 15 minutes

Our expert team has been notified and will reach out to you shortly via:
• In-app chat
• Email to your registered address
• WhatsApp: {WHATSAPP_NUMBER}

You can check your ticket status anytime by typing 'ticket {ticket_id}'"""
        session_state["resolved_count"] -= 1
    else:
        response = "Great! I'm glad I could help. Feel free to ask if you have any more questions about Novarsis! 🚀"
        session_state["resolved_count"] += 1

    bot_message = {
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now(),
        "show_feedback": True  # Changed to True
    }
    session_state["chat_history"].append(bot_message)

    return {"response": response}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, and PNG files are allowed")

    # Read file and convert to base64
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode('utf-8')

    return {"image_data": base64_image, "filename": file.filename}


@app.get("/api/chat-history")
async def get_chat_history():
    return {"chat_history": session_state["chat_history"]}


@app.get("/api/suggestions")
async def get_suggestions():
    """Get initial suggestions when the chat loads."""
    return {"suggestions": QUICK_REPLY_SUGGESTIONS["initial"]}


@app.post("/api/typing-suggestions")
async def get_typing_suggestions(request: dict):
    """Get real-time suggestions based on what user is typing."""
    user_input = request.get("input", "")
    suggestions = get_context_suggestions(user_input)
    return {"suggestions": suggestions}


# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Create index.html template
with open("templates/index.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Novarsis Support Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            font-family: 'Inter', sans-serif !important;
        }

        body {
            background: #f0f2f5;
            margin: 0;
            padding: 0;
        }

        .main-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }

        .header-container {
            background: white;
            border-radius: 16px;
            padding: 16px 24px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-right: 10px;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            padding: 6px 12px;
            background: #e8f5e9;
            border-radius: 20px;
            font-size: 13px;
            color: #2e7d32;
            font-weight: 500;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: #4caf50;
            border-radius: 50%;
            margin-right: 6px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .chat-container {
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            height: 70vh;
            min-height: 500px;
            overflow-y: auto;
            padding: 20px;
            margin-bottom: 20px;
            position: relative;
        }

        .message-wrapper {
            display: flex;
            margin-bottom: 20px;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .user-message-wrapper {
            justify-content: flex-end;
        }

        .bot-message-wrapper {
            justify-content: flex-start;
        }

        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            font-size: 15px;
            line-height: 1.5;
            position: relative;
            word-wrap: break-word;
        }

        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 5px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }

        .bot-message {
            background: #f1f3f5;
            color: #2d3436;
            border-bottom-left-radius: 5px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }

        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            margin: 0 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 14px;
            flex-shrink: 0;
        }

        .user-avatar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .bot-avatar {
            background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
            color: white;
        }

        .timestamp {
            font-size: 11px;
            color: rgba(0,0,0,0.5);
            margin-top: 6px;
            font-weight: 400;
        }

        .user-timestamp {
            color: rgba(255,255,255,0.8);
            text-align: right;
        }

        .typing-indicator {
            display: flex;
            align-items: center;
            padding: 15px;
            background: #f1f3f5;
            border-radius: 18px;
            width: fit-content;
            margin-left: 45px;
            margin-bottom: 20px;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            background: #95a5a6;
            border-radius: 50%;
            margin: 0 3px;
            animation: typing 1.4s infinite;
        }

        .typing-dot:nth-child(1) { animation-delay: 0s; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }

        .input-container {
            background: white;
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            position: sticky;
            bottom: 20px;
        }

        .suggestions-container {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
            flex-wrap: wrap;
            max-height: 80px;
            overflow-y: auto;
            padding: 4px 0;
            transition: opacity 0.15s ease;
            min-height: 32px;
        }

        .suggestion-pill {
            padding: 8px 14px;
            background: #f0f2f5;
            border: 1px solid #e1e4e8;
            border-radius: 20px;
            font-size: 13px;
            color: #24292e;
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
            flex-shrink: 0;
            font-weight: 500;
            animation: slideInFade 0.3s ease-out forwards;
            opacity: 0;
        }

        @keyframes slideInFade {
            from {
                opacity: 0;
                transform: translateY(-5px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .suggestion-pill:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
            transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(102, 126, 234, 0.2);
        }

        .suggestion-pill:active {
            transform: translateY(0);
        }

        .suggestions-container::-webkit-scrollbar {
            height: 4px;
        }

        .suggestions-container::-webkit-scrollbar-track {
            background: transparent;
        }

        .suggestions-container::-webkit-scrollbar-thumb {
            background: #d0d0d0;
            border-radius: 2px;
        }

        .message-form {
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .message-input {
            flex: 1;
            border-radius: 24px;
            border: 1px solid #e0e0e0;
            padding: 12px 20px;
            font-size: 15px;
            background: #f8f9fa;
            color: #333333;
            outline: none;
        }

        .message-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        }

        .send-btn {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .send-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }

        .attachment-btn {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
            color: #54656f;
            padding: 0;
        }

        .attachment-btn:hover {
            background-color: #f1f3f5;
            border-color: #667eea;
            transform: scale(1.05);
        }

        .attachment-btn.success {
            background-color: #e8f5e9;
            color: #4caf50;
            border-color: #4caf50;
            pointer-events: none;
        }

        .attachment-btn.success svg path {
            fill: #4caf50;
        }

        .feedback-container {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            margin-left: 45px;
        }

        .feedback-btn {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            border: 1px solid #e0e0e0;
            background: white;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .feedback-btn:hover {
            background: #f8f9fa;
            border-color: #667eea;
        }

        .file-input {
            display: none;
        }

        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 6px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            .main-container {
                padding: 10px;
            }

            .chat-container {
                height: 65vh;
                border-radius: 12px;
                padding: 15px;
            }

            .message-content {
                max-width: 80%;
                font-size: 14px;
            }

            .input-container {
                padding: 12px;
                border-radius: 12px;
            }

            .header-container {
                padding: 12px 16px;
                border-radius: 12px;
            }


        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="header-container">
            <div class="logo-section">
                <span class="logo">🚀 NOVARSIS</span>
                <span style="color: #95a5a6; font-size: 14px;">AI Support Center</span>
            </div>
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span>Nova is Online</span>
            </div>
        </div>

        <div class="chat-container" id="chat-container">
            <div class="message-wrapper bot-message-wrapper">
                <div class="avatar bot-avatar">N</div>
                <div class="message-content bot-message">
                    👋 Hi! I'm Nova, your Novarsis AI assistant.<br><br>
                    How can I assist you today?
                    <div class="timestamp">
                        <span id="welcome-timestamp"></span>
                    </div>
                </div>
            </div>
        </div>

        <div class="input-container">
            <div class="suggestions-container" id="suggestions-container">
                <!-- Quick reply suggestions will be dynamically added here -->
            </div>

            <form class="message-form" id="message-form">
                <input type="file" id="file-input" class="file-input" accept="image/jpeg,image/jpg,image/png">
                <button type="button" class="attachment-btn" id="attachment-btn">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z" fill="currentColor"/>
                    </svg>
                </button>
                <input type="text" class="message-input" id="message-input" placeholder="Type your message...">
                <button type="submit" class="send-btn">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="white"/>
                    </svg>
                </button>
            </form>
        </div>
    </div>

    <script>
        // Format time function
        function formatTime(timestamp) {
            const date = new Date(timestamp);
            return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
        }

        // Current time for welcome message
        document.addEventListener('DOMContentLoaded', function() {
            const now = new Date();
            document.getElementById('welcome-timestamp').textContent = formatTime(now);
            // Load initial suggestions
            loadInitialSuggestions();
        });

        // Chat container
        const chatContainer = document.getElementById('chat-container');

        // Message input
        const messageForm = document.getElementById('message-form');
        const messageInput = document.getElementById('message-input');
        const attachmentBtn = document.getElementById('attachment-btn');
        const fileInput = document.getElementById('file-input');

        // File handling
        let uploadedImageData = null;
        let uploadedFileName = '';

        attachmentBtn.addEventListener('click', function() {
            fileInput.click();
        });

        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    uploadedImageData = event.target.result.split(',')[1]; // Get base64 data
                    uploadedFileName = file.name;
                    attachmentBtn.classList.add('success');
                    // Change icon to checkmark
                    attachmentBtn.innerHTML = `
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" fill="currentColor"/>
                        </svg>
                    `;
                };
                reader.readAsDataURL(file);
            }
        });

        // Add message to chat
        function addMessage(role, content, showFeedback = true) {  // Changed default to true
            const messageWrapper = document.createElement('div');
            messageWrapper.className = `message-wrapper ${role}-message-wrapper`;

            const avatar = document.createElement('div');
            avatar.className = `avatar ${role}-avatar`;
            avatar.textContent = role === 'user' ? '@' : 'N';

            const messageContent = document.createElement('div');
            messageContent.className = `message-content ${role}-message`;
            // Format content: keep paragraphs and line breaks
            const formattedContent = '<p>' + content
                .replace(/\\n\\n/g, '</p><p>')
                .replace(/\\n/g, '<br>') + '</p>';
            messageContent.innerHTML = formattedContent;

            const timestamp = document.createElement('div');
            timestamp.className = `timestamp ${role}-timestamp`;
            timestamp.textContent = formatTime(new Date());

            messageContent.appendChild(timestamp);

            if (role === 'user') {
                messageWrapper.appendChild(messageContent);
                messageWrapper.appendChild(avatar);
            } else {
                messageWrapper.appendChild(avatar);
                messageWrapper.appendChild(messageContent);
                // Feedback buttons removed: assistant messages now only show avatar and content.
            }

            chatContainer.appendChild(messageWrapper);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        // Show typing indicator
        function showTypingIndicator() {
            const typingIndicator = document.createElement('div');
            typingIndicator.className = 'typing-indicator';
            typingIndicator.innerHTML = `
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            `;
            chatContainer.appendChild(typingIndicator);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return typingIndicator;
        }

        // Update suggestions with smooth animation
        function updateSuggestions(suggestions) {
            const container = document.getElementById('suggestions-container');

            // Smooth transition
            container.style.opacity = '0';

            setTimeout(() => {
                container.innerHTML = '';

                if (suggestions && suggestions.length > 0) {
                    suggestions.forEach((suggestion, index) => {
                        const pill = document.createElement('div');
                        pill.className = 'suggestion-pill';
                        pill.textContent = suggestion;
                        pill.style.animationDelay = `${index * 50}ms`;
                        pill.onclick = () => {
                            messageInput.value = suggestion;
                            messageForm.dispatchEvent(new Event('submit'));
                        };
                        container.appendChild(pill);
                    });
                }

                container.style.opacity = '1';
            }, 150);
        }

        // Load initial suggestions
        async function loadInitialSuggestions() {
            try {
                const response = await fetch('/api/suggestions');
                const data = await response.json();
                updateSuggestions(data.suggestions);
            } catch (error) {
                console.error('Error loading suggestions:', error);
            }
        }

        // Real-time typing suggestions with debouncing
        let typingTimer;
        const doneTypingInterval = 300; // ms

        async function fetchTypingSuggestions(input) {
            if (input.trim().length < 2) {
                // Show initial suggestions if input is empty or very short
                if (input.trim().length === 0) {
                    loadInitialSuggestions();
                } else {
                    updateSuggestions([]);
                }
                return;
            }

            try {
                const response = await fetch('/api/typing-suggestions', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ input: input })
                });

                const data = await response.json();
                updateSuggestions(data.suggestions);
            } catch (error) {
                console.error('Error fetching suggestions:', error);
            }
        }

        // Handle input changes for real-time suggestions
        messageInput.addEventListener('input', function(e) {
            clearTimeout(typingTimer);
            const inputValue = e.target.value;

            // Debounce the API call
            typingTimer = setTimeout(() => {
                fetchTypingSuggestions(inputValue);
            }, doneTypingInterval);
        });

        // Handle focus to show suggestions
        messageInput.addEventListener('focus', function(e) {
            if (e.target.value.trim().length === 0) {
                loadInitialSuggestions();
            } else {
                fetchTypingSuggestions(e.target.value);
            }
        });

        // Send message
        async function sendMessage(message, imageData = null) {
            // Handle special commands
            if (message.toLowerCase() === 'check ticket status') {
                // Clear suggestions
                updateSuggestions([]);

                // Call the check ticket status API
                try {
                    const response = await fetch('/api/check-ticket-status', {
                        method: 'POST'
                    });
                    const data = await response.json();
                    addMessage('assistant', data.response, true);  // Added true
                } catch (error) {
                    console.error('Error checking ticket status:', error);
                    addMessage('assistant', 'Sorry, I encountered an error checking ticket status.', true);  // Added true
                }

                // Load initial suggestions after a delay
                setTimeout(() => {
                    loadInitialSuggestions();
                }, 500);
                return;
            }

            if (message.toLowerCase() === 'connect with an expert') {
                // Clear suggestions
                updateSuggestions([]);

                // Call the connect expert API
                try {
                    const response = await fetch('/api/connect-expert', {
                        method: 'POST'
                    });
                    const data = await response.json();
                    addMessage('assistant', data.response, true);  // Added true
                } catch (error) {
                    console.error('Error connecting with expert:', error);
                    addMessage('assistant', 'Sorry, I encountered an error connecting you with an expert.', true);  // Added true
                }

                // Load initial suggestions after a delay
                setTimeout(() => {
                    loadInitialSuggestions();
                }, 500);
                return;
            }

            // Normal message handling
            // Add user message
            addMessage('user', message);

            // Clear suggestions after sending
            updateSuggestions([]);

            // Show typing indicator
            const typingIndicator = showTypingIndicator();

            try {
                // Send to API
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        image_data: imageData
                    })
                });

                const data = await response.json();

                // Remove typing indicator
                typingIndicator.remove();

                // Add bot response
                addMessage('assistant', data.response, data.show_feedback);

                // Load initial suggestions after response
                setTimeout(() => {
                    loadInitialSuggestions();
                }, 500);

                // Reset attachment
                if (uploadedImageData) {
                    attachmentBtn.classList.remove('success');
                    attachmentBtn.innerHTML = `
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z" fill="currentColor"/>
                        </svg>
                    `;
                    uploadedImageData = null;
                    uploadedFileName = '';
                    fileInput.value = '';
                }

            } catch (error) {
                console.error('Error sending message:', error);
                typingIndicator.remove();
                addMessage('assistant', 'Sorry, I encountered an error. Please try again.', true);  // Added true
                // Show initial suggestions even on error
                setTimeout(() => {
                    loadInitialSuggestions();
                }, 500);
            }
        }

        // Send feedback
        async function sendFeedback(feedback) {
            const messageIndex = document.querySelectorAll('.message-wrapper').length - 1;

            try {
                const response = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        feedback: feedback,
                        message_index: messageIndex
                    })
                });

                const data = await response.json();
                addMessage('assistant', data.response, true);  // Added true

            } catch (error) {
                console.error('Error sending feedback:', error);
            }
        }

        // Handle form submission
        messageForm.addEventListener('submit', async function(e) {
            e.preventDefault();

            const message = messageInput.value.trim();
            if (message) {
                await sendMessage(message, uploadedImageData);
                messageInput.value = '';
            }
        });

        // Handle Enter key in message input
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                messageForm.dispatchEvent(new Event('submit'));
            }
        });
    </script>
</body>
</html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
