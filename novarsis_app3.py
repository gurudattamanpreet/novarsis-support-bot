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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBMCGZFLUH3E_atf1mdc2clXw5lnE0ypyo")
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
WHATSAPP_NUMBER = "+91-8962810180"
SUPPORT_EMAIL = "support@novarsis.tech"

# Enhanced System Prompt
SYSTEM_PROMPT = """You are Nova, the official AI support assistant for Novarsis AIO SEO Tool.

PERSONALITY:
- Professional yet friendly
- Proactive in offering solutions
- Empathetic to user frustrations
- Clear and concise in explanations
- Conversational and natural tone
- Uses simple, everyday language
- Adds small friendly phrases like "Got it", "No worries", "Let's fix this together"
- Can use light humor where appropriate, while staying professional

SCOPE:
You ONLY help with Novarsis-related queries:
 SEO analysis issues and errors
 Account and subscription management
Technical troubleshooting
Feature explanations and tutorials
Billing and payment issues
Report generation problems
API and integration support
Performance optimization tips

For non-Novarsis queries, politely redirect:
"I specialize in Novarsis SEO Tool support. For this query, I'd recommend [appropriate resource]. 
Is there anything about Novarsis I can help you with instead?"

RESPONSE STYLE:
1. Acknowledge the issue in a natural way (like "Got it, I see what's happening here")
2. Provide step-by-step solutions in simple language
3. Offer alternatives if needed, explained in plain words
4. Confirm understanding or ask if further help is required ("Did that help?")

Use emojis sparingly for friendliness: ✅ ❌ 💡 🔧 📊 🚀
Also use friendly fillers occasionally like 🙂 👍 🙌
Format responses with clear sections and bullet points."""

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


# Novarsis Keywords
NOVARSIS_KEYWORDS = [
    'novarsis', 'seo', 'website analysis', 'meta tags', 'page structure', 'link analysis',
    'seo check', 'seo report', 'subscription', 'account', 'billing', 'plan', 'premium',
    'starter', 'error', 'bug', 'issue', 'problem', 'not working', 'failed', 'crash',
    'login', 'password', 'analysis', 'report', 'dashboard', 'settings', 'integration'
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
    "last_user_query": ""
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


def is_novarsis_related(query: str) -> bool:
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


def get_ai_response(user_input: str, image_data: Optional[str] = None) -> str:
    if not model:
        return "I apologize, but I'm having trouble connecting to my AI service. Please try again in a moment, or click 'Connect to Human' for immediate assistance."

    try:
        if not is_novarsis_related(user_input):
            return """I understand you need help, but I specialize specifically in Novarsis SEO Tool support. 

For your query, you might want to try:
• General tech support forums
• Product-specific documentation
• Or Google search for more information

Is there anything about Novarsis SEO Tool I can help you with? Such as:
• SEO analysis issues
• Account management
• Report generation
• Technical problems"""

        if image_data:
            prompt = f"{SYSTEM_PROMPT}\n\nUser query with screenshot: {user_input}"
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            response = model.generate_content([prompt, image])
        else:
            prompt = f"{SYSTEM_PROMPT}\n\nUser query: {user_input}"
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

    # Get AI response
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
        response = get_ai_response(request.message, request.image_data)
        show_feedback = True  # Already True

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
            color: #667eea;
            font-size: 18px;
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
                <div class="attachment-btn" id="attachment-btn">📎</div>
                <input type="text" class="message-input" id="message-input" placeholder="Type your message...">
                <button type="submit" class="send-btn">➤</button>
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
                    attachmentBtn.innerHTML = '✓';
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
            avatar.textContent = role === 'user' ? 'U' : 'N';

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
                    attachmentBtn.innerHTML = '📎';
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
