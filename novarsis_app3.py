from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
import time
import random
import google.generativeai as genai
from datetime import datetime
import base64
import io
from PIL import Image
import asyncio
import uvicorn
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(title="Novarsis Support Bot API")

# Add CORS middleware - more secure for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API Configuration - better to use environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAry6BNYcjHE2tGUWSjZlq5RohT8F6I7bs")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini Flash 2.0 model
try:
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    logger.info("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini model: {str(e)}")
    model = None

# In-memory storage (for demo - use Redis or database in production)
sessions = {}

# Novarsis-specific keywords
NOVARSIS_KEYWORDS = [
    'novarsis', 'seo', 'website analysis', 'meta tags', 'page structure', 'link analysis',
    'seo check', 'seo report', 'subscription', 'account', 'billing', 'plan', 'premium',
    'starter', 'error', 'bug', 'issue', 'problem', 'not working', 'failed', 'crash',
    'login', 'password', 'analysis', 'report', 'dashboard', 'settings', 'integration',
    'google', 'api', 'website', 'url', 'scan', 'audit', 'optimization', 'mobile',
    'speed', 'performance', 'competitor', 'ranking', 'keywords', 'backlinks',
    'technical seo', 'canonical', 'schema', 'sitemap', 'robots.txt', 'crawl',
    'index', 'search console', 'analytics', 'traffic', 'organic', 'serp'
]

# System prompt
SYSTEM_PROMPT = """You are the official support assistant for Novarsis AIO SEO Tool. 
IMPORTANT: You ONLY help with Novarsis tool-related queries such as:
- SEO analysis issues and errors
- Account and subscription problems
- Technical issues with the Novarsis platform
- Feature explanations and usage guidance
- Billing and payment queries
- Report generation problems
- Integration issues
For ANY query not related to Novarsis tool, respond with:
"❌ This question is not relevant to Novarsis Support. I can only assist with Novarsis tool-related issues, account management, and technical problems."
When helping with Novarsis issues:
- Be professional and helpful
- Provide step-by-step solutions
- Reference specific Novarsis features when relevant
- Guide users through the tool's interface
- Explain SEO concepts as they relate to Novarsis reports
Available Novarsis Features:
- Website SEO Analysis (Basic & Advanced)
- SEO Reports (Meta, Structure, Links, Technical, Performance)
- Mobile Optimization Checker
- Page Speed Analysis
- Competitor Analysis
- Account Management
- Subscription Plans (FREE, STARTER $100/year, PREMIUM $150/year)"""


# Pydantic models
class ChatMessage(BaseModel):
    message: str
    session_id: str
    image_data: Optional[str] = None


class QuickAction(BaseModel):
    action: str
    session_id: str


class TicketRequest(BaseModel):
    ticket_number: str
    session_id: str


class SessionData(BaseModel):
    chat_history: List[Dict] = []
    unresolved_queries: List[Dict] = []
    support_tickets: Dict[str, Dict] = {}
    current_plan: Dict = {}
    current_query: Dict = {}
    executive_connected: bool = False


# Helper functions
def get_or_create_session(session_id: str) -> SessionData:
    if session_id not in sessions:
        plans = [
            {"name": "STARTER", "price": "$100/Year", "validity": "Valid till: Dec 31, 2025"},
            {"name": "PREMIUM", "price": "$150/Year", "validity": "Valid till: Dec 31, 2025"}
        ]
        sessions[session_id] = SessionData(
            current_plan=random.choice(plans)
        )
    return sessions[session_id]


def is_novarsis_related(query: str) -> bool:
    query_lower = query.lower()
    for keyword in NOVARSIS_KEYWORDS:
        if keyword in query_lower:
            return True
    generic_terms = ['help', 'support', 'assist', 'guide', 'how to', 'tutorial']
    tool_context = ['tool', 'software', 'platform', 'app', 'application', 'system']
    has_generic = any(term in query_lower for term in generic_terms)
    has_context = any(term in query_lower for term in tool_context)
    return has_generic and has_context


async def get_ai_response(user_input: str, image_data: Optional[bytes] = None) -> str:
    if not model:
        return "I apologize, but the AI service is currently unavailable. Please try again later or connect with a human executive."

    try:
        if not is_novarsis_related(user_input):
            return "❌ **This question is not relevant to Novarsis Support.**\n\nI can only assist with:\n• Novarsis tool errors and issues\n• SEO analysis problems\n• Account and subscription queries\n• Report generation issues\n• Technical problems with Novarsis platform\n\nPlease ask a question related to Novarsis AIO SEO Tool."

        if image_data:
            prompt = f"{SYSTEM_PROMPT}\n\nUser has shared a screenshot about a Novarsis issue: {user_input}\n\nAnalyze the image and provide a solution specific to Novarsis tool."
            try:
                image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                response = model.generate_content([prompt, image])
            except Exception as img_error:
                logger.error(f"Error processing image: {str(img_error)}")
                return "I apologize, but I couldn't process the uploaded image. Please describe your Novarsis issue in text, and I'll help you with that."
        else:
            prompt = f"{SYSTEM_PROMPT}\n\nUser query about Novarsis: {user_input}"
            chat = model.start_chat(history=[])
            response = chat.send_message(prompt)

        return response.text
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        return "I apologize, but I'm having trouble processing your Novarsis-related request. Please try again or connect with a human executive for immediate assistance."


def save_unresolved_query(session: SessionData, query_data: Dict) -> str:
    query_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query_data['ticket_id'] = f"NVS{random.randint(10000, 99999)}"
    query_data['status'] = "In Progress"
    session.unresolved_queries.append(query_data)
    session.support_tickets[query_data['ticket_id']] = query_data
    return query_data['ticket_id']


# API Endpoints
@app.get("/")
async def home():
    return HTMLResponse(content=HTML_CONTENT)


@app.post("/api/chat")
async def chat_endpoint(chat_message: ChatMessage):
    session = get_or_create_session(chat_message.session_id)
    # Add user message to history
    session.chat_history.append({
        "role": "user",
        "content": chat_message.message
    })
    # Get AI response
    response = await get_ai_response(
        chat_message.message,
        chat_message.image_data if chat_message.image_data else None
    )
    # Add assistant response to history
    session.chat_history.append({
        "role": "assistant",
        "content": response,
        "show_feedback": is_novarsis_related(chat_message.message)
    })
    # Store current query if relevant
    if is_novarsis_related(chat_message.message):
        session.current_query = {
            "query": chat_message.message,
            "image": chat_message.image_data
        }
    return {
        "response": response,
        "show_feedback": is_novarsis_related(chat_message.message),
        "chat_history": session.chat_history
    }


@app.post("/api/quick-action")
async def quick_action_endpoint(request: QuickAction):
    session = get_or_create_session(request.session_id)
    # Define features and limits outside the f-string
    starter_features = """✓ Basic SEO Checks
✓ 5 Websites tracking
✓ Monthly Reports
✓ Google Integrations
✓ Email support"""
    premium_features = """✓ Advanced SEO Analysis
✓ Unlimited Websites
✓ Weekly Reports
✓ Priority Support
✓ API Access
✓ Competitor Analysis
✓ White-label Reports"""
    starter_api = "5,000/month"
    premium_api = "Unlimited"
    starter_reports = "50/month"
    premium_reports = "Unlimited"
    action_responses = {
        "1": {
            "title": "Report Novarsis Error 🚨",
            "response": """📸 **How to Report a Novarsis Error:**
1. **Take a Screenshot** of the Novarsis error message
2. **Upload the Screenshot** using the file uploader below
3. **Describe what you were doing** in Novarsis when the error occurred
4. Our AI will analyze and provide a Novarsis-specific solution
**Common Novarsis Issues We Solve:**
• SEO analysis not completing
• Report generation errors
• Login/authentication problems
• Integration failures with Google tools
• Dashboard loading issues
⏱️ **Resolution Time:**
- AI Solution: Instant
- If unresolved: Novarsis expert within 15 minutes
Please upload your Novarsis error screenshot now."""
        },
        "2": {
            "title": "Account & Subscription 💳",
            "response": f"""👤 **Your Novarsis Account & Subscription:**
🔍 **Tool:** Novarsis AIO SEO Assistant
📋 **Current Plan:** {session.current_plan['name']}
💰 **Price:** {session.current_plan['price']}
📅 **{session.current_plan['validity']}**
**Your Novarsis Plan Features:**
{starter_features if session.current_plan['name'] == 'STARTER' else premium_features}
**Novarsis Usage Limits:**
• API Calls: {starter_api if session.current_plan['name'] == 'STARTER' else premium_api}
• Reports: {starter_reports if session.current_plan['name'] == 'STARTER' else premium_reports}
Need to upgrade? Visit: novarsis.tech/pricing"""
        },
        "3": {
            "title": "Connect with Expert 👥",
            "response": "🎫 **Connect with Novarsis Expert**\n\nPlease enter your Novarsis Ticket Number (starts with NVS):",
            "need_ticket": True
        },
        "4": {
            "title": "Check Ticket Status 📋",
            "response": "🔍 **Check Novarsis Ticket Status**\n\nPlease enter your Novarsis Ticket Number (format: NVS#####):",
            "check_status": True
        },
        "5": {
            "title": "Contact Novarsis 📞",
            "response": """📞 **Novarsis Support Contact:**
🏢 **Novarsis Technologies**
Your trusted SEO analysis partner
📧 **Email:** support@novarsis.tech
☎️ **Phone:** +1-800-NOVARSIS (668-2774)
💬 **Live Chat:** Available 24x7 on novarsis.tech
🌐 **Website:** www.novarsis.tech/support
**Support Hours:**
• Technical Support: 24x7
• Sales Team: Mon-Fri 9AM-6PM EST
• Billing Support: Mon-Fri 10AM-5PM EST
📍 **Headquarters:**
Novarsis Technologies Inc.
123 SEO Boulevard, Suite 500
San Francisco, CA 94105
✨ **Feel free to reach out to us for any Novarsis-related queries!**"""
        }
    }
    if request.action in action_responses:
        action_data = action_responses[request.action]
        # Add to chat history
        session.chat_history.append({
            "role": "user",
            "content": action_data["title"]
        })
        session.chat_history.append({
            "role": "assistant",
            "content": action_data["response"],
            "show_feedback": False
        })
        return {
            "response": action_data["response"],
            "need_ticket": action_data.get("need_ticket", False),
            "check_status": action_data.get("check_status", False),
            "chat_history": session.chat_history
        }
    raise HTTPException(status_code=400, detail="Invalid action")


@app.post("/api/feedback")
async def feedback_endpoint(request: dict):
    session = get_or_create_session(request["session_id"])
    feedback = request["feedback"]
    if feedback == "no":
        ticket_id = save_unresolved_query(session, session.current_query)
        return {
            "ticket_id": ticket_id,
            "message": f"🎫 Novarsis Ticket ID: **{ticket_id}**\n🔄 Connecting to a Novarsis expert... Your query will be resolved in next 15 minutes."
        }
    else:
        return {
            "message": "Great! Thank you for using Novarsis Support. Have a productive SEO analysis! 🚀"
        }


@app.post("/api/check-ticket")
async def check_ticket_endpoint(request: TicketRequest):
    session = get_or_create_session(request.session_id)
    if request.ticket_number.startswith("NVS"):
        if request.ticket_number in session.support_tickets:
            ticket = session.support_tickets[request.ticket_number]
            response = f"""✅ **Novarsis Ticket Found!**
🎫 **Ticket ID:** {request.ticket_number}
🔍 **Tool:** Novarsis AIO
📅 **Created:** {ticket['timestamp']}
📊 **Status:** In Progress
⏱️ **Resolution Time:** Within 15 minutes
🔄 Connecting to Novarsis expert...
A specialist familiar with your issue is reviewing your case."""
            return {"status": "found", "message": response}
        else:
            return {"status": "not_found", "message": "❌ Novarsis ticket not found. Please check the ticket number."}
    else:
        return {"status": "invalid", "message": "Please enter a valid Novarsis ticket (starts with NVS)"}


@app.get("/api/session/{session_id}/stats")
async def get_session_stats(session_id: str):
    session = get_or_create_session(session_id)
    return {
        "total_queries": len(session.chat_history) // 2,
        "open_tickets": len(session.unresolved_queries),
        "current_plan": session.current_plan,
        "unresolved_queries": session.unresolved_queries
    }


@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Convert to base64 for storage/transmission
        image_base64 = base64.b64encode(contents).decode('utf-8')
        return {"image_data": image_base64, "filename": file.filename}
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/reset-session")
async def reset_session(request: dict):
    session_id = request["session_id"]
    if session_id in sessions:
        # Keep tickets and plan, reset everything else
        tickets = sessions[session_id].support_tickets
        plan = sessions[session_id].current_plan
        unresolved = sessions[session_id].unresolved_queries
        sessions[session_id] = SessionData(
            current_plan=plan,
            support_tickets=tickets,
            unresolved_queries=unresolved
        )
    return {"status": "success", "message": "Session reset successfully"}


# HTML Template (keeping the same as your original)
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Novarsis Support Bot - 24x7 Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header h1 {
            color: #333;
            font-size: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .header p {
            color: #666;
            margin-top: 5px;
        }
        .container {
            flex: 1;
            display: flex;
            max-width: 1400px;
            margin: 20px auto;
            width: 100%;
            padding: 0 20px;
            gap: 20px;
        }
        .sidebar {
            width: 300px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 20px;
            overflow-y: auto;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .main-content {
            flex: 1;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .chat-container {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .message {
            display: flex;
            gap: 10px;
            animation: fadeIn 0.3s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user {
            justify-content: flex-end;
        }
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 10px;
            word-wrap: break-word;
        }
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .message.assistant .message-content {
            background: #f3f4f6;
            color: #333;
        }
        .input-container {
            padding: 20px;
            border-top: 1px solid #e5e5e5;
            background: white;
            border-radius: 0 0 10px 10px;
        }
        .input-wrapper {
            display: flex;
            gap: 10px;
        }
        input[type="text"] {
            flex: 1;
            padding: 12px;
            border: 2px solid #e5e5e5;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:active {
            transform: translateY(0);
        }
        .quick-actions {
            padding: 20px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .quick-action-btn {
            width: 100%;
            margin: 8px 0;
            text-align: left;
            padding: 12px 16px;
            background: white;
            border: 2px solid #e5e5e5;
            color: #333;
            transition: all 0.3s;
        }
        .quick-action-btn:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: transparent;
        }
        .metric {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
        }
        .metric-title {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .plan-info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        .status-online {
            color: #10b981;
            font-weight: bold;
        }
        .feedback-buttons {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #e5e5e5;
        }
        .feedback-btn {
            flex: 1;
            padding: 8px 16px;
            font-size: 14px;
        }
        .feedback-btn.yes {
            background: #10b981;
        }
        .feedback-btn.no {
            background: #ef4444;
        }
        .upload-area {
            border: 2px dashed #e5e5e5;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
            cursor: pointer;
            transition: border-color 0.3s;
        }
        .upload-area:hover {
            border-color: #667eea;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-radius: 50%;
            border-top: 3px solid #667eea;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .ticket-badge {
            background: #fbbf24;
            color: #333;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Novarsis Support Bot</h1>
        <p>Official 24x7 Support for Novarsis AIO SEO Tool</p>
    </div>
    <div class="container">
        <div class="sidebar">
            <h3>📊 Support Dashboard</h3>
            <div class="metric">
                <div class="metric-title">Total Queries</div>
                <div class="metric-value" id="totalQueries">0</div>
            </div>
            <div class="metric">
                <div class="metric-title">Open Tickets</div>
                <div class="metric-value" id="openTickets">0</div>
            </div>
            <div class="plan-info" id="planInfo">
                <h4>💳 Your Plan</h4>
                <p id="planName">Loading...</p>
                <p id="planPrice"></p>
                <p id="planValidity"></p>
            </div>
            <div style="margin-top: 20px;">
                <h4>🟢 System Status</h4>
                <p class="status-online">✅ All services operational</p>
                <small>• SEO Analysis: Online</small><br>
                <small>• Report Generation: Online</small><br>
                <small>• API Services: Online</small>
            </div>
            <button onclick="resetChat()" style="width: 100%; margin-top: 20px;">
                🔄 Start New Chat
            </button>
        </div>
        <div class="main-content">
            <div class="chat-container" id="chatContainer">
                <div class="message assistant">
                    <div class="message-content">
                        🔍 <strong>Welcome to Novarsis Support!</strong><br><br>
                        I'm here to help you with:<br>
                        • Novarsis tool errors and issues<br>
                        • SEO analysis problems<br>
                        • Account and subscription queries<br>
                        • Technical support for Novarsis features
                    </div>
                </div>
            </div>
            <div class="input-container">
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    📸 Click to upload Novarsis error screenshot
                    <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFileUpload(event)">
                </div>
                <div class="input-wrapper">
                    <input type="text" id="messageInput" placeholder="Describe your Novarsis issue..." onkeypress="handleKeyPress(event)">
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
        <div class="quick-actions">
            <h3>⚡ Quick Actions</h3>
            <button class="quick-action-btn" onclick="quickAction('1')">1. Report Novarsis Error 🚨</button>
            <button class="quick-action-btn" onclick="quickAction('2')">2. Account & Subscription 💳</button>
            <button class="quick-action-btn" onclick="quickAction('3')">3. Connect with Expert 👥</button>
            <button class="quick-action-btn" onclick="quickAction('4')">4. Check Ticket Status 📋</button>
            <button class="quick-action-btn" onclick="quickAction('5')">5. Contact Novarsis 📞</button>
        </div>
    </div>
    <script>
        const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        let uploadedImageData = null;
        let waitingForTicket = false;
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;
            // Add user message to chat
            addMessage('user', message);
            input.value = '';
            // Show loading
            const loadingDiv = addMessage('assistant', '<div class="loading"></div>');
            try {
                // Check if waiting for ticket
                if (waitingForTicket) {
                    const response = await fetch('/api/check-ticket', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            ticket_number: message,
                            session_id: sessionId 
                        })
                    });
                    const data = await response.json();
                    loadingDiv.remove();
                    addMessage('assistant', data.message);
                    waitingForTicket = false;
                } else {
                    // Normal chat message
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            message: message,
                            session_id: sessionId,
                            image_data: uploadedImageData
                        })
                    });
                    const data = await response.json();
                    loadingDiv.remove();
                    // Add response with feedback buttons if needed
                    const msgDiv = addMessage('assistant', data.response);
                    if (data.show_feedback) {
                        addFeedbackButtons(msgDiv);
                    }
                    uploadedImageData = null;
                }
                updateStats();
            } catch (error) {
                loadingDiv.remove();
                addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
            }
        }
        async function quickAction(action) {
            // Add loading
            const loadingDiv = addMessage('assistant', '<div class="loading"></div>');
            try {
                const response = await fetch('/api/quick-action', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        action: action,
                        session_id: sessionId 
                    })
                });
                const data = await response.json();
                loadingDiv.remove();
                addMessage('assistant', data.response);
                if (data.need_ticket || data.check_status) {
                    waitingForTicket = true;
                }
                updateStats();
            } catch (error) {
                loadingDiv.remove();
                addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
            }
        }
        function addMessage(role, content) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.innerHTML = content.replace(/\\n/g, '<br>');
            messageDiv.appendChild(contentDiv);
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return messageDiv;
        }
        function addFeedbackButtons(messageDiv) {
            const feedbackDiv = document.createElement('div');
            feedbackDiv.className = 'feedback-buttons';
            feedbackDiv.innerHTML = `
                <button class="feedback-btn yes" onclick="sendFeedback('yes')">✅ Yes, Resolved</button>
                <button class="feedback-btn no" onclick="sendFeedback('no')">❌ No, Need Help</button>
            `;
            messageDiv.querySelector('.message-content').appendChild(feedbackDiv);
        }
        async function sendFeedback(feedback) {
            try {
                const response = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        feedback: feedback,
                        session_id: sessionId 
                    })
                });
                const data = await response.json();
                if (feedback === 'no' && data.ticket_id) {
                    addMessage('assistant', data.message + `<span class="ticket-badge">${data.ticket_id}</span>`);
                } else {
                    addMessage('assistant', data.message);
                }
                updateStats();
            } catch (error) {
                addMessage('assistant', 'Sorry, I encountered an error processing your feedback.');
            }
        }
        async function handleFileUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = function(e) {
                // Convert to base64
                uploadedImageData = e.target.result.split(',')[1];
                addMessage('user', `📸 Uploaded screenshot: ${file.name}`);
            };
            reader.readAsDataURL(file);
        }
        async function updateStats() {
            try {
                const response = await fetch(`/api/session/${sessionId}/stats`);
                const data = await response.json();
                document.getElementById('totalQueries').textContent = data.total_queries;
                document.getElementById('openTickets').textContent = data.open_tickets;
                if (data.current_plan) {
                    document.getElementById('planName').textContent = data.current_plan.name;
                    document.getElementById('planPrice').textContent = data.current_plan.price;
                    document.getElementById('planValidity').textContent = data.current_plan.validity;
                }
            } catch (error) {
                console.error('Error updating stats:', error);
            }
        }
        async function resetChat() {
            try {
                await fetch('/api/reset-session', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId })
                });
                // Clear chat container
                const chatContainer = document.getElementById('chatContainer');
                chatContainer.innerHTML = `
                    <div class="message assistant">
                        <div class="message-content">
                            🔍 <strong>Welcome to Novarsis Support!</strong><br><br>
                            I'm here to help you with:<br>
                            • Novarsis tool errors and issues<br>
                            • SEO analysis problems<br>
                            • Account and subscription queries<br>
                            • Technical support for Novarsis features
                        </div>
                    </div>
                `;
                updateStats();
            } catch (error) {
                console.error('Error resetting chat:', error);
            }
        }
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        // Initialize stats on load
        window.onload = function() {
            updateStats();
        }
    </script>
</body>
</html>
"""

# Run the server
if __name__ == "__main__":
    print("🚀 Starting Novarsis Support Bot API...")
    print("📍 Open http://localhost:8000 in your browser")
    print("📚 API docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)