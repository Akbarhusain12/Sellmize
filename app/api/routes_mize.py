import logging
import asyncio
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from app.db.connection import database as db
from app.db.models import ChatSession, ChatMessage
from app.agents.unified_agent import SellMizeAssistant

logger = logging.getLogger(__name__)
mize_bp = Blueprint("mize_bp", __name__)

# --- AGENT SETUP ---
try:
    mize_agent = SellMizeAssistant()
except:
    mize_agent = None

# --- PAGE ROUTE ---
@mize_bp.route('/mize')
@mize_bp.route('/mize/')
@login_required
def mize_page():
    return render_template('mize.html')

# --- API 1: CREATE NEW SESSION (FIXED) ---
@mize_bp.route('/api/mize/session/new', methods=['POST'])
@login_required
def create_session():
    try:
        # FIX: Check if the last session is empty. If so, reuse it.
        last_session = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.created_at.desc()).first()
        
        # Check if last session exists AND has 0 messages
        if last_session and len(last_session.messages) == 0:
            return jsonify({'success': True, 'session_id': last_session.id})

        # Otherwise, create a new one
        new_session = ChatSession(user_id=current_user.id, title="New Chat")
        db.session.add(new_session)
        db.session.commit()
        return jsonify({'success': True, 'session_id': new_session.id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# --- API 2: GET CHAT HISTORY (FIXED) ---
@mize_bp.route('/api/mize/history', methods=['GET'])
@login_required
def get_history():
    # Fetch last 30 sessions
    sessions = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.updated_at.desc()).limit(30).all()
    
    # FIX: Only return sessions that actually have messages
    # This hides all the empty "New Chat" entries from the sidebar
    valid_history = []
    for s in sessions:
        if len(s.messages) > 0:
            valid_history.append({
                'id': s.id, 
                'title': s.title, 
                'date': s.updated_at.strftime('%d %b')
            })
            
    return jsonify({'success': True, 'history': valid_history})

# --- API 3: GET MESSAGES ---
@mize_bp.route('/api/mize/session/<int:session_id>', methods=['GET'])
@login_required
def get_session_messages(session_id):
    session = ChatSession.query.get_or_404(session_id)
    if session.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
        
    messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.created_at.asc()).all()
    
    return jsonify({
        'success': True,
        'messages': [{
            'sender': m.sender,
            'content': m.content,
            'meta_data': m.meta_data
        } for m in messages]
    })

# --- API 4: SEND MESSAGE ---
@mize_bp.route('/api/mize/chat', methods=['POST'])
@login_required
async def chat_api():
    try:
        data = request.get_json()
        query = data.get('query')
        session_id = data.get('session_id')
        mode = data.get('mode', 'auto')

        if not session_id:
            return jsonify({'error': 'Session ID missing'}), 400

        # 1. Save USER message to DB
        user_msg = ChatMessage(session_id=session_id, sender='user', content=query)
        db.session.add(user_msg)
        
        # 2. Run AI
        if not mize_agent:
            return jsonify({'error': 'AI Offline'}), 503
        
        ai_result = await mize_agent.process_request(query, mode)
        
        # 3. Save AI message to DB
        is_complex = ai_result.get('type') in ['product', 'content']
        content_text = ai_result.get('content') if not is_complex else "See results below."
        meta_data = ai_result if is_complex else None

        ai_msg = ChatMessage(
            session_id=session_id, 
            sender='ai', 
            content=str(content_text), 
            meta_data=meta_data
        )
        db.session.add(ai_msg)

        # 4. Auto-Update Title (if it's the first message)
        session = ChatSession.query.get(session_id)
        if session.title == "New Chat":
            # Generate a short title from the user query (first 6 words)
            new_title = " ".join(query.split()[:6])
            if len(new_title) > 30: new_title += "..."
            session.title = new_title
            session.updated_at = datetime.utcnow()
        
        db.session.commit()

        return jsonify({'success': True, 'result': ai_result})

    except Exception as e:
        db.session.rollback()
        logger.error(f"Chat Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500