import logging
import asyncio
from flask import Blueprint, render_template, request, jsonify

from app.services.market_research_service import handle_marketlens_query

logger = logging.getLogger(__name__)

marketlens_bp = Blueprint("marketlens", __name__)


@marketlens_bp.route('/')
def strategy_coach_page():
    """
    Renders the Market Research / Strategy Agent page.
    """
    return render_template('marketlens.html')


@marketlens_bp.route('/', methods=['POST'])
def unified_market_api():
    """
    Unified Endpoint: Handles Market Research, Product Scouting, and Chat.
    """
    try:
        data = request.get_json(force=True)

        # backward compatibility for old frontend
        user_query = data.get('query') or data.get('niche')

        if not user_query:
            return jsonify({
                'success': False,
                'error': 'Please provide a query.'
            }), 400

        logger.info(f"MarketLens query received: {user_query}")

        # ---- Run async agent safely ----
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(handle_marketlens_query(user_query))

        return jsonify({
            'success': True,
            'type': result.get('type', 'chat'),
            'report': result.get('content'),
            'analysis': result.get('content'),
            'products': result.get('products', [])
        })

    except Exception as e:
        logger.exception(f"MarketLens API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
