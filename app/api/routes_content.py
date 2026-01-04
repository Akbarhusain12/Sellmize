from flask import Blueprint, render_template, request, jsonify
from app.services.content_service import generate_product_content  # adjust import path if needed

content_bp = Blueprint("content", __name__)

@content_bp.route('/')
def content_generator_page():
    """Serves the content generator page."""
    return render_template('content_generator.html')

@content_bp.route('/', methods=['POST'])
def generate_content_api():
    try:
        data = request.get_json(force=True)
        attributes = data.get("attributes", {})

        result = generate_product_content(attributes)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
