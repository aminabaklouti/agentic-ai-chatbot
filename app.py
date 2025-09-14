from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from graph_workflow import invoke_graph
import os
from typing import List

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Agentic AI Chatbot API is running"
    })

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    Main chat endpoint that processes user queries using the AI agent
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' field in request body"
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                "error": "Query cannot be empty"
            }), 400
        
        max_length = data.get('max_length', 1000)
        
        # Invoke the agent
        result = invoke_graph(query)
        
        # Truncate response if needed
        response_text = result["response"]
        if max_length and len(response_text) > max_length:
            response_text = response_text[:max_length] + "..."
        
        return jsonify({
            "response": response_text,
            "tools_used": result["tools_used"],
            "success": True,
            "message": "Query processed successfully"
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Error processing query: {str(e)}",
            "success": False
        }), 500

@app.route('/tools')
def list_tools():
    """List available tools"""
    return jsonify({
        "tools": [
            {
                "name": "arxiv",
                "description": "Search academic papers on ArXiv",
                "use_cases": ["Research questions", "Scientific topics", "Academic studies"]
            },
            {
                "name": "wikipedia", 
                "description": "Search Wikipedia for general knowledge",
                "use_cases": ["Definitions", "Historical facts", "General information"]
            },
            {
                "name": "tavily",
                "description": "Web search for current information",
                "use_cases": ["Current events", "Recent news", "Real-time data"]
            }
        ]
    })

@app.route('/examples')
def get_examples():
    """Get example queries"""
    return jsonify({
        "examples": [
            {
                "category": "Research",
                "queries": [
                    "What are the latest developments in quantum computing?",
                    "Recent advances in transformer architectures",
                    "Latest papers on computer vision"
                ]
            },
            {
                "category": "General Knowledge", 
                "queries": [
                    "Explain machine learning in simple terms",
                    "History of artificial intelligence",
                    "What is quantum entanglement?"
                ]
            },
            {
                "category": "Current Events",
                "queries": [
                    "Latest AI news today",
                    "Current tech industry trends",
                    "Recent developments in renewable energy"
                ]
            }
        ]
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)