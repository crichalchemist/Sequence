"""Flask REST API server for MQL5 integration."""
import logging
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from pathlib import Path

from mql5.bridge import bridge as mql5_bridge
from execution.backtest_manager import manager as backtest_manager
from train.training_manager import manager as training_manager
from data.pipeline_controller import controller as data_controller

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MQL5API")

# Create Flask app
app = Flask(__name__)
CORS(app)

# API version
API_VERSION = "1.0.0"
API_PORT = 5000


# ============================================================================
# MQL5 Live Data Endpoints
# ============================================================================

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "OPERATIONAL",
        "api_version": API_VERSION,
        "timestamp": datetime.now().isoformat(),
        "components": {
            "mql5_bridge": mql5_bridge.health_check(),
            "backtest_manager": {"status": "OK"},
            "training_manager": training_manager.get_queue_status(),
            "data_controller": data_controller.get_pipeline_status()
        }
    })


@app.route('/api/v1/live_data/tick', methods=['POST'])
def receive_live_tick():
    """Receive live price tick from MT5."""
    try:
        data = request.get_json()

        if not data or 'symbol' not in data:
            return jsonify({"error": "Missing symbol"}), 400

        success = mql5_bridge.store_live_tick(data)

        return jsonify({
            "success": success,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error receiving tick: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/live_data/<symbol>', methods=['GET'])
def get_live_data(symbol):
    """Get recent live data for symbol."""
    try:
        limit = request.args.get('limit', 100, type=int)
        data = mql5_bridge.get_live_data(symbol, limit)

        return jsonify({
            "symbol": symbol,
            "data": data,
            "count": len(data),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting live data: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Trading Signals Endpoints
# ============================================================================

@app.route('/api/v1/signals/create', methods=['POST'])
def create_signal():
    """Create trading signal."""
    try:
        data = request.get_json()

        required_fields = ['symbol', 'signal_type']
        if not all(field in data for field in required_fields):
            return jsonify({"error": f"Missing required fields: {required_fields}"}), 400

        success = mql5_bridge.create_signal(data)

        return jsonify({
            "success": success,
            "signal": data,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error creating signal: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/signals/pending', methods=['GET'])
def get_pending_signals():
    """Get pending signals to send to MT5."""
    try:
        signals = mql5_bridge.get_pending_signals(sent=False)

        return jsonify({
            "pending_signals": signals,
            "count": len(signals),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/signals/<int:signal_id>/sent', methods=['POST'])
def mark_signal_sent(signal_id):
    """Mark signal as sent to MT5."""
    try:
        success = mql5_bridge.mark_signal_sent(signal_id)

        return jsonify({
            "success": success,
            "signal_id": signal_id,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error marking signal sent: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Backtest Results Endpoints
# ============================================================================

@app.route('/api/v1/backtest/import', methods=['POST'])
def import_backtest_result():
    """Import backtest result from MT5."""
    try:
        data = request.get_json()

        success = mql5_bridge.import_backtest_result(data)

        return jsonify({
            "success": success,
            "backtest": data.get("strategy_name"),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error importing backtest: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/backtest/comparison/<symbol>', methods=['GET'])
def get_backtest_comparison(symbol):
    """Get backtest comparison for symbol."""
    try:
        limit = request.args.get('limit', 10, type=int)
        df = mql5_bridge.get_backtest_comparison(symbol, limit)

        if df.empty:
            return jsonify({
                "results": [],
                "symbol": symbol,
                "count": 0
            })

        return jsonify({
            "results": df.to_dict('records'),
            "symbol": symbol,
            "count": len(df),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting comparison: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/backtest/export', methods=['POST'])
def export_backtest_csv():
    """Export backtest results to CSV."""
    try:
        data = request.get_json()
        symbol = data.get("symbol")
        output_path = data.get("output_path", f"output_central/backtest_{symbol}.csv")

        success = mql5_bridge.export_backtest_csv(output_path, symbol)

        return jsonify({
            "success": success,
            "export_path": output_path,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error exporting backtest: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Training Management Endpoints
# ============================================================================

@app.route('/api/v1/training/queue/status', methods=['GET'])
def get_training_queue_status():
    """Get training queue status."""
    try:
        status = training_manager.get_queue_status()

        return jsonify({
            "queue_status": status,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/training/gpu/status', methods=['GET'])
def get_gpu_status():
    """Get GPU status and utilization."""
    try:
        gpu_status = training_manager.gpu_monitor.get_status()

        return jsonify({
            "gpu_status": gpu_status,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting GPU status: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Data Pipeline Endpoints
# ============================================================================

@app.route('/api/v1/data/pipeline/status', methods=['GET'])
def get_pipeline_status():
    """Get data pipeline status."""
    try:
        status = data_controller.get_pipeline_status()

        return jsonify({
            "pipeline_status": status,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Documentation Endpoint
# ============================================================================

@app.route('/api/v1/docs', methods=['GET'])
def get_api_docs():
    """Get API documentation."""
    return jsonify({
        "api_version": API_VERSION,
        "endpoints": {
            "health": {
                "path": "/api/v1/health",
                "method": "GET",
                "description": "System health check"
            },
            "live_data": {
                "receive": {
                    "path": "/api/v1/live_data/tick",
                    "method": "POST",
                    "description": "Receive live tick from MT5"
                },
                "retrieve": {
                    "path": "/api/v1/live_data/<symbol>",
                    "method": "GET",
                    "description": "Get recent live data for symbol"
                }
            },
            "signals": {
                "create": {
                    "path": "/api/v1/signals/create",
                    "method": "POST",
                    "description": "Create trading signal"
                },
                "pending": {
                    "path": "/api/v1/signals/pending",
                    "method": "GET",
                    "description": "Get pending signals for MT5"
                },
                "mark_sent": {
                    "path": "/api/v1/signals/<id>/sent",
                    "method": "POST",
                    "description": "Mark signal as sent"
                }
            },
            "backtest": {
                "import": {
                    "path": "/api/v1/backtest/import",
                    "method": "POST",
                    "description": "Import MT5 backtest result"
                },
                "comparison": {
                    "path": "/api/v1/backtest/comparison/<symbol>",
                    "method": "GET",
                    "description": "Get backtest comparison"
                },
                "export": {
                    "path": "/api/v1/backtest/export",
                    "method": "POST",
                    "description": "Export backtest to CSV"
                }
            },
            "training": {
                "queue": {
                    "path": "/api/v1/training/queue/status",
                    "method": "GET",
                    "description": "Get training queue status"
                },
                "gpu": {
                    "path": "/api/v1/training/gpu/status",
                    "method": "GET",
                    "description": "Get GPU status"
                }
            },
            "data": {
                "pipeline": {
                    "path": "/api/v1/data/pipeline/status",
                    "method": "GET",
                    "description": "Get data pipeline status"
                }
            }
        }
    })


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found",
        "path": request.path
    }), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({
        "error": "Internal server error",
        "details": str(error)
    }), 500


# ============================================================================
# Startup
# ============================================================================

if __name__ == '__main__':
    logger.info(f"Starting MQL5 REST API Server v{API_VERSION}")
    logger.info(f"Listening on http://localhost:{API_PORT}")
    logger.info(f"API Documentation: http://localhost:{API_PORT}/api/v1/docs")

    app.run(
        host='0.0.0.0',
        port=API_PORT,
        debug=True,
        use_reloader=False
    )

