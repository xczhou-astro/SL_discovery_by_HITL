import sys
import os
from flask import Flask

# Import config first (parses args and sets up device)
from configurations import config

# Set up logging BEFORE importing detector
from utils import Tee

os.makedirs(config.results_path, exist_ok=True)

log_file = open(os.path.join(config.results_path, 'app.log'), 'w')
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# Import detector and routes
from detector import SLDetector
from routes import api_bp, views_bp
from routes.api import init_api
from routes.views import init_views

# Create Flask app
app = Flask(__name__)

# Initialize detector
sl_detector = SLDetector()

# Initialize routes with detector
init_api(sl_detector)
init_views(sl_detector)

# Register blueprints
app.register_blueprint(api_bp)
app.register_blueprint(views_bp)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=6543)
    log_file.close()
