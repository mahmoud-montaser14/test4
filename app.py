# import os
# import logging
# import json
# from flask import Flask, request, render_template, jsonify
# from io import BytesIO
# from utils import preprocess_image, predict_and_format_result

# # Allowed file extensions
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# # MAX_CONTENT_LENGTH = 0.5 * 1024 * 1024  # Limit file size to 0.5MB
# MAX_CONTENT_LENGTH = 150 * 1024  # Limit file size to 150 KB


# # Flask application
# app = Flask(__name__, template_folder='templates')
# app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# # Configure structured logging
# class JsonFormatter(logging.Formatter):
#     def format(self, record):
#         log_record = {
#             'timestamp': self.formatTime(record, self.datefmt),
#             'level': record.levelname,
#             'message': record.getMessage(),
#             'module': record.module,
#             'funcName': record.funcName,
#             'lineNo': record.lineno,
#         }
#         if record.exc_info:
#             log_record['exception'] = self.formatException(record.exc_info)
#         return json.dumps(log_record)

# # Set up logging
# log_file = "app.log"
# json_formatter = JsonFormatter()
# file_handler = logging.FileHandler(log_file)
# file_handler.setFormatter(json_formatter)
# file_handler.setLevel(logging.DEBUG)

# logging.basicConfig(level=logging.DEBUG, handlers=[file_handler])
# app.logger.addHandler(file_handler)

# def allowed_file(filename):
#     """Check if file has a valid extension."""
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     """Render the index page and handle form submissions."""
#     result = None
#     image_filename = None
#     error_message = None

#     if request.method == 'POST':
#         file = request.files.get('image')
#         if not file:
#             error_message = "No file uploaded. Please upload an image."
#             app.logger.error(error_message)
#             return render_template('index.html', result=None, error=error_message)

#         if file and allowed_file(file.filename):
#             try:
#                 file_content = BytesIO(file.read())
#                 result = predict_and_format_result(file_content)
#                 image_filename = file.filename
#                 return render_template('index.html', result=result, image=image_filename)
#             except Exception as e:
#                 app.logger.error(f"Error during prediction: {e}")
#                 error_message = f"Prediction error: {str(e)}"
#                 return render_template('index.html', result=None, error=error_message)
#         else:
#             error_message = "Invalid file type. Please upload a valid image."
#             app.logger.error(error_message)
#             return render_template('index.html', result=None, error=error_message)

#     return render_template('index.html', result=result, image=image_filename, error=error_message)

# @app.errorhandler(413)
# def request_entity_too_large(error):
#     app.logger.error("Uploaded file exceeds size limit.")
#     if request.content_type == 'application/json' or 'application/json' in request.headers.get('Accept', ''):
#         return jsonify({'error': 'File size exceeds the 150 KB limit.'}), 413
#     return render_template('index.html', error="File size exceeds the 150 KB limit."), 413

# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     """Handle API predictions and return plain-text or JSON message."""
#     try:
#         file = request.files.get('image')
#         if not file:
#             app.logger.error("No file uploaded.")
#             return "No file uploaded. Please upload an image.", 400

#         # Check if the file size exceeds the limit
#         file.seek(0, os.SEEK_END)  # Move to the end of the file
#         file_size = file.tell()  # Get the current position, which is the size
#         file.seek(0)  # Reset file pointer to the beginning

#         max_size_in_bytes = 150 * 1024  # 150 KB
#         if file_size > max_size_in_bytes:
#             error_message = "File size exceeds the 150 KB limit."
#             app.logger.error(error_message)
#             return error_message, 400

#         if file and allowed_file(file.filename):
#             file_content = BytesIO(file.read())
#             result = predict_and_format_result(file_content)
#             if result == "Anomalous":
#                 app.logger.error("Image is anomalous and cannot be classified.")
#                 return "Image is anomalous and cannot be classified.", 400
#             return result, 200  # Directly return the class name as plain text

#         else:
#             app.logger.error("Invalid file type.")
#             return "Invalid file type. Please upload a valid image.", 400
#     except Exception as e:
#         app.logger.error(f"Prediction error: {e}")
#         return f"Prediction error: {str(e)}", 500
import os
import logging
import json
from flask import Flask, request, render_template, jsonify
from io import BytesIO
from utils import preprocess_image, predict_and_format_result

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 150 * 1024  # Limit file size to 150 KB

# Flask application
app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Configure structured logging
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'funcName': record.funcName,
            'lineNo': record.lineno,
        }
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_record)

# Set up logging
log_file = "app.log"
json_formatter = JsonFormatter()
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(json_formatter)
file_handler.setLevel(logging.DEBUG)

logging.basicConfig(level=logging.DEBUG, handlers=[file_handler])
app.logger.addHandler(file_handler)

def allowed_file(filename):
    """Check if file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the index page and handle form submissions."""
    result = None
    image_filename = None
    error_message = None

    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            error_message = "No file uploaded. Please upload an image."
            app.logger.error(error_message)
            return render_template('index.html', result=None, error=error_message)

        if file and allowed_file(file.filename):
            try:
                file_content = BytesIO(file.read())
                result = predict_and_format_result(file_content)
                image_filename = file.filename
                return render_template('index.html', result=result, image=image_filename)
            except Exception as e:
                app.logger.error(f"Error during prediction: {e}")
                error_message = f"Prediction error: {str(e)}"
                return render_template('index.html', result=None, error=error_message)
        else:
            error_message = "Invalid file type. Please upload a valid image."
            app.logger.error(error_message)
            return render_template('index.html', result=None, error=error_message)

    return render_template('index.html', result=result, image=image_filename, error=error_message)

@app.errorhandler(413)
def request_entity_too_large(error):
    app.logger.error("Uploaded file exceeds size limit.")
    # if request.content_type == 'application/json' or 'application/json' in request.headers.get('Accept', ''):
    return 'File size exceeds the 150 KB limit.', 413
    # return render_template('index.html', error="File size exceeds the 150 KB limit."), 413


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Handle API predictions."""
    file = request.files.get('image')
    if not file:
        app.logger.error("No file uploaded.")
        return 'No file uploaded. Please upload an image.', 400

    # Check if the file size exceeds the limit
    file.seek(0, os.SEEK_END)  # Move to the end of the file
    file_size = file.tell()  # Get the current position, which is the size
    file.seek(0)  # Reset file pointer to the beginning

    if file_size > MAX_CONTENT_LENGTH:
        error_message = "File size exceeds the 0.5 MB limit."
        app.logger.error(error_message)
        return error_message, 400

    if file and allowed_file(file.filename):
        try:
            file_content = BytesIO(file.read())
            result = predict_and_format_result(file_content)
            if result == "Anomalous":
                app.logger.error("Image is anomalous and cannot be classified.")
                return 'Image is anomalous and cannot be classified.', 400
            return result
        except Exception as e:
            app.logger.error(f"API Prediction error: {e}")
            return f"Prediction error: {str(e)}", 500
    else:
        app.logger.error("Invalid file type.")
        return 'Invalid file type. Please upload a valid image.', 400

# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     """Handle API predictions."""
#     file = request.files.get('image')
#     if not file:
#         app.logger.error("No file uploaded.")
#         return jsonify({'error': 'No file uploaded. Please upload an image.'}), 400

#     if file and allowed_file(file.filename):
#         try:
#             file_content = BytesIO(file.read())
#             result = predict_and_format_result(file_content)
#             if result == "Anomalous":
#                 app.logger.error("Image is anomalous and cannot be classified.")
#                 return jsonify({'error': 'Image is anomalous and cannot be classified.'}), 400
#             return jsonify(result)
#         except Exception as e:
#             app.logger.error(f"API Prediction error: {e}")
#             return jsonify({'error': f"Prediction error: {str(e)}"}), 500
#     else:
#         app.logger.error("Invalid file type.")
#         return jsonify({'error': 'Invalid file type. Please upload a valid image.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
