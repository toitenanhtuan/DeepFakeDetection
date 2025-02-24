import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from utils import process_video, analyze_deepfake
from models import db, Analysis

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the database
db.init_app(app)

with app.app_context():
    # Create database tables
    db.create_all()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        flash('No video file uploaded')
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process video
            frames, face_frames, original_frames = process_video(filepath)

            # Analyze for deepfake
            result = analyze_deepfake(frames, original_frames)

            # Save analysis to database
            analysis = Analysis(
                filename=filename,
                result=result,
                timestamp=datetime.utcnow()
            )
            db.session.add(analysis)
            db.session.commit()

            # Clean up
            os.remove(filepath)

            return render_template('analysis.html', 
                                frames=frames, 
                                face_frames=face_frames,
                                result=result)
        except Exception as e:
            flash(f'Error processing video: {str(e)}')
            return redirect(url_for('index'))

    flash('Invalid file type')
    return redirect(url_for('index'))

@app.route('/history')
def history():
    analyses = Analysis.query.order_by(Analysis.timestamp.desc()).all()
    return render_template('history.html', analyses=analyses)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)