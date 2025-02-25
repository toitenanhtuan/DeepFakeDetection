import os
from datetime import datetime
from flask import render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from extensions import app, db, logger

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    with app.app_context():
        from utils import process_video, analyze_deepfake
        from deepfake_detector import DeepfakeDetector
        # Import models after db initialization
        from models import Analysis
        # Create database tables
        db.create_all()
        logger.info("Database tables created successfully")

        # Initialize detector to get available models
        detector = DeepfakeDetector()
        available_models = detector.available_models
        # Create accuracy dictionary (you can adjust these values based on your model's performance)
        accuracies = {
            10: 84,
            20: 87,
            40: 89,
            60: 90,
            80: 91,
            100: 93
        }
except Exception as e:
    logger.error(f"Error initializing database: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', 
                         available_models=sorted(available_models.keys()),
                         accuracies=accuracies)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        flash('No video file uploaded')
        return redirect(url_for('index'))

    file = request.files['video']
    sequence_length = request.form.get('sequence_length')

    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if sequence_length:
        sequence_length = int(sequence_length)

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process video
            frames, face_frames, original_frames = process_video(filepath)

            # Initialize detector with sequence length
            detector = DeepfakeDetector(sequence_length)

            # Analyze for deepfake
            result, confidence = detector.analyze_video_frames(original_frames)

            # Save analysis to database
            analysis = Analysis(
                filename=filename,
                result=result,
                timestamp=datetime.utcnow()
            )
            db.session.add(analysis)
            db.session.commit()

            return render_template('analysis.html', 
                               frames=frames, 
                               face_frames=face_frames,
                               result=result,
                               video_path=filename,
                               model_info=detector.model_info)

        except Exception as e:
            flash(f'Error processing video: {str(e)}')
            return redirect(url_for('index'))

    flash('Invalid file type')
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history')
def history():
    analyses = Analysis.query.order_by(Analysis.timestamp.desc()).all()
    return render_template('history.html', analyses=analyses)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)