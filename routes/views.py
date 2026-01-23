import os
from flask import render_template, request, jsonify, send_from_directory
from routes import views_bp
from configurations import config


# This will be set by app.py
sl_detector = None


def init_views(detector):
    """Initialize view routes with detector instance"""
    global sl_detector
    sl_detector = detector


@views_bp.route('/')
def index():
    return render_template('index.html')


@views_bp.route('/gallery')
def gallery():
    return render_template('gallery.html')


@views_bp.route('/app/submit_selections', methods=['POST'])
def submit_selections():
    print('\n=== /app/submit_selections called ===')
    
    try:
        data = request.json
        sl_names = data.get('sl_names', [])
        non_sl_names = data.get('non_sl_names', [])
        mode = data.get('mode', 'random')
        
        print(f'Received SL names: {sl_names}')
        print(f'Received Non-SL names: {non_sl_names}')
        print(f'Mode: {mode}')
        
        sl_detector.add_selections(sl_names, non_sl_names)
        
        print(f'After add_selections - SL count: {len(sl_detector.selected_sl_names)}, Non-SL count: {len(sl_detector.selected_non_sl_names)}')
        num_submission_train = sl_detector.num_submission_train
        
        response_data = {
            'success': True,
            'round': sl_detector.current_round,
            'sl_count': len(sl_detector.selected_sl_names),
            'non_sl_count': len(sl_detector.selected_non_sl_names),
            'total_submissions': sl_detector.total_submissions,
            'available_count': sl_detector.get_available_galaxies(),
        }
        
        if sl_detector.total_submissions % num_submission_train == 0 and sl_detector.total_submissions > 0:
            response_data['should_train'] = True
        else:
            response_data['should_train'] = False
            # Load next batch of images
            if sl_detector.model_trained:
                names, scores = sl_detector.get_images()
            else:
                names, scores = sl_detector.get_random_batch(10)
            response_data['galaxy_names'] = names
            response_data['scores'] = scores
            response_data['model_trained'] = sl_detector.model_trained
        
        return jsonify(response_data)
    except Exception as e:
        print(f'Error submitting selections: {str(e)}')
        return jsonify({'error': str(e)}), 500


@views_bp.route('/images/<filename>')
def serve_image(filename):
    # Add .jpg extension if not present
    if not filename.endswith('.jpg'):
        filename = f"{filename}.jpg"
    return send_from_directory(sl_detector.images_path, filename)


@views_bp.route('/visualizations/<filename>')
def serve_visualization(filename):
    try:
        if hasattr(sl_detector, 'round_save_path') and sl_detector.round_save_path:
            return send_from_directory(sl_detector.round_save_path, filename)
        else:
            # Try to find the latest results directory
            results_dirs = [d for d in os.listdir(config.results_path) if d.startswith('round_')]
            if results_dirs:
                latest_dir = max(results_dirs, key=lambda x: int(x.split('_')[1]))
                return send_from_directory(os.path.join(config.results_path, latest_dir), filename)
            else:
                return "No visualizations available", 404
    except Exception as e:
        return f"Error serving visualization: {str(e)}", 500
