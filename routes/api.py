import os
import numpy as np
from flask import request, jsonify, send_from_directory
from routes import api_bp
from configurations import config


# This will be set by app.py
sl_detector = None


def init_api(detector):
    """Initialize API routes with detector instance"""
    global sl_detector
    sl_detector = detector


@api_bp.route('/get_random_batch')
def api_get_random_batch():
    names, scores = sl_detector.get_random_batch(10)
    
    try:
        return jsonify({
            'success': True,
            'galaxy_names': names,
            'scores': scores, 
            'round': sl_detector.current_round,
            'sl_count': len(sl_detector.selected_sl_names),
            'non_sl_count': len(sl_detector.selected_non_sl_names),
            'total_submissions': sl_detector.total_submissions,
            'available_count': sl_detector.get_available_galaxies(),
            'model_trained': sl_detector.model_trained,
        })
    except Exception as e:
        print('Error geting random batch.')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/get_images')
def api_get_images():
    try:
        print(f'\n=== /api/get_images called ===')
        print(f'Model trained: {sl_detector.model_trained}')
        
        if sl_detector.model_trained:
            print('Using get_images() (smart selection)')
            names, scores = sl_detector.get_images()
        else:
            print('Using get_random_batch() (random selection)')
            names, scores = sl_detector.get_random_batch(10)
        
        print(f'Returning {len(names)} images')
        
        return jsonify({
            'success': True,
            'galaxy_names': names,
            'scores': scores,
            'round': sl_detector.current_round,
            'sl_count': len(sl_detector.selected_sl_names),
            'non_sl_count': len(sl_detector.selected_non_sl_names),
            'total_submissions': sl_detector.total_submissions,
            'available_count': sl_detector.get_available_galaxies(),
            'model_trained': sl_detector.model_trained,
        })
    except Exception as e:
        print('Error getting images:', str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/run_training', methods=['POST'])
def run_training():
    epochs = request.json.get('epochs', 300)
    
    try:
        result = sl_detector.train_model(epochs)
        return jsonify(result)
    except Exception as e:
        print('Error running training:', str(e))
        return jsonify({'error': str(e)}), 500


@api_bp.route('/get_status')
def get_status():
    return jsonify({
        'success': True,
        'round': sl_detector.current_round,
        'sl_count': len(sl_detector.selected_sl_names),
        'non_sl_count': len(sl_detector.selected_non_sl_names),
        'total_submissions': sl_detector.total_submissions,
        'available_count': sl_detector.get_available_galaxies(),
        'model_trained': sl_detector.model_trained,
    })


@api_bp.route('/get_gallery_data')
def get_gallery_data():
    try:
        print('\n=== /api/get_gallery_data called ===')
        
        # Separate confirmed and user-selected
        confirmed_sl_names = set(sl_detector.cowls_sl_names)
        
        # Preserve selection order for user-selected items
        user_selected_sl_names = [name for name in sl_detector.selected_sl_names if name not in confirmed_sl_names]
        user_selected_non_sl_names = sl_detector.selected_non_sl_names
        
        print(f'Confirmed SL: {len(confirmed_sl_names)}')
        print(f'User-selected SL: {len(user_selected_sl_names)}')
        print(f'User-selected non-SL: {len(user_selected_non_sl_names)}')
        if len(user_selected_non_sl_names) > 0:
            print(f'First 3 non-SL names: {user_selected_non_sl_names[:3]}')
        
        # Define grade order for sorting confirmed items (high to low)
        high_grades = ['M25'] + [f'S{i:02d}' for i in range(10, 0, -1)]
        grade_order = {grade: i for i, grade in enumerate(high_grades)}
        
        # Create confirmed SL items sorted by grade
        confirmed_items = []
        for name in sl_detector.cowls_sl_names:
            confirmed_items.append({
                'name': name,
                'type': 'sl',
                'is_confirmed': True,
                'grade': sl_detector.name_to_grade.get(name, 'N/A')
            })
        
        # Sort confirmed items by grade (high to low)
        confirmed_items.sort(key=lambda x: (grade_order.get(x['grade'], 999), x['name']))
        
        # Create user-selected SL items in selection order
        user_selected_sl_items = []
        for name in user_selected_sl_names:
            user_selected_sl_items.append({
                'name': name,
                'type': 'sl',
                'is_confirmed': False,
                'grade': sl_detector.name_to_grade.get(name, 'N/A')
            })
        
        # Create user-selected non-SL items in selection order
        user_selected_non_sl_items = []
        for name in user_selected_non_sl_names:
            user_selected_non_sl_items.append({
                'name': name,
                'type': 'non_sl',
                'is_confirmed': False,
                'grade': 'N/A'
            })
        
        # Combine: confirmed SL first, then user-selected SL, then user-selected non-SL
        gallery_items = confirmed_items + user_selected_sl_items + user_selected_non_sl_items
        
        print(f'Total gallery items: {len(gallery_items)}')
        print(f'  - Confirmed SL items: {len(confirmed_items)}')
        print(f'  - User-selected SL items: {len(user_selected_sl_items)}')
        print(f'  - User-selected non-SL items: {len(user_selected_non_sl_items)}')
        
        return jsonify({
            'success': True,
            'items': gallery_items.tolist() if isinstance(gallery_items, np.ndarray) else gallery_items,
            'total_count': len(gallery_items),
            'confirmed_count': len(confirmed_sl_names),
            'selected_sl_count': len(user_selected_sl_names),
            'selected_non_sl_count': len(user_selected_non_sl_names)
        })
    except Exception as e:
        print('Error getting gallery data:', str(e))
        return jsonify({'error': str(e)}), 500
