from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session
import os
import pymysql
from werkzeug.security import check_password_hash

login_bp = Blueprint('login_bp', __name__)


def get_db_connection():
    return pymysql.connect(
        host=os.getenv('DB_SERVER'),
        port=int(os.getenv('DB_PORT', '3306')),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASS'),
        database=os.getenv('DB_NAME'),
        cursorclass=pymysql.cursors.DictCursor
    )


@login_bp.route('/login', methods=['GET', 'POST'], endpoint='login')
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not email or not password:
            return jsonify({'success': False, 'message': 'All fields are required'})

        if '@' not in email or '.' not in email:
            return jsonify({'success': False, 'message': 'Invalid email format'})

        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, password FROM users1 WHERE email = %s", (email,))
                user = cursor.fetchone()

                if not user:
                    return jsonify({'success': False, 'message': 'Email not found'})

                if not check_password_hash(user['password'], password):
                    return jsonify({'success': False, 'message': 'Invalid password'})

                # Set session on success
                session['user_id'] = user['id']
                session['email'] = email
                return jsonify({'success': True, 'message': 'Login successful'})
        except Exception as e:
            return jsonify({'success': False, 'message': f"Error: {str(e)}"})
        finally:
            if conn is not None:
                conn.close()

    # GET method
    return render_template('login.html')


@login_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
