from flask import Blueprint, request, jsonify, render_template
import os
import pymysql
from werkzeug.security import generate_password_hash

signup_bp = Blueprint('signup_bp', __name__)


def get_db_connection():
    return pymysql.connect(
        host=os.getenv('DB_SERVER'),
        port=int(os.getenv('DB_PORT', '3306')),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASS'),
        database=os.getenv('DB_NAME'),
        cursorclass=pymysql.cursors.DictCursor
    )


@signup_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm-password', '').strip()

        if not name or not email or not password or not confirm_password:
            return jsonify({'success': False, 'message': 'All fields are required'})

        if password != confirm_password:
            return jsonify({'success': False, 'message': 'Passwords do not match'})

        if '@' not in email or '.' not in email:
            return jsonify({'success': False, 'message': 'Invalid email format'})

        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id FROM users1 WHERE email = %s", (email,))
                if cursor.fetchone():
                    return jsonify({'success': False, 'message': 'Email already exists'})

                hashed_password = generate_password_hash(password)
                cursor.execute(
                    "INSERT INTO users1 (firstname, lastname, email, password) VALUES (%s, %s, %s, %s)",
                    (name, '', email, hashed_password)
                )
                conn.commit()
                return jsonify({'success': True, 'message': 'User registered successfully'})
        except Exception as e:
            print("Signup error:", str(e))
            return jsonify({'success': False, 'message': str(e)})
        finally:
            if conn is not None:
                conn.close()

    return render_template('signup.html')
