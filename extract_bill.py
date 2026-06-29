from flask import Blueprint, render_template, request, session, redirect, url_for
import os
import shutil
import subprocess
import tempfile
import pymysql

from src.extraction import extract_fields

extract_bill_bp = Blueprint(
    'extract_bill_bp', __name__, template_folder='templates')

DB_CONFIG = {
    'host': os.getenv('DB_SERVER'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASS'),
    'database': os.getenv('DB_NAME')
}


def get_db_connection():
    return pymysql.connect(**DB_CONFIG)


def _resolve_pdftotext():
    """Locate poppler's pdftotext without a machine-specific hardcoded path.

    Resolution order:
      1. POPPLER_PDFTOTEXT env var (explicit override),
      2. a `pdftotext` binary already on PATH (Linux/macOS/Docker, or a Windows
         install added to PATH).
    Returns the executable path, or None if poppler is unavailable.
    """
    env = os.getenv('POPPLER_PDFTOTEXT')
    if env and os.path.exists(env):
        return env
    return shutil.which('pdftotext')


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF, preferring the pure-Python pdfplumber backend
    (cross-platform, no system binary) and falling back to poppler's pdftotext.

    The previous implementation hardcoded ``C:\\poppler-24.07.0\\...`` so it only
    ran on one machine; this version is portable and Docker-friendly.
    """
    # Backend 1: pdfplumber (pure Python, no external binary).
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = [(page.extract_text() or '') for page in pdf.pages]
        text = '\n'.join(pages).strip()
        if text:
            return text
    except ImportError:
        pass  # pdfplumber not installed -> try poppler

    # Backend 2: poppler pdftotext, located dynamically (no hardcoded path).
    pdftotext = _resolve_pdftotext()
    if not pdftotext:
        raise Exception(
            'No PDF backend available. Install pdfplumber (pip install pdfplumber) '
            'or set POPPLER_PDFTOTEXT to a pdftotext executable.')

    with tempfile.TemporaryDirectory() as tmp:
        output_path = os.path.join(tmp, 'extracted_text.txt')
        result = subprocess.run(
            [pdftotext, pdf_path, output_path], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception('Error extracting text from PDF: ' + result.stderr)
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                return f.read()
    raise Exception('Extracted text file not found.')


def find_bill_details(text):
    """Extract (signed_amount, bill_date) from receipt text.

    Day-5 change: delegates to the Day-2 champion extractor (`rules_smart`,
    amount-acc 0.58 / date-acc 0.87 vs the old regex's 0.15 / 0.49). The naive
    first-decimal regex is retained inside the extractor purely as a fallback,
    so the (amount, date) return signature is unchanged and the Flask route and
    every caller keep working.
    """
    res = extract_fields(text)
    amount = float(res.get('amount') or 0.0)
    bill_date = res.get('date') or 'Not found'
    return -abs(amount), bill_date


@extract_bill_bp.route('/extract_bill', methods=['GET', 'POST'])
def extract_bill():
    if 'user_id' not in session:
        return redirect(url_for('login_bp.login'))

    bill_amount = ''
    bill_date = ''
    error_message = ''
    debug_text = ''

    if request.method == 'POST':
        if 'pdf' not in request.files:
            error_message = 'No file uploaded.'
        else:
            file = request.files['pdf']
            if file and file.filename.endswith('.pdf'):
                temp_path = os.path.join('temp', file.filename)
                os.makedirs('temp', exist_ok=True)
                file.save(temp_path)
                try:
                    text = extract_text_from_pdf(temp_path)
                    debug_text = text
                    amount, date = find_bill_details(text)

                    bill_amount = amount
                    bill_date = date

                    # Save to DB — scoped to the logged-in user (multi-tenancy fix).
                    conn = get_db_connection()
                    with conn.cursor() as cursor:
                        cursor.execute(
                            "INSERT INTO transactions (user_id, description, amount, date) "
                            "VALUES (%s, %s, %s, %s)",
                            (session['user_id'], 'Bill', amount, date)
                        )
                        conn.commit()
                    conn.close()
                except Exception as e:
                    error_message = str(e)
                finally:
                    os.remove(temp_path)

    return render_template('extract_bill.html',
                           bill_amount=bill_amount,
                           bill_date=bill_date,
                           error_message=error_message,
                           debug_text=debug_text)
