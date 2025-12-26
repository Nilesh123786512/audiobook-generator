from flask import Flask, request, render_template, url_for # Added url_for
from werkzeug.utils import secure_filename
from pdf_utils import *  # Assuming this contains extract_and_structure_pymupdf
from tts_utils import *
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env if present

app = Flask(__name__)
BOOK_FILE_NAME='book'
TIME_TAKEN=0
AUDIO_FILE_NAME='audio'

# Define Voice Options Helper
def get_voice_options():
    voices_dir = "voices"
    options = []
    if os.path.exists(voices_dir):
        for f in os.listdir(voices_dir):
            if f.lower().endswith(('.wav', '.mp3', '.flac')):
                options.append({"name": f, "value": f})
    
    if not options:
        options.append({"name": "Default (Upload a .wav file to 'voices' folder)", "value": "default.wav"})
    return options

@app.route("/", methods=["GET", "POST"])  # Assuming POST is for some other functionality or future use
def home():
    return render_template("home.html")

@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        if 'formFile' not in request.files:
            # Handle case where no file part is in the request
            return "No file part in the request.", 400

        file = request.files['formFile']
        global BOOK_FILE_NAME
        BOOK_FILE_NAME=file.filename[:-4]

        if file.filename == '':
            # Handle case where no file is selected
            return "No file selected.", 400

        if file and file.filename.endswith(".pdf"):
            # Ensure the target directory exists
            upload_folder = "static/uploads"
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
        
            pdf_save_path = os.path.join(upload_folder, file.filename)

            if os.path.exists(pdf_save_path):
                try:
                    os.remove(pdf_save_path)
                except OSError as e:
                    print(f"Warning: Could not remove existing file {pdf_save_path}: {e}")

            file.save(pdf_save_path)

            try:
                pdf_reader = PdfReader(pdf_save_path)
                page_count = len(pdf_reader.pages)
            except Exception as e:
                print(f"Error reading PDF {pdf_save_path}: {e}")
                return "Error processing PDF file. Please ensure it's a valid PDF.", 500


            return render_template('display_pdf.html',
                                   pdf_path=pdf_save_path, # Path for url_for('static', ...)
                                   page_count=page_count,
                                   voices=get_voice_options()) # Pass voice options to the template
        else:
            print(f"File type not supported: {file.filename}")
            return '''
                <h1>File type not supported! Only PDF files are allowed.</h1>
                <p><a href="/">Go back</a></p>'''

@app.route("/audio_generate", methods=["POST","GET"])
def audio_generate():
    if request.method == "POST":
        try:
            init_page = int(request.form["initial_page"])
            end_page = int(request.form["final_page"])
            
            # Logic to determining voice: Check file upload first, then dropdown
            selected_voice = request.form.get("voice_option")
            
            if 'custom_voice' in request.files:
                file = request.files['custom_voice']
                if file and file.filename != '':
                    filename = secure_filename(file.filename)
                    # Ensure unique or safe saving. Overwriting same name is allowed here.
                    save_path = os.path.join("voices", filename)
                    file.save(save_path)
                    selected_voice = filename

            if not selected_voice:
                 return "Voice option is missing.", 400

            if not (1 <= init_page <= end_page):
                 return "Invalid page range selected.", 400

        except ValueError:
            return "Invalid page numbers or voice option provided.", 400
        except KeyError:
            return "Missing form data (page numbers or voice option).", 400

        pdf_file_path = f"static/uploads/{BOOK_FILE_NAME}.pdf"
        if not os.path.exists(pdf_file_path):
            return "Uploaded PDF not found. Please re-upload.", 404

        # text=preprocess_text(extract_specific_pages("static/uploads/book.pdf",pages=range(init_page,end_page+1)))
        text = extract_and_structure_pymupdf(pdf_file_path, pages=range(init_page, end_page + 1))
        
        # DEBUG: Check what text was actually extracted
        print(f"DEBUG: Extracted text length: {len(text)} characters")
        print(f"DEBUG: Text preview: {text[:100]}...")

        if not text or not text.strip():
            return "No text could be extracted from the selected PDF pages.", 400

        try:
            # Assuming KPipeline handles lang_code based on voice or a default like 'a' is fine.
            pipeline = load_pipeline() # from tts_utils
            
            # Pass the selected_voice to generate_audio
            # Now generate_audio raises descriptive exceptions on failure
            audio_data, time_taken = generate_audio(text, pipeline, voice_option=selected_voice)

        except Exception as e:
            import traceback
            traceback.print_exc() # Print full stack trace to server logs
            return f"An error occurred during audio generation: {str(e)}", 500

        # Ensure static/uploads directory exists for saving audio
        audio_output_folder = "static/" # Save audio in the same folder as PDF
        if not os.path.exists(audio_output_folder):
            os.makedirs(audio_output_folder)

        high_quality_audio_path, low_quality_audio_path = save_audio(audio_data, output_path_folder=audio_output_folder,page_numbers=[init_page,end_page],voice_option=selected_voice,book_name=BOOK_FILE_NAME)
        global AUDIO_FILE_NAME
        AUDIO_FILE_NAME=high_quality_audio_path[:-4]
        global TIME_TAKEN
        TIME_TAKEN=time_taken
        print("Audio file written successfully")
        



    return render_template("play_audio.html",
                            time_taken=TIME_TAKEN,
                            high_audio_path=f"{AUDIO_FILE_NAME}.wav",
                            low_audio_path=f"{AUDIO_FILE_NAME}.mp3",
                            pdf_file_path=f"static/uploads/{BOOK_FILE_NAME}.pdf",
                            )

if __name__ == "__main__":
    # Ensure the static directory and uploads subdirectory exist at startup
    if not os.path.exists("static"):
        os.makedirs("static")
    if not os.path.exists("static/uploads"):
        os.makedirs("static/uploads")
    # Ensure voices directory exists for Chatterbox references
    if not os.path.exists("voices"):
        os.makedirs("voices")

    app.run(debug=True)