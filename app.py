import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from find_object import find_object_in_scene
import cv2

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXT = {'png','jpg','jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.secret_key = 'replace-with-a-random-secret'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'template' not in request.files or 'scene' not in request.files:
            flash('Please upload both template and scene images.')
            return redirect(request.url)
        template_file = request.files['template']
        scene_file = request.files['scene']
        if template_file.filename == '' or scene_file.filename == '':
            flash('No selected file(s).')
            return redirect(request.url)
        if not (allowed_file(template_file.filename) and allowed_file(scene_file.filename)):
            flash('Allowed file types: png, jpg, jpeg')
            return redirect(request.url)

        tname = secure_filename(template_file.filename)
        sname = secure_filename(scene_file.filename)
        tpath = os.path.join(app.config['UPLOAD_FOLDER'], 't_' + tname)
        spath = os.path.join(app.config['UPLOAD_FOLDER'], 's_' + sname)
        template_file.save(tpath)
        scene_file.save(spath)

        result_img, msg = find_object_in_scene(tpath, spath, min_matches=10)
        if result_img is None:
            flash("Result: " + msg)
            return render_template('index.html', result_url=None, message=msg)

        result_filename = 'result_' + sname
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_img)
        result_url = url_for('static', filename='results/' + result_filename)
        return render_template('index.html', result_url=result_url, message=msg)

    return render_template('index.html', result_url=None, message=None)

if __name__ == '__main__':
    app.run(debug=True)

