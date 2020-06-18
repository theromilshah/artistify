import os
from datetime import datetime

from flask import Flask, request, redirect, render_template, url_for
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, send, emit
from flask_material import Material


from nst import training_loop

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
Material(app)
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
socketio = SocketIO(app, cors_allowed_origins="*")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@socketio.on('message')
def handle_message(curr, total):
    socketio.send(curr)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        content, style, iterations = request.files['content'], request.files['style'], int(request.form["iterations"])
        if content.filename == '' or style.filename == "":
            return redirect(request.url)
        if allowed_file(content.filename) and allowed_file(style.filename):
            content_filename = secure_filename(content.filename)
            content_filepath = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            content.save(content_filepath)
            style_filename = secure_filename(style.filename)
            style_filepath = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            style.save(style_filepath)
            name = ""
            for iteration in training_loop(content_filepath, style_filepath, iterations=iterations):
                print("<<<", iteration)
                if isinstance(iteration, int): handle_message(iteration+1, iterations)
                name = iteration
        return render_template("output.html", content_path=url_for(app.config['UPLOAD_FOLDER'], filename=content_filename),
                                                style_path=url_for(app.config['UPLOAD_FOLDER'], filename=style_filename),
                                                generated_path=url_for("static", filename=name))
    else:
        return render_template("input.html")


if __name__ == '__main__':
    socketio.run(app)


# Check Ports: sudo lsof -i -P -n | grep LISTEN, rememeber python wala. kill -9 9704(90 pid)
