from os.path import join as create_path

from flask import Flask, request, redirect, render_template, url_for
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, send, emit

from nst import NeuralStyleTransfer

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
socketio = SocketIO(app, cors_allowed_origins="*")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@socketio.on('message')
def handle_message(curr):
    socketio.send(curr)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        
        content, style = request.files['content'], request.files['style']
        iterations, lr, alpha, beta = int(request.form["iterations"]), float(request.form["lr"]), int(request.form["alpha"]), int(request.form["beta"])
        
        if content.filename == '' or style.filename == "":
            return redirect(request.url)
        
        if allowed_file(content.filename) and allowed_file(style.filename):
            
            content_filename = secure_filename(content.filename)
            content_filepath = create_path(app.config['UPLOAD_FOLDER'], content_filename)
            content.save(content_filepath)
            
            style_filename = secure_filename(style.filename)
            style_filepath = create_path(app.config['UPLOAD_FOLDER'], style_filename)
            style.save(style_filepath)
            i = 0
            nst = NeuralStyleTransfer(content_filepath, style_filepath)
            nst.set_paramers_and_hyper_parameters(iterations, alpha=alpha, beta=beta, lr=lr)
            for val, is_training in nst.train():
                if is_training:
                    i+=1
                    handle_message(i+1)
                else: fname = nst.save_image(val)
            return render_template("output.html", content_path=url_for(app.config['UPLOAD_FOLDER'], filename=content_filename),
                                                    style_path=url_for(app.config['UPLOAD_FOLDER'], filename=style_filename),
                                                    generated_path=url_for("static", filename=fname))
    else:
        return render_template("input.html")


if __name__ == '__main__':
    socketio.run(app)


# Check Ports: sudo lsof -i -P -n | grep LISTEN, rememeber python wala. kill -9 9704(90 pid)
