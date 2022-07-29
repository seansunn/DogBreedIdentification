import os
import glob
from flask import Flask, render_template, send_from_directory, url_for
from flask_uploads import UploadSet, IMAGES,configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
import model


UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'I_Love_Deep_Learning!'
app.config['UPLOADED_IMAGES_DEST'] = UPLOAD_FOLDER

images = UploadSet('images', IMAGES)
configure_uploads(app, images)


class UploadForm(FlaskForm):
    image = FileField(
        validators=[
            FileAllowed(images, 'Only images are allowed'),
            FileRequired('File field should not be empty')
        ]
    )

    submit = SubmitField('Upload')


@app.route("/", methods=['GET', 'POST'])
def upload_image():
    # initiate variable 'output'
    output = ''

    # empty the folder to prevent saving too many images
    if len(os.listdir(UPLOAD_FOLDER)) != 0:
        files = glob.glob(f'{UPLOAD_FOLDER}/*')
        for f in files:
            os.remove(f)

    form = UploadForm()
    if form.validate_on_submit():
        filename = images.save(form.image.data)
        file_url = url_for('get_file', filename=filename)
        # use pre-trained model to classify the saved image
        output = model.classifier(f'{UPLOAD_FOLDER}/{filename}')

    else:
        file_url = None

    return render_template("index.html", form=form, file_url=file_url, output=output)


@app.route('/uploads/<filename>')
# show the uploaded image from user
def get_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run()