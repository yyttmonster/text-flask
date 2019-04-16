import os
from flask_wtf import FlaskForm
from flask_wtf.file import FileField,FileRequired,FileAllowed
from wtforms import SubmitField


class UploadForm(FlaskForm):
    photo = FileField(validators=[
        FileAllowed(os.environ.get('IMAGE_EXTENSION'),u'Only image allowed!'),
        FileRequired(u'Please select one image.')])
    submit = SubmitField(u'Upload')

