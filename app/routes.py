from flask import render_template, request, jsonify
from app import app
import os
from app import textObject


@app.route('/')
@app.route('/index')
def index():
    user = {"name": "index"}
    return render_template('index.html', user=user)


@app.route('/upload_image', methods=['POST', 'GET'])
def upload_image():
    img_name = request.form.get('name')
    image = request.files['image']
    print(img_name,image)
    if image and img_name.split('.')[1] in os.environ.get('IMAGE_EXTENSION'):
        file_path = os.path.join(os.getenv('ABSOLUTE_PATH'), os.getenv('UPLOAD_PATH'),
                                 image.filename)
        image.save(file_path)
        json_dict = textObject.output(file_path)
        print(json_dict)

        # output = ''
        # output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get('OUTPUT_PATH'),
        #              'img_1.jpg')

        return jsonify({'message': 'Upload successfully!', 'data':json_dict})
    else:
        return jsonify({'message': 'Check your image!'})

