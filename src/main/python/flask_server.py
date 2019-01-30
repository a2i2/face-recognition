from flask import Flask, url_for, send_from_directory, request, make_response, flash, redirect, render_template
from face_encoder.pipeline import Pipeline
from PIL import Image
from flask import send_file
import json
import logging, os
from werkzeug import secure_filename
from face_encoder.stages import PipelineData, PhotoExtraction, DownsampleImage, RotateImage, ImageTooDark, DetectAndAlignFaces, LargestFace, FaceTooBlurry, ExtractEncodingsResNet1
from surround import Stage, SurroundData, Surround, Config
from face_encoder import compare

app = Flask(__name__)

file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

UPLOAD_FOLDER = '/opt/theia/src/main/python/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
	return '.' in filename and \
	filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_new_folder(local_dir):
		newpath = local_dir
		if not os.path.exists(newpath):
				os.makedirs(newpath)
		return newpath

surround = Surround([PhotoExtraction(), DownsampleImage(), RotateImage(), ImageTooDark(), DetectAndAlignFaces(), LargestFace(), FaceTooBlurry(), ExtractEncodingsResNet1()])

config = Config()
config.read_config_files(["/opt/theia/src/main/python/config.yml"])
surround.set_config(config)
surround.init_stages()

@app.route("/")
def start():
	return render_template("index.html")

@app.route("/", methods=["POST"])
# curl -F "file=@image.jpg" http://192.168.99.100:80 >> test.json
#curl -F "file=@image.jpg" http://192.168.99.100:80
def home():
	# check if the post request has the file part
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']

	# if user does not select file, browser also
	# submit an empty part without filename
	if file.filename == '':
		flash('No selected file')
		return redirect(request.url)

	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

		img = filename

		data = PipelineData(img)
		surround.process(data)

		if data.error != None:
			return make_response(json.dumps(data.error))

		return make_response(json.dumps(data.output_data_dict))
	
@app.route("/upload", methods=['POST'])
def upload():
	# check if the post request has the file part
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']

	# if user does not select file, browser also
	# submit an empty part without filename
	if file.filename == '':
		flash('No selected file')
		return redirect(request.url)

	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

		img = filename

		data = PipelineData(img)
		surround.process(data)

		if data.error != None:
			return make_response(json.dumps(data.error))

		return make_response(json.dumps(data.output_data_dict))	

@app.route("/compare_json", methods=['POST'])
def compare_json():
	input1 = request.form['encoding1']
	input2 = request.form['encoding2']

	# convert string to json format
	json1 = json.loads(input1)
	json2 = json.loads(input2)

	# get encoding (128x1 vector) from json file
	encoding1 = json1["faceEncodings"]
	encoding2 = json2["faceEncodings"]
	
	distance = compare.distance(encoding1, encoding2)

	# get 2 digits after decimal point
	return str(distance)[0:4]

if __name__ == '__main__':
	# run http://192.168.99.100/ on host will work
	app.run(host='0.0.0.0', port=80)
