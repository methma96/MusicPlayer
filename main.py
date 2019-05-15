
from werkzeug import secure_filename
import cv2
import uuid
import numpy as np
import pandas as pd
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from shutil import copyfile
import pickle
import zipfile
import pymysql
import os.path
import os, shutil
from os.path import isfile, join
import pymysql
import http.client, urllib.request, urllib.parse, urllib.error, base64,sys
import simplejson
from flask import Flask ,request, flash,render_template , url_for,send_from_directory,redirect
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)



app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = set(['zip'])

def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
	return render_template("index.html")

@app.route("/page2")
def page2():
	return render_template('page2.html')

emo=''
@app.route("/page3", methods=['GET', 'POST'])
def page3():
	global emo
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files :
			flash('No file part')
			return redirect(url_for('page2'))
		file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
		if file.filename == ''or file.filename.rsplit('.', 1)[1].lower() not in ['jpg','png','jpeg','gif']:
			return redirect(url_for('page2'))
			flash('No selected file')
			
		if file and file.filename.rsplit('.', 1)[1].lower() in ['jpg','png','jpeg','gif']:
		
			data=file.read()
	headers = {
	'Content-Type': 'application/octet-stream',
	'Ocp-Apim-Subscription-Key': '3e2333b880434e57a1793be3734ecdfc'
	}
	params = urllib.parse.urlencode({
	'returnFaceId': 'true',
	'returnFaceLandmarks': 'false',
	'returnFaceAttributes': 'emotion'
	})

	try:
		conn =http.client.HTTPSConnection('westcentralus.api.cognitive.microsoft.com')
		conn.request("POST", "/face/v1.0/detect?%s" % params, data, headers)
		response = conn.getresponse()
		data = response.read()
		parse_data=simplejson.loads(data)
		print(simplejson.dumps(parse_data,indent=2))
		val=parse_data[0]["faceAttributes"]["emotion"]
		emo=max(val,key=val.get)
		print(emo)
		conn.close()


	except Exception as e:
		print("[Errno {0}] {1}".format(e.errno, e.strerror))


	return render_template('page3.html',data=emo )


import librosa
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

'''
    function: extract_features
    input: path to mp3 files
    output: csv file containing features extracted

    This function reads the content in a directory and for each mp3 file detected
    reads the file and extracts relevant features using librosa library for audio
    signal processing
'''
def extract_feature(path):
    id = 1  # Song ID
    feature_set = pd.DataFrame()  # Feature Matrix

    # Individual Feature Vectors
    songname_vector = pd.Series()
    tempo_vector = pd.Series()
    total_beats = pd.Series()
    average_beats = pd.Series()
    chroma_stft_mean = pd.Series()
    chroma_stft_std = pd.Series()
    chroma_stft_var = pd.Series()
    chroma_cq_mean = pd.Series()
    chroma_cq_std = pd.Series()
    chroma_cq_var = pd.Series()
    chroma_cens_mean = pd.Series()
    chroma_cens_std = pd.Series()
    chroma_cens_var = pd.Series()
    mel_mean = pd.Series()
    mel_std = pd.Series()
    mel_var = pd.Series()
    mfcc_mean = pd.Series()
    mfcc_std = pd.Series()
    mfcc_var = pd.Series()
    mfcc_delta_mean = pd.Series()
    mfcc_delta_std = pd.Series()
    mfcc_delta_var = pd.Series()
    rmse_mean = pd.Series()
    rmse_std = pd.Series()
    rmse_var = pd.Series()
    cent_mean = pd.Series()
    cent_std = pd.Series()
    cent_var = pd.Series()
    spec_bw_mean = pd.Series()
    spec_bw_std = pd.Series()
    spec_bw_var = pd.Series()
    contrast_mean = pd.Series()
    contrast_std = pd.Series()
    contrast_var = pd.Series()
    rolloff_mean = pd.Series()
    rolloff_std = pd.Series()
    rolloff_var = pd.Series()
    poly_mean = pd.Series()
    poly_std = pd.Series()
    poly_var = pd.Series()
    tonnetz_mean = pd.Series()
    tonnetz_std = pd.Series()
    tonnetz_var = pd.Series()
    zcr_mean = pd.Series()
    zcr_std = pd.Series()
    zcr_var = pd.Series()
    harm_mean = pd.Series()
    harm_std = pd.Series()
    harm_var = pd.Series()
    perc_mean = pd.Series()
    perc_std = pd.Series()
    perc_var = pd.Series()
    frame_mean = pd.Series()
    frame_std = pd.Series()
    frame_var = pd.Series()


    # Traversing over each file in path
    file_data = [f for f in listdir(path) if isfile (join(path, f)) and f.rsplit('.', 1)[1].lower() in ['mp3']]
    for line in file_data:
        if ( line[-1:] == '\n' ):
            line = line[:-1]

        # Reading Song
        songname = path + line
        y, sr = librosa.load(songname, duration=60)
        S = np.abs(librosa.stft(y))

        # Extracting Features
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        poly_features = librosa.feature.poly_features(S=S, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_delta = librosa.feature.delta(mfcc)

        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)

        # Transforming Features
        songname_vector.set_value(id, line)  # song name
        tempo_vector.set_value(id, tempo)  # tempo
        total_beats.set_value(id, sum(beats))  # beats
        average_beats.set_value(id, np.average(beats))
        chroma_stft_mean.set_value(id, np.mean(chroma_stft))  # chroma stft
        chroma_stft_std.set_value(id, np.std(chroma_stft))
        chroma_stft_var.set_value(id, np.var(chroma_stft))
        chroma_cq_mean.set_value(id, np.mean(chroma_cq))  # chroma cq
        chroma_cq_std.set_value(id, np.std(chroma_cq))
        chroma_cq_var.set_value(id, np.var(chroma_cq))
        chroma_cens_mean.set_value(id, np.mean(chroma_cens))  # chroma cens
        chroma_cens_std.set_value(id, np.std(chroma_cens))
        chroma_cens_var.set_value(id, np.var(chroma_cens))
        mel_mean.set_value(id, np.mean(melspectrogram))  # melspectrogram
        mel_std.set_value(id, np.std(melspectrogram))
        mel_var.set_value(id, np.var(melspectrogram))
        mfcc_mean.set_value(id, np.mean(mfcc))  # mfcc
        mfcc_std.set_value(id, np.std(mfcc))
        mfcc_var.set_value(id, np.var(mfcc))
        mfcc_delta_mean.set_value(id, np.mean(mfcc_delta))  # mfcc delta
        mfcc_delta_std.set_value(id, np.std(mfcc_delta))
        mfcc_delta_var.set_value(id, np.var(mfcc_delta))
        rmse_mean.set_value(id, np.mean(rmse))  # rmse
        rmse_std.set_value(id, np.std(rmse))
        rmse_var.set_value(id, np.var(rmse))
        cent_mean.set_value(id, np.mean(cent))  # cent
        cent_std.set_value(id, np.std(cent))
        cent_var.set_value(id, np.var(cent))
        spec_bw_mean.set_value(id, np.mean(spec_bw))  # spectral bandwidth
        spec_bw_std.set_value(id, np.std(spec_bw))
        spec_bw_var.set_value(id, np.var(spec_bw))
        contrast_mean.set_value(id, np.mean(contrast))  # contrast
        contrast_std.set_value(id, np.std(contrast))
        contrast_var.set_value(id, np.var(contrast))
        rolloff_mean.set_value(id, np.mean(rolloff))  # rolloff
        rolloff_std.set_value(id, np.std(rolloff))
        rolloff_var.set_value(id, np.var(rolloff))
        poly_mean.set_value(id, np.mean(poly_features))  # poly features
        poly_std.set_value(id, np.std(poly_features))
        poly_var.set_value(id, np.var(poly_features))
        tonnetz_mean.set_value(id, np.mean(tonnetz))  # tonnetz
        tonnetz_std.set_value(id, np.std(tonnetz))
        tonnetz_var.set_value(id, np.var(tonnetz))
        zcr_mean.set_value(id, np.mean(zcr))  # zero crossing rate
        zcr_std.set_value(id, np.std(zcr))
        zcr_var.set_value(id, np.var(zcr))
        harm_mean.set_value(id, np.mean(harmonic))  # harmonic
        harm_std.set_value(id, np.std(harmonic))
        harm_var.set_value(id, np.var(harmonic))
        perc_mean.set_value(id, np.mean(percussive))  # percussive
        perc_std.set_value(id, np.std(percussive))
        perc_var.set_value(id, np.var(percussive))
        frame_mean.set_value(id, np.mean(frames_to_time))  # frames
        frame_std.set_value(id, np.std(frames_to_time))
        frame_var.set_value(id, np.var(frames_to_time))

        print(songname)
        id = id+1

    # Concatenating Features into one csv and json format
    feature_set['song_name'] = songname_vector  # song name
    feature_set['tempo'] = tempo_vector  # tempo
    feature_set['total_beats'] = total_beats  # beats
    feature_set['average_beats'] = average_beats
    feature_set['chroma_stft_mean'] = chroma_stft_mean  # chroma stft
    feature_set['chroma_stft_std'] = chroma_stft_std
    feature_set['chroma_stft_var'] = chroma_stft_var
    feature_set['chroma_cq_mean'] = chroma_cq_mean  # chroma cq
    feature_set['chroma_cq_std'] = chroma_cq_std
    feature_set['chroma_cq_var'] = chroma_cq_var
    feature_set['chroma_cens_mean'] = chroma_cens_mean  # chroma cens
    feature_set['chroma_cens_std'] = chroma_cens_std
    feature_set['chroma_cens_var'] = chroma_cens_var
    feature_set['melspectrogram_mean'] = mel_mean  # melspectrogram
    feature_set['melspectrogram_std'] = mel_std
    feature_set['melspectrogram_var'] = mel_var
    feature_set['mfcc_mean'] = mfcc_mean  # mfcc
    feature_set['mfcc_std'] = mfcc_std
    feature_set['mfcc_var'] = mfcc_var
    feature_set['mfcc_delta_mean'] = mfcc_delta_mean  # mfcc delta
    feature_set['mfcc_delta_std'] = mfcc_delta_std
    feature_set['mfcc_delta_var'] = mfcc_delta_var
    feature_set['rmse_mean'] = rmse_mean  # rmse
    feature_set['rmse_std'] = rmse_std
    feature_set['rmse_var'] = rmse_var
    feature_set['cent_mean'] = cent_mean  # cent
    feature_set['cent_std'] = cent_std
    feature_set['cent_var'] = cent_var
    feature_set['spec_bw_mean'] = spec_bw_mean  # spectral bandwidth
    feature_set['spec_bw_std'] = spec_bw_std
    feature_set['spec_bw_var'] = spec_bw_var
    feature_set['contrast_mean'] = contrast_mean  # contrast
    feature_set['contrast_std'] = contrast_std
    feature_set['contrast_var'] = contrast_var
    feature_set['rolloff_mean'] = rolloff_mean  # rolloff
    feature_set['rolloff_std'] = rolloff_std
    feature_set['rolloff_var'] = rolloff_var
    feature_set['poly_mean'] = poly_mean  # poly features
    feature_set['poly_std'] = poly_std
    feature_set['poly_var'] = poly_var
    feature_set['tonnetz_mean'] = tonnetz_mean  # tonnetz
    feature_set['tonnetz_std'] = tonnetz_std
    feature_set['tonnetz_var'] = tonnetz_var
    feature_set['zcr_mean'] = zcr_mean  # zero crossing rate
    feature_set['zcr_std'] = zcr_std
    feature_set['zcr_var'] = zcr_var
    feature_set['harm_mean'] = harm_mean  # harmonic
    feature_set['harm_std'] = harm_std
    feature_set['harm_var'] = harm_var
    feature_set['perc_mean'] = perc_mean  # percussive
    feature_set['perc_std'] = perc_std
    feature_set['perc_var'] = perc_var
    feature_set['frame_mean'] = frame_mean  # frames
    feature_set['frame_std'] = frame_std
    feature_set['frame_var'] = frame_var

    # Converting Dataframe into CSV Excel and JSON file
    feature_set.to_csv('Emotion_features.csv')
    feature_set.to_json('Emotion_features.json')

# Extracting Feature Function Call



@app.route('/page5',methods=['POST'])
def page5():
	unique_filename = str(uuid.uuid4())
	target = os.path.join(APP_ROOT,unique_filename)
	if not os.path.isdir(target):
		os.mkdir(target)

	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files :
			flash('No file part')
			return redirect(url_for('page4'))
		file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
		if file.filename == '' or allowed_file(file.filename)==False:
			return redirect(url_for('page4'))
			flash('No selected file')
			
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(APP_ROOT, filename))
			zip_ref = zipfile.ZipFile(os.path.join(APP_ROOT, filename), 'r')
			zip_ref.extractall(target)
			zip_ref.close()
			extract_feature(target+'/')
		
	loaded_model = pickle.load(open('finalized_model3.sav', 'rb'))
	p=pd.read_csv('Emotion_features.csv')
	feature1=p.loc[:, 'tempo':]
	f=feature1.values
	print(loaded_model.predict(f))
	print(loaded_model.predict(f)[0])
	l=[]
	for i in range(len(f)):
		a=[]
		songName=p.loc[i][1]
		emotionType=loaded_model.predict(f)[i]
		a.append(songName)
		a.append(emotionType)	
		l.append(a)
# disconnect from server
	db = pymysql.connect("remotemysql.com","bXQkGwO66B","wBAXO36s0D","bXQkGwO66B",3306)
	cursor = db.cursor()
	sql = """ INSERT INTO Songs (songName,emotionType) VALUES (%s,%s)"""

	for row in l:
		print(row)
		try:
			cursor.execute(sql, row)
			db.commit()
		except:
			db.rollback()

	db.close()

	db = pymysql.connect("remotemysql.com","bXQkGwO66B","wBAXO36s0D","bXQkGwO66B",3306)
	cursor = db.cursor()
	if emo=='neutral' or emo=='happiness':
		sql="SELECT DISTINCT songName FROM Songs where emotionType='happy'"
		try:
			cursor.execute(sql)

		except:
			print("Error")

			
	elif(emo=='sadness' or emo=='disgust' or emo=='contempt') :
		sql="SELECT DISTINCT songName FROM Songs where emotionType='sad'"
		try:
			cursor.execute(sql)
				
		except:
			print("Error")

			
	elif(emo=='anger'):
		sql="SELECT DISTINCT songName FROM Songs where emotionType='angry'"
		try:
			cursor.execute(sql)

		except:
			print("Error")
			

	else:
		sql="SELECT DISTINCT songName FROM Songs where emotionType='relax'"
		try:
			cursor.execute(sql)

		except:
			print("Error")
		
	new_files = os.listdir(target)
	for f in new_files:
		m=target+'/'+f
		d=os.path.dirname(os.path.abspath(__file__))+'/static/'
		des=d+f
		copyfile(m, des)
	data = cursor.fetchall()
	db.close()

	return render_template('page5.html',data=data)


@app.route('/page51')
def page51():


	return render_template('page5.html')

@app.route("/page4")
def page4():
	return render_template('page4.html')


if __name__ == "__main__":
	app.run(host='0.0.0.0')
