import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from google.colab import drive
import tensorflow as tf
from datetime import *
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import time
import base64
from twilio.rest import Client

# Configurations
ALERT_OBJECT = 'helmet'
ALERT_IMAGE_NAME = 'alert_image.jpeg'
IS_SEND_ALERT = False

twilio_account_sid = '****'
twilio_auth_token = '****'
twilio_from_number = '+19*********'
twilio_to_number = '****'
    
gmail_user = 'user@email.com'
gmail_password = '****'

email_server_name = 'smtp.gmail.com'
email_server_port = '587'
email_sent_from = gmail_user
email_to = 'user@email.com'
email_subject = 'ALERT! Un authorized person detected! '+format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

execution_path = os.getcwd()

# send alert sms
def sendAlertSMS():

  try:
    # Your Account Sid and Auth Token from twilio.com/console
    client = Client(twilio_account_sid, twilio_auth_token)
    message = client.messages.create(body = email_subject, from_= twilio_from_number, to = twilio_to_number)
    #print(message.sid)
    print_log = "Successfully sent alert SMS"
    print(print_log)
    logData(print_log)

  except Exception:
   print_log = "Error: unable to send alert SMS"
   print(print_log)
   logData(print_log)

# send alert email
def sendAlertEmail():

  try:
    email_img_data = open(ALERT_IMAGE_NAME, 'rb').read()
    #encoded_img_content = base64.b64encode(email_img_data)  # base64
    email_msg = MIMEMultipart()
    email_msg['Subject'] = email_subject
    email_msg['From'] = email_sent_from
    email_msg['To'] = email_to

    email_body = "ALERT! Un authorized person detected! "+format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    email_text = MIMEText(email_body)
    email_msg.attach(email_text)
    email_image = MIMEImage(email_img_data, _subtype="jpeg", name=ALERT_IMAGE_NAME)
    email_msg.attach(email_image)

    email_server = smtplib.SMTP(email_server_name, email_server_port)
    email_server.ehlo()
    email_server.starttls()
    email_server.ehlo()
    email_server.login(gmail_user, gmail_password)
    email_server.sendmail(email_sent_from, email_to, email_msg.as_string())
    email_server.quit()
    print_log = "Successfully sent alert email"
    print(print_log)
    logData(print_log)

  except Exception:
    print_log = "Error: unable to send alert email"
    print(print_log)
    logData(print_log)

# Log data
def logData(log_data, file_name = 'log_file.txt'):

  updated_log_data = format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + " : " + log_data
  logfile = open(file_name, "a")
  logfile.write(updated_log_data+"\n")
  logfile.close()


def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())

model_path = 'converted-resnet50_csv_45.h5'    ## replace this with your model path
model = models.load_model(model_path, backbone_name='resnet50')
labels_to_names = {0: 'helmet'}                    ## replace with your model labels and its index value

video_path = 'youtube-samples/video2.mp4'  ## replace with input video path
output_path = 'youtube-samples/output_video2.mp4' ## replace with path where you want to save the output
fps = 15


vcapture = cv2.VideoCapture(video_path)

width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))  # uses given video width and height
height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
vwriter = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'),fps, (width, height)) #

num_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))

def run_detection_video(video_path):

    count = 0
    success = True
    start = time.time()
    while success:
        if count % 100 == 0:
            print("frame: ", count)
        count += 1
        # Read next image
        success, image = vcapture.read()

        if success:

            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            image = preprocess_image(image)
            image, scale = resize_image(image)

            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

            boxes /= scale
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.4:
                    break

                color = label_color(label)

                b = box.astype(int)
                draw_box(draw, b, color=color)

                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, b, caption)
            detected_frame = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
            vwriter.write(detected_frame)  # overwrites video slice

            print_log = "ALERT : Identified Object : "+str(ALERT_OBJECT) 
            print(print_log)
            logData(print_log)
            cv2.imwrite(ALERT_IMAGE_NAME, image)
            global IS_SEND_ALERT
            print('IS_SEND_ALERT : '+str(IS_SEND_ALERT))
            if IS_SEND_ALERT == False: 
                sendAlertEmail()
                sendAlertSMS()
                IS_SEND_ALERT = True

    vcapture.release()
    vwriter.release()  #
    end = time.time()

    print("Total Time: ", end - start)

run_detection_video(video_path)