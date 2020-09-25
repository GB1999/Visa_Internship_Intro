import os
import json
import re
import nltk
import urllib
import cv2
import numpy as np

from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class PersonalityAnalysis:
    

    def __init__(self, path, name):
        self.file_path = os.path.join(path, 'Source Data/Likes/like.js')
        self.image_path = os.path.join(path, 'Processed Images')
        self.name = name
        self.images = []
        self.popularity_dict = {}

        self.punctuation = re.compile(r'[-.?,:;()|0-9]')
        self.emojis = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
        self.urls = re.compile('(?P<url>https?://[^\s]+)')

    def load_twitter_data(self):
        with open(self.file_path,'r',encoding="utf8") as js_file:
            data_string = js_file.read().replace('\n', '')
        data_string = data_string[data_string.index('=') + 1:]
        self.like_dict = json.loads(data_string)
    
    def load_yolo(self):
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        self.classes = []
        with open('coco.names', "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.popularity_dict = dict((label,0) for label in self.classes)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def url_to_image(self, url):
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # return the image
        return image
    
    def analyze_images(self):
        # parse json string
        for like in self.like_dict:
            like_text = like['like']['fullText']
            tweet_id = like['like']['tweetId']
            tweet_url = like['like']['expandedUrl']
            
            score_text = self.punctuation.sub(r'', like_text)
            score_text = self.emojis.sub(r'', score_text)          

            score = SentimentIntensityAnalyzer().polarity_scores(score_text)

            try:
                # check if tweet contains attachment
                if re.search('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', like_text):
                    page_data = urllib.request.urlopen(tweet_url).read()
                    scraper = BeautifulSoup(page_data, 'html.parser')
                    image_tag = scraper.find_all('img', {'alt': True, 'src': True})[4]
                    link = image_tag.get('src')
                    # check if image is media image (vs profile image)
                    if 'media' in link:
                        print(score_text)
                        image = self.url_to_image(link)
                        self.detect_objects(image, tweet_id, score)

            except Exception as e: 
                print(e)
        
        

    def detect_objects(self, image, image_id, score):
        image = cv2.resize(image, None, fx=0.4, fy=0.4)
        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        pos_score = score['pos']
        neg_score = score['neg']
        tot_score = pos_score - neg_score
        print(tot_score)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                self.popularity_dict[label] += tot_score
                color = self.colors[i]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label, (x, y + 30), font, 3, color, 3)
                
        cv2.imwrite(os.path.join(self.image_path , "Image" +  image_id + '.jpg'), image)

    def print_personality(self):
        sorted_labels = sorted(self.popularity_dict.items(), key = lambda x :x[1], reverse=True)
        top5 = sorted_labels[:5]
        bottom5 = sorted_labels[-5:]
        print(f'Hi, my name is {self.name}, and I like ',  end =" ")

        for rank in top5:
            label = rank[0]
            # get plural form
            if(label[-1] == 's'):
                label = label + 'es'
            else:
                label = label + 's'
            # print list with oxford comma
            if rank == top5[-1]:
                print(f' and {label}.' )
            else:
                print(f'{label}, ' , end =" ")
        
        print("However, I don't like  ",  end =" ")

        for rank in bottom5:
            label = rank[0]
            # get plural form
            if(label[-1] == 's'):
                label = label + 'es'
            else:
                label = label + 's'
            # print list with oxford comma
            if rank == bottom5[-1]:
                print(f' and {label}.', end =" " )
            else:
                print(f'{label}, ' , end =" ")


gage_analysis = PersonalityAnalysis("D:\Documents\Side Projects (CS)\Python\Visa Intro", "Gage Benham")
gage_analysis.load_twitter_data()
gage_analysis.load_yolo()
gage_analysis.analyze_images()
gage_analysis.print_personality()




    























    






































































































































































































