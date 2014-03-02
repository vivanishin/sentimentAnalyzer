# coding=utf-8
import json 
from sklearn import *
from numpy import *
import re
class SentimentAnalyzer:
    #constructor (optional)
	def __init__(self):
		self.simples = [
				'?', '!', '.', 'T_T', '*'
		]
		self.regexen = [ 
			re.compile(u'[А-Я][^А-Я]', re.UNICODE), # Uppercase capital letters 
			re.compile(u'[А-Я]{3,}', re.UNICODE), # CAPS
			re.compile('[:=][\|/]'), # =\
			re.compile('[:=;]-?[)DpP\]]'), # :-) :p :]
			re.compile("[:=][-']?[(]"), # :'( :(
			re.compile('[(]-?[:=;]'), # (: 
			re.compile('[D)]-?[:=;]'), # D: 
			re.compile('\S{2,}\.(com|ru|org|net|ua|co|su)\S*'), # link
			re.compile('\S{2,}\.(com|ru|org|net|ua|co|su)\S*$') # link at the end
		]
		self.lowcase = [
			re.compile(u'печал', re.UNICODE),
			re.compile(u'ненави', re.UNICODE),
			re.compile(u'ч[её]рт', re.UNICODE), 
			re.compile(u'бля', re.UNICODE),
			re.compile(u'[её]б', re.UNICODE),
			re.compile(u'муд[аио]', re.UNICODE),
			re.compile(u'п[ие]д[оа]?р', re.UNICODE),
			re.compile(u'[^о]ху[йеёяю]',re.UNICODE), 
			re.compile(u'хер', re.UNICODE),
			re.compile(u'хрен', re.UNICODE),
			re.compile(u'[^а-я]дрян', re.UNICODE),
			re.compile(u'бе[сш]', re.UNICODE),
			re.compile(u'ущерб', re.UNICODE),
			re.compile(u'г[ао]в[её]?н|дерьм', re.UNICODE),
			re.compile(u'урод|к[оа]з[её]?л', re.UNICODE),
			re.compile(u'адск|адов|[^а-я]жоп', re.UNICODE),			
			re.compile(u'г[ао]ндон|мраз[ьиоея]', re.UNICODE),			
			re.compile(u'[^а-я]д[еи]бил|[^а-я]дур[aод]', re.UNICODE),
			re.compile(u'сук[аи]', re.UNICODE),
			re.compile(u'пизд[аоеяуиыёл][^т]', re.UNICODE),
			re.compile(u'пиздат', re.UNICODE),
			re.compile(u'благодар', re.UNICODE),
			re.compile(u'спасиб', re.UNICODE),
			re.compile(u'замечате', re.UNICODE),
			re.compile(u'молодец', re.UNICODE),
			re.compile(u'приятн', re.UNICODE),
			re.compile(u'позити', re.UNICODE),
			re.compile(u'кайф', re.UNICODE),
			re.compile(u'рай', re.UNICODE),
			re.compile(u'ништяк', re.UNICODE),
			re.compile(u'вес[её]л', re.UNICODE),
			re.compile(u'радост', re.UNICODE),
			re.compile(u'поздрав.?л', re.UNICODE),
			re.compile(u'люб[лияо]', re.UNICODE),
			re.compile(u'обожа',re.UNICODE),
		]

    #trainer of classifier (mandatory)
	def get_polarity(self, json_tweet):
		pol = json_tweet['polarity']
		if pol == 'negative':
			return -1
		elif pol == 'positive':
			return +1
		else:
			return 0

	def reverse_map(self, label):
		if label < -0.5:
			return 'negative'
		elif label > 0.5:
			return 'positive'
		else:
			return 'neutral'

	def get_featutes_string(self, string, p=True):
#		print string
		ans = []
		ans += [string.count(')') - string.count('(')]
		for sub in self.simples:
			ans += [string.count(sub)]
		if p:
			print string
		for regex in self.regexen:
			if p:
				print regex.findall(string), regex.pattern
#			str2 = string.decode('utf-8')
			res = regex.findall(string)
			ans += [len(res)]
		string = string.lower()
		for regex in self.lowcase:
			if p:
				print ans
			ans += [len(regex.findall(string))]
			if p:
				print ans
			if p:
				print regex.findall(string), regex.pattern
		return ans
	
	def get_features(self, corpus):
		return map(lambda x: self.get_featutes_string(x['text'], False), corpus)

	def train(self, training_corpus):
		self.classifier = svm.SVC(class_weight='auto')
		arrX = array(self.get_features(training_corpus), dtype = float64)
		arrY = array(map(self.get_polarity, training_corpus), dtype = float64)
		self.classifier.fit(arrX, arrY)
		
	#returns sentiment score of input text (mandatory)
	def getClasses(self, texts):
#		print u''.join(texts)
		x = array(map(lambda x: self.get_featutes_string(x), texts), dtype = float64)
		print map(self.reverse_map, self.classifier.predict(x))
#		return map(self.reverse_map, self.classifier.predict(x))

#debug!
#print training_corpus[0]['polarity']
#print training_corpus[0]['text']
#print len(training_corpus)

analyzer = SentimentAnalyzer()
training_corpus = json.load(open('tweets.json'))
analyzer.train(training_corpus[:-1])
#analyzer.getClasses(map (lambda x: x['text'], training_corpus))
#analyzer.getClasses([training_corpus[1]['text']])
analyzer.getClasses([training_corpus[-1]['text']])
#print analyzer.getClasses(texts)
