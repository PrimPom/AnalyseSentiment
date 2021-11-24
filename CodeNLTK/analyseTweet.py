# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:50:18 2021

@author: primp
"""

import re
import pandas as pd
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score



def get_tweets():
    """Récupération des tweets en ligne en utilisant l'api de twitter 
        Demande faite à Twitter pour avoir un compte développeur
        Ce qui permet d'avoir les clés d'accès suivantes
        - consumer_key
        - consumer_secret
        - access_key
        - access_secret
    """
    import tweepy as tw
    consumer_key ="IX8g9gWldgAo8SuCSeCoduMg0"
    consumer_secret ="UWMPPTlbiUki4f2jEUTepOtuiHAgDbeUv6TXluWH7o9Ev9lrL4"
    access_key ="1382752360594804736-Sz2V0e6XdL1yiHNghtI9fVgK2sdumG" 
    access_secret="iF8PoyoVdIH9IK3P0HZyjNEp3Vecx9oHYHwj33mMsc2Gc"
    "Gestion de l'authentification"
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key,access_secret)
    api = tw.API(auth, wait_on_rate_limit=True)
    
    requete_tweets ="Covid-19 OR Covid OR Corona OR Pandémie OR épidémie OR Coronavirus OR virus"
    tweets = tw.Cursor(api.search,
                   q = requete_tweets,
                   lang = "fr",
                   since='2021-03-01').items(5000)

    all_tweets = [tweet.text for tweet in tweets]
    return all_tweets
    


#Cette fonction pour nettoyer les tweets récupéré
def nlp_pipeline(text):
    #print("Dans le pipelin")
    text = text.lower() 
    text = text.replace('\n', ' ').replace('\r', '')
    text = ' '.join(text.split())
    text = re.sub(r"[A-Za-z\.]*[0-9]+[A-Za-z%°\.]*", "", text)
    text = re.sub(r"(\s\-\s|-$)", "", text)
    text = re.sub(r"[,\!\?\%\(\)\/\"]", "", text)
    text = re.sub(r"\&\S*\s", "", text)
    text = re.sub(r"\&", "", text)
    text = re.sub(r"\+", "", text)
    text = re.sub(r"\#", "", text)
    text = re.sub(r"\$", "", text)
    text = re.sub(r"\£", "", text)
    text = re.sub(r"\%", "", text)
    text = re.sub(r"\:", "", text)
    text = re.sub(r"\@", "", text)
    text = re.sub(r"\-", "", text)

    return text


def text_processing(tweet):
    """Une autre méthode de prétraitement de text en utilisant les fonctions nltk"""
    
    #Generating the list of words in the tweet (hastags and other punctuations removed)
    def form_sentence(tweet):
        tweet_blob = TextBlob(tweet)
        return ' '.join(tweet_blob.words)
    new_tweet = form_sentence(tweet)
    
    #Removing stopwords and words with unusual symbols
    def no_user_alpha(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('french')]
        return clean_mess
    no_punc_tweet = no_user_alpha(new_tweet)
    
    #Normalizing the words in tweets 
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet
    
    
    return normalization(no_punc_tweet)

def create_tweets_df(tweets, polarite):
    tuple_ligne = list(zip(tweets,polarite))
    dataFrame = pd.DataFrame(tuple_ligne, columns= ['Tweets','Label'])
    print("Dataset de travail créé")
    return dataFrame

def polarite_textuelle(polarite_chiffree):
    polarite_data = []
    for polarite in polarite_chiffree:
        if polarite in (0,0.5): 
            text ="Neutre"
            polarite_data.append(text)
        if polarite <0: 
            text ="Negatif"
            polarite_data.append(text)
        if polarite >0.5: 
            text ="Positif"
            polarite_data.append(text)
    print("Définition des polarités textuelles faites")
    return polarite_data


def polarite_en_chiffre_Entier(polarite_chiffree):
    polarite_data = []
    for polarite in polarite_chiffree:
        if polarite in (0,0.5): 
            text = 0
            polarite_data.append(text)
        if polarite <0: 
            text = -1
            polarite_data.append(text)
        if polarite >0.5: 
            text = 1
            polarite_data.append(text)
    return polarite_data

def formation_Train_Test_data(data):
    train_set = data.sample(frac=0.66, replace=True, random_state=1)
    test_set = data.sample(frac=0.33, replace=True, random_state=1)
    return train_set, test_set

def calcul_polarite(corpus_clean):
    polarity = []
    for tweet in corpus_clean:
      polarity.append(TextBlob(tweet,pos_tagger=PatternTagger(),analyzer=PatternAnalyzer()).sentiment[0])
      
    return polarity

def exploration_dataSet_1(dataSet):
    """Quelques techniques d'exploration de données
    Celle ci pour voir si la taille du tweet influence sur le label"""
    dataSet['length'] = dataSet['Tweets'].apply(len)
    fig1 = sns.barplot('Label','length',data = dataSet,palette='PRGn')
    plt.title('Average Word Length vs label')
    plot = fig1.get_figure()
    plot.savefig('Barplot.png')
    
def exploration_dataSet_2(dataSet):
    """Celle ci pour voir la répartition du dataSet suivant les labels"""
    fig2 = sns.countplot(x= 'Label',data = dataSet)
    plt.title('Decompte par Label')
    plot = fig2.get_figure()
    plot.savefig('Count Plot.png')
    

    

            
def main():
    #print(get_tweets()[1])
    
    
    """
    #A decommenter si vous avez disposez des données sur votre environnement 
    #Chargement fichier de données test
    tweet = pd.read_csv("tweet_covid.csv")
    
    #recuperation de la colonne des tweets
    corpus = tweet['tweet']"""
    
    """Procédure de récupération en ligne avec l'api TWITTER 
    voir méthode get_tweets() 
    Vous avez juste à définir
        - votre requête, Exemple :         requete ="---OR----"
        - la date de début d'extraction,   dateD = 'AAAA-MM-JJ'
        - la langue,                       langue='fr'
        - et la limite de données          limite=1000
    """
    
   
    corpus = get_tweets()
    #corpus = get_tweets(requete,dateD, langue, limite) #Méthode get_tweets à modifier 
    print("Récupération des tweets OK")
     
    """Reformaatage des données """
    ligne = list(corpus)
    dataFrame = pd.DataFrame(ligne, columns= ['Tweets'])
    corpus = dataFrame['Tweets']
    
    
    #Epuration du champ des tweets pour y elever les chiffres ou consort
    #corpus_clean = corpus.apply(nlp_pipeline)
    corpus_clean = corpus.apply(nlp_pipeline)
    print("Fin Epuration")
    
    #Calcul de la  polarité pour déterminer si un tweet est Positi, Negatif ou neutre
    polarity = calcul_polarite(corpus_clean)
    print("Calcul des polarité OK")
    
    plt.plot(polarity)    
    polarity =polarite_en_chiffre_Entier(polarity)
    dataFrame_Tweets = create_tweets_df(corpus_clean,  polarity)
    
    train_set, test_set = formation_Train_Test_data(dataFrame_Tweets)
    
    #print(train_set)
    #Machine Learning Pipeline
    from time import time
    t0 = time()
    pipeline = Pipeline([
        ('bow',CountVectorizer(analyzer=text_processing)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])
    pipeline.fit(train_set['Tweets'],train_set['Label'])
    
    predictions = pipeline.predict(test_set['Tweets'])
    tt = time() - t0
    #print(file)
    print ("Temps d'exécution"+" {} ".format(round(tt,3)))
    print(classification_report(predictions,test_set['Label']))
    print ('\n Matrice de confusion')
    print(confusion_matrix(predictions,test_set['Label']))
    #print(accuracy_score(predictions,test_set['Label']))
    
    
    
    #model=MultinomialNB()
    #model.fit(train_set,train_set)
    #dataFrame_Tweets.to_csv("donnees.csv")
    #print(dataFrame_Tweets)
    
    
    
    
    #print(train_set)
    
    
    
    #dataFrame_Tweets.to_csv("donnees_traitees.csv")
    #print(dataFrame_Tweets)
    
    #plt.plot(polarity)  
    #print(polarity)
    
    
    
    
    
    """
    #Technique d'exploration des données
    group = lambda liste, size : [liste[i:i+size] for i in range(0, len(liste), size)]
    
    polarity_par_paquet = group(polarity,100)
    
    liste_moyennes = []
    for l in polarity_par_paquet :
      liste_moyennes.append(np.mean(l))
    
    plt.plot(liste_moyennes)
    
    """





if __name__ == "__main__":
    main()

