# This python file allows to preprocess datasets

from preprocessing import *

print("Preprocessing Negative Tweets")
new_negative=open("../data/train_neg_full.txt",'rb')
negative_tweets=[]
i=0
#Adjust the size of the preprocessed dataset from 1 to 1250000:
size=100
for line in new_negative:
    if i<size:
        line = line.decode('utf8')
        #Apply all preprocessing functions from preprocessing file
        #Filter digits
        line=filter_digits(line)
        #Remove Stop words
        line=remove_stopwords(line)
        #Interpret emojis in Tweet
        line=interpret_emoji(line)
        #Remove punctuation
        line=remove_punctuation(line)
        #Remove words user, url, number
        line=remove_words(line)
        #Remove repeated letters
        line=replace_moreletters(line)
        #Lematize words in the tweet
        line=lemmatize(line)
        #Split number and text in tweets
        line=split_number_text(line)
        #Correct misspell in tweet
        line=correct_misspell(line)
        
        ##Add other preprocessing steps, Refer to python file preprocessing.py to see all preprocessing functions
        
        #Leave only one space between words
        line = re.sub("\s\s+", " ", line)
        i=i+1
        negative_tweets.append(line)
print("DONE..")
print("Saving file with preprocessed Tweets")
f = open("preprocessed/pre_negative.txt", "w")
f.write("\n".join(map(lambda x: str(x), negative_tweets)))
f.close()
    

print("Preprocessing Positive Tweets")
new_positive=open("../data/train_pos_full.txt",'rb')
positive_tweets=[]
i=0
for line in new_positive:
    if i<size:
        line = line.decode('utf8')
        #Apply all preprocessing functions from preprocessing file
        #Filter digits
        line=filter_digits(line)
        #Remove Stop words
        line=remove_stopwords(line)
        #Interpret emojis in Tweet
        line=interpret_emoji(line)
        #Remove punctuation
        line=remove_punctuation(line)
        #Remove words user, url, number
        line=remove_words(line)
        #Remove repeated letters
        line=replace_moreletters(line)
        #Lematize words in the tweet
        line=lemmatize(line)
        #Split number and text in tweets
        line=split_number_text(line)
        #Correct misspell in tweet
        line=correct_misspell(line)
        
        ##Add other preprocessing steps, Refer to python file preprocessing.py to see all preprocessing functions
        
        #Leave only one space between words
        line = re.sub("\s\s+", " ", line)
        i=i+1
        positive_tweets.append(line)
print("DONE..")
print("Saving file with preprocessed Tweets")
f = open("preprocessed/pre_positive.txt", "w")
f.write("\n".join(map(lambda x: str(x), positive_tweets)))
f.close()



print("Preprocessing Test Tweets")
new_test=open("../data/test_data.txt",'rb')
test_tweets=[]
for line in new_test:
    line = line.decode('utf8')
    #Apply all preprocessing functions from preprocessing file
    #Filter digits
    line=filter_digits(line)
    #Remove Stop words
    line=remove_stopwords(line)
    #Interpret emojis in Tweet
    line=interpret_emoji(line)
    #Remove punctuation
    line=remove_punctuation(line)
    #Remove words user, url, number
    line=remove_words(line)
    #Remove repeated letters
    line=replace_moreletters(line)
    #Lematize words in the tweet
    line=lemmatize(line)
    #Split number and text in tweets
    line=split_number_text(line)
    #Correct misspell in tweet
    line=correct_misspell(line)
        
    ##Add other preprocessing steps, Refer to python file preprocessing.py to see all preprocessing functions
        
    #Leave only one space between words
    line = re.sub("\s\s+", " ", line)
    test_tweets.append(line)
print("DONE..")
print("Saving file with preprocessed Tweets")
f = open("preprocessed/pre_test.txt", "w")
f.write("\n".join(map(lambda x: str(x), test_tweets)))
f.close()

print("Finished preprocessing")


