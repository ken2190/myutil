from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode, udf, lit

spark = SparkSession.builder.appName('read JSON files').getOrCreate()

json_df=spark.read.json("hdfs://localhost:8020/user/cloudera/twitter_Extract")

# show the schema
json_df.printSchema()

# show the schema for tweets
json_df.select('data').printSchema()

# convert array to dict
data_df=json_df.select('data').withColumn('data', explode('data').alias('data'))

# print the schema after conversion
data_df.printSchema()

# print the schema after conversion
data_df.printSchema()

# Unravel the nested json
import pyspark.sql.types as T
from pyspark.sql.functions import col

def read_nested_json(df):
    column_list = []

    for column_name in df.schema.names:
        print("Outside isinstance loop: " + column_name)
        # Checking column type is ArrayType
        if isinstance(df.schema[column_name].dataType, T.ArrayType):
            print("Inside isinstance loop of ArrayType: " + column_name)
            df = df.withColumn(column_name, explode(column_name).alias(column_name))
            column_list.append(column_name)

        elif isinstance(df.schema[column_name].dataType, T.StructType):
            print("Inside isinstance loop of StructType: " + column_name)
            for field in df.schema[column_name].dataType.fields:
                column_list.append(col(column_name + "." + field.name).alias(column_name + "_" + field.name))
        else:
            column_list.append(column_name)

def flatten_nested_json(df):
    read_nested_json_flag = True
    while read_nested_json_flag:
        print("Reading Nested JSON File ... ")
        df = read_nested_json(df)
        #df.show(100, False)
        read_nested_json_flag = False

        for column_name in df.schema.names:
            if isinstance(df.schema[column_name].dataType, T.ArrayType):
              read_nested_json_flag = True
            elif isinstance(df.schema[column_name].dataType, T.StructType):
              read_nested_json_flag = True
    return df
    
# flatten the tweet content
data_df=flatten_nested_json(data_df)

# Preprocess steps
import re
from pyspark.sql.functions import udf
from pyspark.sql.functions import to_timestamp
import pyspark.sql.types as T

# convert create_at column to date
raw_tweets=raw_tweets.withColumn("created_date", raw_tweets['created_at'].cast(T.DateType()))

# define process function
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@â'
def remove_links(tweet):
    tweet = re.sub(r'http\S+', '', tweet) 
    tweet = re.sub(r'bit.ly/\S+', '', tweet) 
    tweet = tweet.strip('[link]') 
    return tweet
def remove_users(tweet):
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) 
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) 
    return tweet
def remove_punctuation(tweet):
    tweet = re.sub('['+my_punctuation + ']+', ' ', tweet) 
    return tweet
def remove_number(tweet):
    tweet = re.sub('([0-9]+)', '', tweet) 
    return tweet
def remove_hashtag(tweet):
    tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) 
    return tweet
    
# register user defined function
remove_links=udf(remove_links)
remove_users=udf(remove_users)
remove_punctuation=udf(remove_punctuation)
remove_number=udf(remove_number)
remove_hashtag=udf(remove_hashtag)

raw_tweets=raw_tweets.withColumn('processed_text', remove_links(raw_tweets['text']))
raw_tweets=raw_tweets.withColumn('processed_text', remove_users(raw_tweets['processed_text']))
raw_tweets=raw_tweets.withColumn('processed_text', remove_punctuation(raw_tweets['processed_text']))
raw_tweets=raw_tweets.withColumn('processed_text', remove_number(raw_tweets['processed_text']))

# show the data before and after pocessed
raw_tweets.select('text','processed_text').show(10, False)

# Create a tokenizer that Filter away tokens with length < 3, and get rid of symbols like $,#,...
tokenizer = RegexTokenizer().setPattern("[\\W_]+").setMinTokenLength(3).setInputCol("processed_text").setOutputCol("tokens")

# Tokenize tweets
tokenized_tweets = tokenizer.transform(raw_tweets)

# perform lemmatization using wordNetLemmatizer
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
#from nltk.stem.lancaster import LancasterStemmer
#from nltk import SnowballStemmer
#from nltk.stem import PorterStemmer

lemmatizer = WordNetLemmatizer()
#st = LancasterStemmer()
#stemmer = SnowballStemmer("english")
#stemmer=PorterStemmer()

def lemmatization(row):
    #row = [stemmer.stem(lemmatizer.lemmatize(word)) for word in row]
    row = [lemmatizer.lemmatize(word,'v') for word in row]
    return row

lemmatization = udf(lemmatization)
tokenized_tweets=tokenized_tweets.withColumn('tokens_lemma', lemmatization(tokenized_tweets['tokens']))

# create cutomized extended stop word list
stopwordList = ["singapore","Singapore"]
StopWordsRemover().getStopWords()
stopwordList.extend(StopWordsRemover().getStopWords())
stopwordList = list(set(stopwordList))

# reference: https://sites.google.com/site/iamgongwei/home/sw
# this is a twitter specific stop word summarized by SMU researcher
# to add more common stopwords,especially in twitter context
f_twitter_stop_words = open('twitter-stopwords - TA - Less.txt','r')
f_twitter_stop_words_content = f_twitter_stop_words.read()
#print(content)
twitter_stopwords = f_twitter_stop_words_content.split(",")
twitter_stopwords