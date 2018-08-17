import csv
import random
import pandas as pd
import numpy as np
import sklearn

with open("Sentiment Analysis Dataset.csv","rb") as source:
    rdr= csv.reader(source)
    with open("twitter.csv","wb") as result:
        wtr= csv.writer( result )
        iterrdr = iter(rdr)
        next(iterrdr)
        for r in iterrdr:
            wtr.writerow( (r[0], r[3], r[1]) )

temp = pd.read_csv("twitter.csv", header=None)
temp = sklearn.utils.shuffle(temp)

train = temp[0:2400]
train.to_csv('twitter_train.txt', header=None, index=None, sep=' ')
test = temp[2400:3000]
test.to_csv('twitter_test.txt', header=None, index=None, sep=' ')

