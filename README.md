# SentimentAnalysis
Sentiment classifier built on data set consisting of 3,000 product reviews. Project for COS 424.

Abstract
----------------------------------------------
Sentiment analysis of this data has a wide range of applications. We build various sentiment classifiers beginning with multinomial and Bernoulli Naive Bayes classifiers that classify a text as either positive or negative. We evaluate these methods on 2,400 (training) and 600 (testing) labeled sentences, processed into bag-of-words representations of each post. We found that logistic regression had the best performance amongst our models, while other more complex techniques worsened performance.

Deployment
----------------------------------------------
This directory comprises the following files

NEW/MODIFIED FILES:

- naiveBayes.py, implements the multinomial and binomial Naive Bayes classifiers on
the training data and uses testing data to calculate metrics.

- preprocessSentences.py, (edited from the originally provided version), called as follows: 
	python ./preprocessSentences.py -p <data> -i <in> -o <out> -v <vocab>
<data> is the path of the directory containing train.txt,
<in> is an optional argument specifying the input file (default is "train.txt"),
<out> is an optional argument specifying the prefix of output files, 
<vocab> is an optional argument specifying the path to an existing 
vocabulary file. 
(SAMPLE SEQUENCE OF COMMANDS:
python ./preprocessSentences.py -p . -i train.txt -o train
python ./preprocessSentences.py -p . -i test.txt -o test -v train_vocab_5.txt)

- preprocessSentencesBinary.py, is called in the same manner and produces binary
values of bag of words representation instead of frequency.

The script generates four output files in <data>: 
	- A vocabulary file 'out_vocab_*.txt’ (if one is not specified 
when calling the script) with tokenized words from the training data 
(where '*' is the word count threshold, set to default as 5 in the 
script), 
	- A list of training samples numbers in 
'out_samples_classes_*.txt',
	- A list of sentiment labels corresponding to each training 
sample, 'out_classes_*.txt', and
	- A 'bag of words' featurized representation of each sentence in 
the training set, 'out_bag_of_words_*.csv' (either binary or frequency)

- preprocessTwitter.py, shuffles twitter.csv and randomly selects 2400 training and 
600 testing data, converts into .txt file to be preprocessed as described above.

- Classifier.ipynb, built off of the existing preprocessing files, generates statistics and ROC curves for the remaining models (logistic regression, GloVe + regression, bigram, monogram + bigram). To execute, open the Jupyter notebook and run all cells. 


PROVIDED DEPENDENCIES:

- train.txt, a training set of 2400 sentiment labelled sentences 
(from Amazon, Yelp and iMDB) in the following form:
<sample num>	<sentence> 	<sentiment> 
where <sentiment> is either 0 (negative) or 1 (positive).

- test.txt, a test set of 600 labelled sentences, in the same format.


ADDITIONAL DEPENDENCIES:

- glove.twitter.27B.200d.txt, a 200-dimensional vector representation of words trained on data from twitter, which can be found in “glove.twitter.27B.zip” here: https://nlp.stanford.edu/projects/glove/

- twitter.csv, contains 1,578,627 tweets, each classified as 1 for positive sentiment and 0 for negative sentiment, found here: http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/. Needs to be renamed to “twitter.csv” after downloading.

Authors
----------------------------------------------
- Peter Chen
- Austin Williams
