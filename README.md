# Part of speech tagging using LSTM

This repo contains example code for tagging words in a sentence with corresponding [part of speech](https://en.wikipedia.org/wiki/Part_of_speech) (or POS) using LSTM.  For example, parts of speech for a sentence *Dogs are awesome* is a noun (for *Dogs*), a verb (for *are*), an adjective (for *awesome*). a period (for the final .).<br>
Before you can predict POS for your sentence, you need to train a model using [Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus).  The script automatically downloads the corpus using the NLTK package.
Once the model is trained, you can enter an English sentence to predict the tag for each word.
If the word is not recognized, a special tag <UNK> (meaning unknown) is output.

## How to train
### Installation
Have a look at the requirements.txt and verify that you have these packages.  If not, install them using pip.

### Use of NVIDIA GPU
Use of NVIDIA GPU is strongly recommended.  
If you don't have NVIDIA GPU, change the model_architecture.py to use LSTM instead of CuDNNLSTM layer. 
I haven't tried LSTM layer, and theoretiacally it should work, though it will be much slower than CuDNNLSTM.
I put a simple performance comparison in notes.txt.

### How to configure
Configurable parameters are specified in config.py.
By default, the model contains the embedding layer.  If you want to try out the impact of not having an embedding layer, you can set use_embedding = False.

### Run the training script.
Run:
```console
pos_lstm.py
```

You may need to add a Python path to the project directory if your IDE does not set it automatically.

## How to predict
Run
```console
pos_lstm_predict.py
```

Again, you may need to add a Python path to the project directory if your IDE does not set it automatically.

Once the application starts, it asks you to enter a sentence:

```console
Enter a sentence (press 'q' to quit):
```

Enter a sample sentence, *I saw a big I saw a big building in the city.* and hit the ENTER key.

```console
Enter a sentence (press 'q' to quit): I saw a big building in the city.
```

It will display the tokenized sentence:
```console
INFO:__main__:Tokens:['I', 'saw', 'a', 'big', 'building', 'in', 'the', 'city', '.']
```

Then outputs the POS tag for each word:
```console
[0] word: I tag: PPSS
[1] word: saw tag: VBD
[2] word: a tag: AT
[3] word: big tag: JJ
[4] word: building tag: NN
[5] word: in tag: IN
[6] word: the tag: AT
[7] word: city tag: NN
[8] word: . tag: .
```

The above Wikipedia page article has a table that shows the mapping between abbreviations and tags. For example, per the table "PPSS" means "other nominative personal pronoun (I, we, they, you)."

## Expected loss after 20 epochs
Here is the loss and accuracy data in my environment after training for 20 epochs:<br>
```console
 loss: 0.0057 - categorical_accuracy: 0.9983 - val_loss: 0.0210 - val_categorical_accuracy: 0.9950
```

# Limitations
If you enter a word that is not in the vocabulary in the Brown corpus, the prediction script maps to the Unknown word token and tries to predict.
The accuracy of the word is not correct.  For example if you change the sample sentence to:
```console
Jim saw a big building in the city.
```
Predictor correctly predicts that the word Jim is a proper noun (NP).  However, if you change the sentence to:
```console
Aimee saw a big building in the city.
```
the word Aimee is not in Brown Corus, and gets tagged as a noun occurring in the headline (NN-HL).

# Credit
1. I used \[1\] below as a reference for the model.<br>
2. I used \[2\] as a reference for loading Brown corpus including conversion between word and tag to IDs.

# References
\[1\] Brownlee, Jason, "How to Develop a Bidirectional LSTM For Sequence Classification in Python with Keras,"
 https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/, 2017.<br>
\[2\] Chainer team, postagging.py, https://github.com/chainer/chainer/blob/master/examples/pos/postagging.py.
