# Part of speech tagging using LSTM

This repo contains example code for tagging a sentence using LSTM.
First, you train a model using [Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus).
Once the model is trained, you can enter an English sentence to predict the tag for each word.
If the word is not recognized, a special tag <UNK> (meaning unknown) is output.

## How to train
### Installation
Have a look at the requirements.txt and verify that you have these packages.  If not, install them using pip.

### Run the training script.
Run:
pos_lstm.py

You may need to add a Python path to the project directory if your IDE does not set it automatically.

## How to predict
Run
```console
pos_lstm_predict.py
```

Again, you may need to add a Python path to the project directory if your IDE does not set it automatically.

Once the application starts, it asks you to enter a sentence:

Enter a sentence (press 'q' to quit):

Enter a sample sentence:
I saw a big I saw a big building in the city.
and hit the ENTER key.

Enter a sentence (press 'q' to quit): I saw a big building in the city.<br>

It will display the tokenized sentence:
INFO:__main__:Tokens:\['I', 'saw', 'a', 'big', 'building', 'in', 'the', 'city', '.'\]<br>

Then outputs the POS tag for each word:
\[0\] word: I tag: PPSS<br>
\[1\] word: saw tag: VBD<br>
\[2\] word: a tag: AT<br>
\[3\] word: big tag: JJ<br>
\[4\] word: building tag: NN<br>
\[5\] word: in tag: IN<br>
\[6\] word: the tag: AT<br>
\[7\] word: city tag: NN<br>
\[8\] word: . tag: .<br>

## Expected loss after 20 epochs
Here is the loss and accuracy data in my environment after training for 20 epochs:<br>
 loss: 0.0057 - categorical_accuracy: 0.9983 - val_loss: 0.0210 - val_categorical_accuracy: 0.9950

# Credit
1. I used \[1\] below as a reference for the model.<br>
2. I used \[2\] as a reference for loading Brown corpus including conversion between word and tag to IDs.

# References
\[1\] Brownlee, Jason, "How to Develop a Bidirectional LSTM For Sequence Classification in Python with Keras,"
 https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/, 2017.<br>
\[2\] Chainer team, postagging.py, https://github.com/chainer/chainer/blob/master/examples/pos/postagging.py.
