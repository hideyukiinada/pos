# Part of speech tagging using LSTM

This repo contains example code for tagging a sentence using LSTM.
First, you train a model using [Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus).
Once the model is trained, you can enter an English sentence to predict the tag for each word.
If the word is not recognized, a special tag <UNK> (meaning unknown) is output.

## How to train
Run pos_lstm.py

## How to predict

Enter a sentence (press 'q' to quit): I saw a big building in the city.
INFO:__main__:Tokens:['I', 'saw', 'a', 'big', 'building', 'in', 'the', 'city', '.']
(256,)
(256, 473)
[0] word: I tag: PPSS
[1] word: saw tag: VBD
[2] word: a tag: AT
[3] word: big tag: JJ
[4] word: building tag: NN
[5] word: in tag: IN
[6] word: the tag: AT
[7] word: city tag: NN
[8] word: . tag: .

## Expected loss after 20 epochs
Here is the loss and accuracy data in my environment after training for 20 epochs:
 loss: 0.0057 - categorical_accuracy: 0.9983 - val_loss: 0.0210 - val_categorical_accuracy: 0.9950

# Credit
1. I used [1] below as a reference for the model.
2. I used [2] as a reference for loading Brown corpus including conversion between word and tag to IDs.

# References
[1] Brownlee, Jason, "How to Develop a Bidirectional LSTM For Sequence Classification in Python with Keras,"
 https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/, 2017.
[2] Chainer team, postagging.py, https://github.com/chainer/chainer/blob/master/examples/pos/postagging.py.
