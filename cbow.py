import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
torch.manual_seed(1)


class CBOW(nn.Module):
    """
    Continuous bag of words.
    Model selects an embedded vector v from the lookup table
    model =

    v * w1 + b
    """
    def __init__(self, embedding_size, corpus):
        super(CBOW, self).__init__()

        vocabulary = np.unique(np.array(corpus))
        vocabulary_size = vocabulary.shape[0]

        # word lookup table. Every row is an index of the vocabulary containing an embedded vector.
        self.v_embedding = nn.Embedding(vocabulary_size, embedding_size)
        # Output layer.
        self.linear = nn.Linear(embedding_size, vocabulary_size)
        self.vocabulary_index = dict(zip(vocabulary, range(len(vocabulary))))

    def forward(self, x):
        idx = []
        for input_words in x:
            idx.append([self.vocabulary_index[w] for w in input_words])
        idx = torch.LongTensor(idx)
        linear_in = self.v_embedding(autograd.Variable(idx)).mean(dim=1)
        return self.linear(linear_in)

    def det_row(self, words):
        return autograd.Variable(
            torch.LongTensor([self.vocabulary_index[w] for w in words]))

    def train_model(self, batch_size, X, Y, epochs=100):
        iterations = X.shape[0] // batch_size
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

        for epoch in range(epochs):

            c = 0
            for i in range(iterations):
                x = X[c: c + batch_size]
                y = self.det_row(Y[c: c + batch_size])
                c += batch_size

                y_pred = self.forward(x)

                optimizer.zero_grad()
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

            if epoch % 15:
                print(loss.data[0])


if __name__ == '__main__':
    CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
    raw_text = """We are about to study the idea of a computational process. Computational processes are abstract
    beings that inhabit computers. As they evolve, processes manipulate other abstract
    things called data. The evolution of a process is directed by a pattern of rules
    called a program. People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells.""".lower().split()
    word_to_ix = {word: i for i, word in enumerate(set(raw_text))}
    X = []
    Y = []
    for i in range(2, len(raw_text) - 2):
        context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]
        X.append(context)
        Y.append(target)

    X = np.array(X)
    Y = np.array(Y)

    model = CBOW(embedding_size=10,
                 corpus=raw_text)

    model.train_model(batch_size=10,
                      X=X,
                      Y=Y,
                      epochs=500)




