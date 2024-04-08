import emoji
import numpy as np
from tensorflow import keras  # Using tensorflow.keras instead of 'keras'
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Emoji dictionary with more descriptive variable names
EMOJI_MAP = {
    "0": emoji.emojize("\u2764\uFE0F"),  # Heart
    "1": emoji.emojize(":baseball:"),    # Baseball
    "2": emoji.emojize(":grinning_face_with_big_eyes:"),  # Grinning Face with Big Eyes
    "3": emoji.emojize(":disappointed_face:"),        # Disappointed Face
    "4": emoji.emojize(":fork_and_knife:"),         # Fork and Knife
}


def load_data(train_file, test_file):
    """Loads training and testing data from CSV files."""
    train_data = pd.read_csv(train_file, header=None)
    test_data = pd.read_csv(test_file, header=None)
    return train_data, test_data


def create_embedding_matrix(glove_file, max_len, embedding_dim):
    """
    Creates an embedding matrix from a GloVe text file.

    Args:
        glove_file (str): Path to the GloVe text file.
        max_len (int): Maximum length of a sentence.
        embedding_dim (int): Dimensionality of word vectors.

    Returns:
        numpy.ndarray: Embedding matrix.
    """
    embeddings_index = {}
    with open(glove_file, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0].lower()
            coeff = np.asarray(values[1:], dtype="float")
            embeddings_index[word] = coeff

    # Initialize embedding matrix with zeros
    embedding_matrix = np.zeros((len(EMOJI_MAP) + 1, embedding_dim))
    for word, vector in embeddings_index.items():
        if word in tokenizer.word_index:
            index = tokenizer.word_index[word]
            embedding_matrix[index] = vector

    return embedding_matrix


def preprocess_data(data, max_len):
    """
    Preprocesses text data by converting to sequences and padding.

    Args:
        data (pd.DataFrame): Dataframe containing text and labels.
        max_len (int): Maximum length of a sentence.

    Returns:
        tuple: Tuple containing preprocessed sequences and labels.
    """
    sentences = data.iloc[:, 0].tolist()
    labels = data.iloc[:, 1].tolist()

    tokenizer = Tokenizer(num_words=len(EMOJI_MAP) + 1)  # +1 for padding
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    return padded_sequences, np.array(labels)


def build_model(max_len, embedding_dim, num_lstm_units, num_emoji_classes):
    """
    Builds a recurrent neural network model for emoji prediction.

    Args:
        max_len (int): Maximum length of a sentence.
        embedding_dim (int): Dimensionality of word vectors.
        num_lstm_units (int): Number of units in the LSTM layers.
        num_emoji_classes (int): Number of emoji classes.

    Returns:
        keras.models.Sequential: Compiled model.
    """
    model = Sequential()
    model.add(Embedding(len(EMOJI_MAP) + 1, embedding_dim, input_length=max_len))
    for _ in range(2):  # Two LSTM layers
        model.add(LSTM(num_lstm_units, return_sequences=True))
        model.add(Dropout(0.5))
    model.add(LSTM(num_lstm_units))
    model.add(Dropout(0.5))  # Dropout layer with 50% dropout rate
    model.add(Dense(num_emoji_classes, activation="softmax"))  # Output layer with softmax activation
    # Compile the model
    model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

    return model
