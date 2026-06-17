import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense


# Step 1: Read the text file
with open("data.txt", "r", encoding="utf-8") as file:
    text = file.read()

lines = text.lower().split("\n")


# Step 2: Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)

vocabulary_size = len(tokenizer.word_index) + 1


# Step 3: Create subsequences
sequences = tokenizer.texts_to_sequences(lines)

subsequences = []

for sequence in sequences:
    for i in range(1, len(sequence)):
        subsequence = sequence[:i + 1]
        subsequences.append(subsequence)


# Step 4: Pad sequences
sequence_length = max(len(sequence) for sequence in subsequences)

sequences = pad_sequences(
    subsequences,
    maxlen=sequence_length,
    padding="pre"
)


# Step 5: Split input and output
x = sequences[:, :-1]
y = sequences[:, -1]

y = to_categorical(y, num_classes=vocabulary_size)


# Step 6: Build the LSTM RNN model
def build_model():
    model = Sequential()

    model.add(
        Embedding(
            input_dim=vocabulary_size,
            output_dim=100,
            input_length=sequence_length - 1
        )
    )

    model.add(LSTM(100))
    model.add(Dropout(0.1))

    model.add(
        Dense(
            vocabulary_size,
            activation="softmax"
        )
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


model = build_model()


# Step 7: Train the model
history = model.fit(
    x,
    y,
    epochs=500,
    verbose=1
)


# Step 8: Generate new text
def generate_text(seed_text, next_words=20):
    result = seed_text

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences(
            [token_list],
            maxlen=sequence_length - 1,
            padding="pre"
        )

        predicted_probabilities = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted_probabilities)

        predicted_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                predicted_word = word
                break

        result += " " + predicted_word

    return result


print("\nGenerated Text:")
print(generate_text("from fairest creatures", next_words=20))


# The trained model is returned here if this code is used inside a notebook
model
