# Shakespeare RNN Language Model

This mini project trains a recurrent neural network using Shakespeare's first sonnet.  
The goal is to build a simple language model that learns word patterns and predicts the next word in a sequence.

## Project Steps

### Step 1: Read the text
The Shakespeare text is loaded from `data.txt`. The text is converted to lowercase and split into lines.

### Step 2: Tokenization
The text is converted into numerical word sequences using Keras Tokenizer.

### Step 3: Create subsequences
Each line is converted into smaller word sequences so the model can learn how words follow each other.

### Step 4: Padding
All sequences are padded to the same length so they can be used for training.

### Step 5: Encode target labels
The last word in each sequence is used as the target output. The target labels are one-hot encoded.

### Step 6: Build the RNN model
The model uses:
- Embedding layer
- LSTM layer with 100 units
- Dropout layer with 10% dropout
- Dense output layer with softmax activation

### Step 7: Train the model
The model is trained using:
- Optimizer: Adam
- Loss function: categorical crossentropy
- Metric: accuracy
- Epochs: 500

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
