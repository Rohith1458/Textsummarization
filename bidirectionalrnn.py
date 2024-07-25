import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Embedding, Concatenate, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
input_texts = ["This is an example sentence.", "Another example sentence."]
target_texts = ["Example summary.", "Another summary."]

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_texts + target_texts)

input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

max_input_length = max(len(seq) for seq in input_sequences)
max_target_length = max(len(seq) for seq in target_sequences)

input_data = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
target_data = pad_sequences(target_sequences, maxlen=max_target_length, padding='post')

vocab_size = len(tokenizer.word_index) + 1

# Build the model
embedding_dim = 256
latent_dim = 512

# Encoder
encoder_inputs = Input(shape=(max_input_length,))
encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_inputs)
encoder_lstm = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Attention Mechanism
attention_dense = Dense(1, activation='tanh')
attention_scores = attention_dense(decoder_outputs)
attention_weights = tf.nn.softmax(attention_scores, axis=1)

context_vector = attention_weights * encoder_outputs
context_vector = tf.reduce_sum(context_vector, axis=1)

decoder_concat_input = Concatenate(axis=-1)([context_vector, decoder_outputs])
decoder_dense = TimeDistributed(Dense(vocab_size, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()

# Prepare target data for training
target_data = np.expand_dims(target_data, -1)

# Train the model
model.fit([input_data, target_data[:, :-1]], target_data[:, 1:], epochs=10, batch_size=32)
