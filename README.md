Concurrence Layer for Keras
===========================

This layer is a one-shot attention layer to be used with Keras v. 2.0 or later.

It takes the output of an RNN layer with return_sequences=True and creates a single return sequence which is 
the weighted sum of the input return sequences based on the trainable internal weights and the input sequences 
themselves.

A code example which uses this layer is:

```python
model = Sequential()
model.add(Bidirectional(GRU(hidden_size, return_sequences=True), merge_mode='concat', 
                        input_shape=(None, input_size)))
model.add(Concurrence())
model.add(RepeatVector(max_out_seq_len + 1))
model.add(GRU(hidden_size * 2, return_sequences=True))
model.add(TimeDistributed(Dense(output_dim=output_size, activation="softmax")))
model.compile(loss="categorical_crossentropy", optimizer="rms_prop")
```