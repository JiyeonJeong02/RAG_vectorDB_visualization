import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorboard.plugins import projector



(train_data, test_data), info = tfds.load(
        "imdb_reviews/subwords8k",
        split=(tfds.Split.TRAIN, tfds.Split.TEST),
        with_info=True,
        as_supervised=True,
    )
encoder = info.features["text"].encoder

# shuffle and pad the data.
train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes=((None,), ()))
test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes=((None,), ()))
train_batch, train_labels = next(iter(train_batches))


# Create an embedding layer
embedding_dim = 16
embedding = tf.keras.layers.Embedding(encoder.vocab_size, embedding_dim)
# Train this embedding as part of a keras model
model = tf.keras.Sequential(
    [
        embedding, # The embedding layer should be the first layer in a model.
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

# Compile model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
    )

# Train model
history = model.fit(
    train_batches, epochs=1, validation_data=test_batches, validation_steps=20
    )


log_dir = '/logs/imdb-example/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

with open(os.path.join(log_dir, 'metadata.tsv'), "w", encoding='utf-8') as f:
    for subwords in encoder.subwords:
        f.write("{}\n".format(subwords))

weights = tf.Variable(model.layers[0].get_weights()[0][1:])
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)


# 데이터셋 로드
(train_data, test_data), info = tfds.load("imdb_reviews/subwords8k", split=['train', 'test'], as_supervised=True, with_info=True)

train_batches = train_data.shuffle(1000).padded_batch(32)
test_batches = test_data.shuffle(1000).padded_batch(32)

# 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=info.features['text'].encoder.vocab_size, output_dim=64),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# TensorBoard 콜백 설정
log_dir = "logs/imdb-example/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 모델 학습
model.fit(train_batches, epochs=5, validation_data=test_batches, callbacks=[tensorboard_callback])

log_dir = "C:/Users/wjd_w/logs/imdb-example/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_batches, epochs=5, validation_data=test_batches, callbacks=[tensorboard_callback])
