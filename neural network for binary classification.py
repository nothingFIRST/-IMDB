import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Загрузка набора данных IMDB
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)

# Установка максимального числа слов в отзыве
max_words = 500

# Добавление нулей для выравнивания длины последовательностей
X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_words)
X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_words)

# Определение модели LSTM
model = Sequential([
    Embedding(input_dim=5000, output_dim=32, input_length=max_words),
    LSTM(100),
    Dense(1, activation='sigmoid')
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train_padded, y_train, epochs=5, batch_size=64, validation_data=(X_test_padded, y_test))

# Оценка модели на тестовых данных
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Предсказание тональности новых отзывов
new_reviews = ["The movie was fantastic!", "The plot made no sense."]
new_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    imdb.get_word_index()[tf.keras.datasets.imdb.get_word_index().keys()].tolist(),
    maxlen=max_words
)
new_predictions = model.predict(new_sequences)

# Вывод предсказаний
for review, prediction in zip(new_reviews, new_predictions):
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")

