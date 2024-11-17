import numpy as np

# Загружаем матрицу встречаемости слов.
# allow_pickle=True сообщает NumPy, что файл сохранён в специальном формате pickle.
url = "https://code.s3.yandex.net/Math/datasets/kinopoisk_term_occurence.npy"
filename = np.DataSource().open(url).name
X = np.load(filename, allow_pickle=True)

# Загружаем слова, соответствующие строкам матрицы.
url = "https://code.s3.yandex.net/Math/datasets/kinopoisk_words.npy"
filename = np.DataSource().open(url).name
words = np.load(filename, allow_pickle=True)

U, s, Vt = np.linalg.svd(X, full_matrices=False)

# Количество используемых сингулярных векторов.
k = 50

# Считаем эмбеддинги слов с помощью матрицы U и s:
word_vectors = (U @ np.diag(s))[:, :k]

search_word = 'сердце'

# Индекс искомого слова.
search_index = np.where(words == search_word)[0]

# Эмбеддинг искомого слова.
search_vector = word_vectors[search_index]

# Возвращает топ-n похожих по L2 слов.
def find_nearest_word(word_vector, n=3):
    # L2 расстояние между выбранным вектором и всеми остальными векторами.
    distances = np.mean((word_vectors - word_vector) ** 2, axis=1)

		# Слова по индексам векторов с наименьшим расстоянием.
    return words[np.argsort(distances)[:n]]

print(f"Слова, похожие на {search_word}:")
print(find_nearest_word(search_vector, n=10))