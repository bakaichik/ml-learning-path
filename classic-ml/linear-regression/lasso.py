import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
# Не забудьте импортировать все необходимые инструменты

X = np.genfromtxt('https://code.s3.yandex.net/Math/datasets/X.csv', delimiter=',')
y = np.genfromtxt('https://code.s3.yandex.net/Math/datasets/y.csv', delimiter=',')
X_test = np.genfromtxt('https://code.s3.yandex.net/Math/datasets/X_test.csv', delimiter=',')
y_test = np.genfromtxt('https://code.s3.yandex.net/Math/datasets/y_test.csv', delimiter=',')


alpha = 1
ridge_min = []
while alpha < 100:
    # Ridge
    print(f'----------- {alpha} ---------------')
    print('Ridge')
    model_ridge = Ridge(alpha=alpha) # Выбираем модель
    model_ridge.fit(X, y) # Коэффициенты регрессии подбираются под данные
    print(f"alpha: {alpha}. Ошибка модели на новых данных: {mean_squared_error(model_ridge.predict(X_test), y_test)}")
    print(f"Коэффициенты модели: {model_ridge.coef_}")
    ridge_min.append(mean_squared_error(model_ridge.predict(X_test), y_test))
    print()
    alpha = alpha + 1
    print('')

    # Lasso
    print('Lasso')
    model_lasso = Lasso(alpha=alpha) # Выбираем модель
    model_lasso.fit(X, y) # Коэффициенты регрессии подбираются под данные
    print(f"alpha: {alpha}. Ошибка модели на новых данных: {mean_squared_error(model_lasso.predict(X_test), y_test)}")
    print(f"Коэффициенты модели: {model_lasso.coef_}")
    print()
    alpha = alpha + 1

print(min(ridge_min))