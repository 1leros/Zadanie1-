# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 3. Предварительная обработка и визуализация данных

# Загрузка данных
data = pd.read_csv('flights_data.csv')

# Просмотр первых строк данных
print(data.head())

# Кодирование аэропортов
label_encoder = LabelEncoder()
data['Departure Airport'] = label_encoder.fit_transform(data['Departure Airport'])
data['Destination Airport'] = label_encoder.fit_transform(data['Destination Airport'])

# Преобразование времени в datetime
data['Scheduled departure time'] = pd.to_datetime(data['Scheduled departure time'])
data['Scheduled arrival time'] = pd.to_datetime(data['Scheduled arrival time'])

# Извлечение новых признаков
data['departure_hour'] = data['Scheduled departure time'].dt.hour
data['departure_day'] = data['Scheduled departure time'].dt.dayofweek
data['flight_duration'] = (data['Scheduled arrival time'] - data['Scheduled departure time']).dt.total_seconds() / 60

# Разделение на обучающую и тестовую выборки
train_data = data[data['Scheduled departure time'].dt.year < 2018]
test_data = data[data['Scheduled departure time'].dt.year == 2018]

# 4. Обнаружение и удаление выбросов

Q1 = train_data['Delay'].quantile(0.25)
Q3 = train_data['Delay'].quantile(0.75)
IQR = Q3 - Q1

# Удаление выбросов
train_data = train_data[(train_data['Delay'] >= Q1 - 1.5 * IQR) & (train_data['Delay'] <= Q3 + 1.5 * IQR)]

# Визуализация задержек
plt.figure(figsize=(10, 6))
sns.boxplot(x=train_data['Delay'])
plt.title('Boxplot of Flight Delays (After Outlier Removal)')
plt.show()

# 5. Модели машинного обучения

# Подготовка данных для моделей
X = train_data[['Departure Airport', 'Destination Airport', 'departure_hour', 'departure_day', 'flight_duration']]
y = train_data['Delay']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Линейная регрессия
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Полиномиальная регрессия
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Регрессия с регуляризацией (Ridge)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Оценка производительности моделей
results = {
    "Linear Regression": {
        "MSE": mean_squared_error(y_test, y_pred_linear),
        "R²": r2_score(y_test, y_pred_linear)
    },
    "Polynomial Regression": {
        "MSE": mean_squared_error(y_test, y_pred_poly),
        "R²": r2_score(y_test, y_pred_poly)
    },
    "Ridge Regression": {
        "MSE": mean_squared_error(y_test, y_pred_ridge),
        "R²": r2_score(y_test, y_pred_ridge)
    }
}

# Вывод результатов
for model, metrics in results.items():
    print(f"{model} - MSE: {metrics['MSE']:.2f}, R²: {metrics['R²']:.2f}")
