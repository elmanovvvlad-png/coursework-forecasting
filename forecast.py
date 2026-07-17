import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Данные о производстве мяса (тыс. тонн)
data = [205, 227, 242, 254, 306]
years = pd.to_datetime(['2017', '2018', '2019', '2020', '2021'])
time_series = pd.Series(data, index=years)

print("Временной ряд:")
print(time_series)

# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(time_series)
plt.title('Производство мяса крупного рогатого скота')
plt.xlabel('Год')
plt.ylabel('Тыс. тонн')
plt.grid(True)
plt.show()

# Модель Хольта-Уинтерса (метод Холта без сезонности, т.к. тренд линейный)
model = ExponentialSmoothing(
    time_series,
    trend='add',
    seasonal=None,
    seasonal_periods=0
)
fitted_model = model.fit()

print("Сводка по модели:")
print(fitted_model.summary())

# Прогноз на 3 года вперёд
forecast = fitted_model.forecast(3)
print("\nПрогноз на следующие 3 года:")
print(forecast)

# Визуализация с прогнозом
plt.figure(figsize=(10, 6))
plt.plot(time_series, label='Исторические данные')
plt.plot(forecast, label='Прогноз', color='red', linestyle='--')
plt.title('Прогноз производства мяса')
plt.xlabel('Год')
plt.ylabel('Тыс. тонн')
plt.legend()
plt.grid(True)
plt.show()
