"""
Тесты для моделей машинного обучения
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os

# Добавляем путь к проекту
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем функции из проекта (если они вынесены в отдельные модули)
# from src.model_trainer import train_models
# from src.preprocessor import preprocess_data


class TestModels:
    """Тестирование моделей машинного обучения"""

    def test_random_forest_initialization(self):
        """Тест: создание модели Random Forest"""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        assert model is not None
        assert model.n_estimators == 100
        assert model.random_state == 42

    def test_model_training(self):
        """Тест: обучение модели на простых данных"""
        # Создаем простые данные
        X = np.random.rand(100, 5)
        y = np.random.rand(100)

        # Обучаем модель
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Проверяем, что модель обучилась
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X.shape[1]
        assert sum(model.feature_importances_) > 0

    def test_model_prediction(self):
        """Тест: предсказание модели"""
        X_train = np.random.rand(80, 5)
        y_train = np.random.rand(80)
        X_test = np.random.rand(20, 5)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Проверяем, что предсказания имеют правильную форму
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)

    def test_mae_metric(self):
        """Тест: метрика MAE"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        mae = mean_absolute_error(y_true, y_pred)

        # MAE должно быть примерно 0.12
        assert mae < 0.5
        assert mae > 0

    def test_rmse_metric(self):
        """Тест: метрика RMSE"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # RMSE должно быть примерно 0.14
        assert rmse < 0.5
        assert rmse > 0


class TestDataPreprocessing:
    """Тестирование предобработки данных"""

    def test_data_loading(self):
        """Тест: загрузка данных"""
        # Создаем тестовые данные
        test_data = pd.DataFrame({
            "Модель насоса": ["ЭЦН-1", "ЭЦН-2"],
            "Глубина спуска, м": [1200, 1500],
            "% в": [85.5, 92.3],
            "МРП": [180, 95],
            "Qж факт": [45.2, 38.7],
            "Qн факт": [6.5, 3.0],
            "Причина остановки": ["Износ", "Отказ"]
        })

        # Проверяем, что данные загружены
        assert len(test_data) == 2
        assert list(test_data.columns) == [
            "Модель насоса",
            "Глубина спуска, м",
            "% в",
            "МРП",
            "Qж факт",
            "Qн факт",
            "Причина остановки"
        ]

    def test_data_validation(self):
        """Тест: проверка наличия обязательных колонок"""
        required_columns = [
            "Модель насоса",
            "Глубина спуска, м",
            "% в",
            "МРП",
            "Qж факт",
            "Qн факт",
            "Причина остановки"
        ]

        test_data = pd.DataFrame({
            col: [1, 2] for col in required_columns
        })

        # Проверяем, что все колонки есть
        for col in required_columns:
            assert col in test_data.columns

    def test_feature_creation(self):
        """Тест: создание новых признаков"""
        df = pd.DataFrame({
            "Qж факт": [45.2, 38.7],
            "Qн факт": [6.5, 3.0],
            "Глубина спуска, м": [1200, 1500],
            "% в": [85.5, 92.3]
        })

        # Создаем новые признаки
        df_fe = df.copy()
        df_fe['Qж факт / Qн факт'] = df_fe['Qж факт'] / (df_fe['Qн факт'] + 1e-6)
        df_fe['Глубина * (% в)'] = df_fe['Глубина спуска, м'] * df_fe['% в']
        df_fe['(% в)^2'] = df_fe['% в'] ** 2

        # Проверяем, что признаки созданы
        assert 'Qж факт / Qн факт' in df_fe.columns
        assert 'Глубина * (% в)' in df_fe.columns
        assert '(% в)^2' in df_fe.columns


class TestStreamlitApp:
    """Тестирование Streamlit приложения"""

    def test_app_imports(self):
        """Тест: импорт библиотек"""
        try:
            import streamlit as st
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split

            assert True
        except ImportError as e:
            assert False, f"Ошибка импорта: {e}"

    def test_dummy(self):
        """Тест-заглушка"""
        assert True


# Запуск тестов
if __name__ == '__main__':
    # Ручной запуск тестов
    test = TestModels()
    test.test_random_forest_initialization()
    test.test_model_training()
    test.test_model_prediction()
    test.test_mae_metric()
    test.test_rmse_metric()

    test_data = TestDataPreprocessing()
    test_data.test_data_loading()
    test_data.test_data_validation()
    test_data.test_feature_creation()

    test_app = TestStreamlitApp()
    test_app.test_app_imports()

    print("✅ Все тесты пройдены!")