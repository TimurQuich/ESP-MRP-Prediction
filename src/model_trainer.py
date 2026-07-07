"""
Модуль обучения моделей и оптимизации гиперпараметров
"""
import time
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna


def train_models(X_train, X_test, y_train, y_test):
    """
    Обучение нескольких моделей и сравнение результатов

    Args:
        X_train, X_test, y_train, y_test: Данные для обучения и тестирования

    Returns:
        tuple: (results, trained_models)
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=42)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        results[name] = {'MAE': mae, 'RMSE': rmse}
        trained_models[name] = model

    return results, trained_models


def optimize_rf_grid(X_train, X_test, y_train, y_test):
    """
    Оптимизация Random Forest с помощью GridSearchCV

    Returns:
        dict: Результаты оптимизации
    """
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 0.8],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_depth': [None, 10, 20],
        'bootstrap': [True, False]
    }

    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0,
        return_train_score=False
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    pred = best_model.predict(X_test)

    return {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_,
        'mae': mean_absolute_error(y_test, pred),
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'model': best_model,
        'time': elapsed_time,
        'cv_results': grid_search.cv_results_
    }


def optimize_rf_random(X_train, X_test, y_train, y_test):
    """
    Оптимизация Random Forest с помощью RandomizedSearchCV

    Returns:
        dict: Результаты оптимизации
    """
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1, warm_start=True)

    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_features': ['sqrt', 0.6, 0.8, 1.0],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_depth': [None, 10, 20, 30],
        'bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0,
        random_state=42
    )

    start_time = time.time()
    random_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    best_model = random_search.best_estimator_
    pred = best_model.predict(X_test)

    return {
        'best_params': random_search.best_params_,
        'best_score': -random_search.best_score_,
        'mae': mean_absolute_error(y_test, pred),
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'model': best_model,
        'time': elapsed_time
    }


def objective(trial, X_train, X_test, y_train, y_test):
    """Целевая функция для Optuna"""
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_features = trial.suggest_float('max_features', 0.1, 1.0)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)

    return mae


def optimize_rf_optuna(X_train, X_test, y_train, y_test):
    """
    Оптимизация Random Forest с помощью Optuna

    Returns:
        dict: Результаты оптимизации
    """

    def optuna_objective(trial):
        return objective(trial, X_train, X_test, y_train, y_test)

    start_time = time.time()
    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_objective, n_trials=30, show_progress_bar=False)
    elapsed_time = time.time() - start_time

    best_model = RandomForestRegressor(
        **study.best_params,
        random_state=42,
        n_jobs=-1
    )
    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)

    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'mae': mean_absolute_error(y_test, pred),
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'model': best_model,
        'time': elapsed_time
    }


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Разделение данных на обучающую и тестовую выборки

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)