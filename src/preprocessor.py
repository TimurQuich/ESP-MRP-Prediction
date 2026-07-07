"""
Модуль предобработки данных
"""
import pandas as pd
import numpy as np


def preprocess_data(df):
    """
    Предобработка данных: отбор колонок, создание новых признаков, кодирование

    Args:
        df (pd.DataFrame): Исходный датасет

    Returns:
        tuple: (df_encoded, df_fe) - закодированный датасет и датасет с новыми признаками
    """
    # Отбор колонок
    df = df[[
        "Модель насоса",
        "Глубина спуска, м",
        "% в",
        "МРП",
        "Qж факт",
        "Qн факт",
        "Причина остановки"
    ]]
    df = df.dropna()

    # Создание новых признаков
    df_fe = df.copy()
    df_fe['Qж факт / Qн факт'] = df_fe['Qж факт'] / (df_fe['Qн факт'] + 1e-6)
    df_fe['Глубина * (% в)'] = df_fe['Глубина спуска, м'] * df_fe['% в']
    df_fe['(% в) ^ 2'] = df_fe['% в'] ** 2
    df_fe['log_Qж'] = np.log(df_fe['Qж факт'] + 1e-3)
    df_fe['log_Qн'] = np.log(df_fe['Qн факт'] + 1e-3)
    df_fe['Глубина * Qж факт'] = df_fe['Глубина спуска, м'] * df_fe['Qж факт']

    # Кодирование категорий
    df_encoded = pd.get_dummies(df_fe, columns=["Модель насоса", "Причина остановки"])

    return df_encoded, df_fe


def get_features_and_target(df_encoded):
    """
    Разделение на признаки и целевую переменную

    Args:
        df_encoded (pd.DataFrame): Закодированный датасет

    Returns:
        tuple: (X, y) - признаки и целевая переменная
    """
    X = df_encoded.drop("МРП", axis=1)
    y = df_encoded["МРП"]
    return X, y