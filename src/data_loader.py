"""
Модуль загрузки данных
"""
import pandas as pd
import streamlit as st


def load_data(use_github=True, uploaded_file=None):
    """
    Загрузка данных с GitHub или из локального файла

    Args:
        use_github (bool): Загружать с GitHub
        uploaded_file: Загруженный файл

    Returns:
        pd.DataFrame: Загруженные данные
    """
    if use_github:
        url = 'https://raw.githubusercontent.com/TimurQuich/ESP-MRP-Prediction/main/data/Dataset_for_ESP.xlsx'
        try:
            df = pd.read_excel(url)
            st.sidebar.success("✅ Данные загружены с GitHub")
            return df
        except Exception as e:
            st.sidebar.error(f"❌ Ошибка загрузки: {e}")
            return None
    else:
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                st.sidebar.success("✅ Данные загружены из файла")
                return df
            except Exception as e:
                st.sidebar.error(f"❌ Ошибка загрузки: {e}")
                return None
    return None


def validate_dataframe(df):
    """
    Проверка наличия всех необходимых колонок

    Args:
        df (pd.DataFrame): Датасет для проверки

    Returns:
        tuple: (is_valid, missing_columns)
    """
    required_columns = [
        "Модель насоса",
        "Глубина спуска, м",
        "% в",
        "МРП",
        "Qж факт",
        "Qн факт",
        "Причина остановки"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        return False, missing_columns
    return True, []


def create_dataset_template():
    """
    Создание шаблона датасета

    Returns:
        pd.DataFrame: Шаблон с примером данных
    """
    template_data = {
        "Модель насоса": ["ЭЦН-1", "ЭЦН-2", "ЭЦН-3", "ЭЦН-1", "ЭЦН-2"],
        "Глубина спуска, м": [1200, 1500, 980, 2100, 1750],
        "% в": [85.5, 92.3, 78.1, 95.0, 88.7],
        "МРП": [180, 95, 220, 60, 140],
        "Qж факт": [45.2, 38.7, 52.1, 29.8, 41.3],
        "Qн факт": [6.5, 3.0, 11.4, 1.5, 4.7],
        "Причина остановки": ["Износ", "Отказ", "Плановый", "Авария", "Износ"]
    }
    return pd.DataFrame(template_data)