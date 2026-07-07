"""
Модуль визуализации
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def create_feature_importance_plot(model, feature_names):
    """
    Создание графика важности признаков

    Args:
        model: Обученная модель с feature_importances_
        feature_names: Список названий признаков

    Returns:
        matplotlib.figure.Figure: График важности признаков
    """
    importances = model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    features_df = features_df.sort_values('Importance', ascending=False)

    # Фильтруем числовые признаки
    features_df_numeric = features_df[
        ~features_df['Feature'].str.contains('Модель насоса|Причина остановки', regex=True)]

    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = features_df_numeric.head(10)
    colors = sns.color_palette('viridis', n_colors=len(top_features))
    ax.barh(top_features['Feature'], top_features['Importance'], height=0.8, color=colors)
    ax.set_xlabel('Важность', fontsize=14)
    ax.set_ylabel('Признак', fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def create_scatter_plot(y_true, y_pred, title):
    """
    Создание графика факт vs предсказание

    Args:
        y_true: Реальные значения
        y_pred: Предсказанные значения
        title: Заголовок графика

    Returns:
        matplotlib.figure.Figure: График факт vs предсказание
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--',
            label='Идеальное предсказание', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.6, color='steelblue')
    ax.set_xlabel("Реальные значения МРП", fontsize=14)
    ax.set_ylabel("Предсказанные значения МРП", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_heatmap_with_highlight(df_for_corr):
    """
    Создание квадратной тепловой карты с выделенным МРП

    Args:
        df_for_corr: DataFrame с числовыми признаками

    Returns:
        matplotlib.figure.Figure: Тепловая карта корреляций
    """
    # Перемещаем колонку МРП в начало
    if 'МРП' in df_for_corr.columns:
        cols = ['МРП'] + [col for col in df_for_corr.columns if col != 'МРП']
        df_for_corr = df_for_corr[cols]

    corr = df_for_corr.corr()

    # Создаем квадратный figure
    n_features = len(corr.columns)
    fig_size = max(10, n_features * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Рисуем полную тепловую карту
    sns.heatmap(corr,
                cmap="coolwarm",
                center=0,
                annot=True,
                fmt='.2f',
                square=True,
                linewidths=0.5,
                ax=ax,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 10})

    # Выделяем строку и колонку МРП жирным шрифтом
    if 'МРП' in corr.columns:
        mrp_idx = corr.columns.get_loc('МРП')

        # Выделяем текст в ячейках строки МРП
        for text in ax.texts:
            if text.get_position()[1] == mrp_idx + 0.5:
                text.set_weight('bold')
                text.set_size(11)

        # Выделяем текст в ячейках колонки МРП
        for text in ax.texts:
            if text.get_position()[0] == mrp_idx + 0.5:
                text.set_weight('bold')
                text.set_size(11)

        # Выделяем подписи для МРП
        for label in ax.get_xticklabels():
            if label.get_text() == 'МРП':
                label.set_weight('bold')
                label.set_size(12)
                label.set_color('darkred')

        for label in ax.get_yticklabels():
            if label.get_text() == 'МРП':
                label.set_weight('bold')
                label.set_size(12)
                label.set_color('darkred')

    ax.set_title('Корреляционная матрица (МРП выделен жирным)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def plot_grid_search_results(cv_results):
    """
    Визуализация результатов GridSearchCV

    Args:
        cv_results: Результаты кросс-валидации из GridSearchCV

    Returns:
        matplotlib.figure.Figure: График результатов
    """
    results_df = pd.DataFrame(cv_results)
    param_cols = [col for col in results_df.columns if col.startswith('param_')]
    results_df = results_df.sort_values('mean_test_score', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    top_n = min(10, len(results_df))
    top_results = results_df.head(top_n)

    labels = []
    for _, row in top_results.iterrows():
        label = ""
        for col in param_cols:
            param_name = col.replace('param_', '')
            label += f"{param_name}={row[col]} "
        labels.append(label)

    x = np.arange(len(labels))
    ax.bar(x, -top_results['mean_test_score'].values, color='steelblue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('MAE (меньше лучше)')
    ax.set_title('Топ-10 комбинаций гиперпараметров')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_mrp_distribution(y):
    """
    Создание графика распределения МРП

    Args:
        y: Целевая переменная

    Returns:
        matplotlib.figure.Figure: Гистограмма распределения МРП
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(y, bins=30, kde=True, color='steelblue')
    ax.set_title('Распределение МРП (гистограмма)')
    ax.set_xlabel('МРП')
    ax.set_ylabel('Частота')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig