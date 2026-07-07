import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
import warnings
import time

warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Анализ МРП ESP",
    page_icon="📊",
    layout="wide"
)

# Заголовок
st.title("📊 Прогнозирование межремонтного периода ЭЦН")
st.markdown("---")

# Боковая панель для управления
with st.sidebar:
    st.header("⚙️ Управление")

    # Загрузка данных
    st.subheader("1. Загрузка данных")
    use_github = st.checkbox("Загрузить с GitHub", value=True)

    uploaded_file = None
    if not use_github:
        st.markdown("**Или загрузите свой файл:**")
        st.caption("Файл должен называться: **Dataset_for_ESP.xlsx**")
        uploaded_file = st.file_uploader("Загрузите файл Excel", type=["xlsx"])

    st.markdown("---")

    # Параметры модели
    st.subheader("2. Параметры модели")
    test_size = st.slider("Размер тестовой выборки", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", value=42, step=1)

    st.markdown("---")

    # Кнопка запуска
    run_analysis = st.button("🚀 Запустить анализ", type="primary", use_container_width=True)


# Основная логика приложения
@st.cache_data
def load_data(use_github, uploaded_file):
    """Загрузка данных"""
    if use_github:
        url = 'https://raw.githubusercontent.com/TimurQuich/Machine-learning-for-determining-the-ESP-overhaul-period/main/Dataset_for_ESP.xlsx'
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
    """Проверка наличия всех необходимых колонок в датасете"""
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


@st.cache_data
def preprocess_data(df):
    """Предобработка данных"""
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


@st.cache_data
def train_models(X_train, X_test, y_train, y_test):
    """Обучение моделей"""
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


@st.cache_data
def optimize_rf_grid(X_train, X_test, y_train, y_test):
    """Оптимизация Random Forest с помощью GridSearchCV"""
    rf_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

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

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    pred = best_model.predict(X_test)

    return {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_,
        'mae': mean_absolute_error(y_test, pred),
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'model': best_model,
        'cv_results': grid_search.cv_results_
    }


@st.cache_data
def optimize_rf_random(X_train, X_test, y_train, y_test):
    """Оптимизация Random Forest с помощью RandomizedSearchCV"""
    rf_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        warm_start=True
    )

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

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    pred = best_model.predict(X_test)

    return {
        'best_params': random_search.best_params_,
        'best_score': -random_search.best_score_,
        'mae': mean_absolute_error(y_test, pred),
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'model': best_model
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


@st.cache_data
def optimize_rf_optuna(X_train, X_test, y_train, y_test):
    """Оптимизация Random Forest с помощью Optuna"""

    def optuna_objective(trial):
        return objective(trial, X_train, X_test, y_train, y_test)

    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_objective, n_trials=30, show_progress_bar=False)

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
        'model': best_model
    }


def create_feature_importance_plot(model, feature_names):
    """Создание графика важности признаков"""
    importances = model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    features_df = features_df.sort_values('Importance', ascending=False)

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
    """Создание графика факт vs предсказание"""
    fig, ax = plt.subplots(figsize=(10, 8))

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Идеальное предсказание', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.6, color='steelblue')
    ax.set_xlabel("Реальные значения МРП", fontsize=14)
    ax.set_ylabel("Предсказанные значения МРП", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_dataset_template():
    """Создание шаблона датасета"""
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


def create_heatmap_with_highlight(df_for_corr):
    """Создание квадратной тепловой карты с выделенным МРП (полная матрица)"""
    # Перемещаем колонку МРП в начало
    if 'МРП' in df_for_corr.columns:
        cols = ['МРП'] + [col for col in df_for_corr.columns if col != 'МРП']
        df_for_corr = df_for_corr[cols]

    corr = df_for_corr.corr()

    # Создаем квадратный figure
    n_features = len(corr.columns)
    fig_size = max(10, n_features * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Рисуем полную тепловую карту (без маски)
    sns.heatmap(corr,
                cmap="coolwarm",
                center=0,
                annot=True,
                fmt='.2f',
                square=True,  # Квадратные ячейки
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
        # Для оси X (колонки)
        for label in ax.get_xticklabels():
            if label.get_text() == 'МРП':
                label.set_weight('bold')
                label.set_size(12)
                label.set_color('darkred')

        # Для оси Y (строки)
        for label in ax.get_yticklabels():
            if label.get_text() == 'МРП':
                label.set_weight('bold')
                label.set_size(12)
                label.set_color('darkred')

    # Добавляем заголовок
    ax.set_title('Корреляционная матрица (МРП выделен жирным)',
                 fontsize=14, fontweight='bold', pad=20)

    # Настройка подписей
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Делаем квадратные оси
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


# Основная логика
if run_analysis:
    # Загрузка данных
    df = load_data(use_github, uploaded_file)

    if df is not None:
        # Проверка структуры данных
        is_valid, missing_columns = validate_dataframe(df)

        if not is_valid:
            st.error(f"❌ Ошибка: В датасете отсутствуют следующие обязательные колонки: {', '.join(missing_columns)}")
            st.info("📋 Пожалуйста, убедитесь, что ваш файл содержит все необходимые колонки с точными названиями.")

            with st.expander("📋 Посмотреть пример правильного формата данных", expanded=True):
                st.subheader("Образец датасета")
                st.markdown("""
                Ваш файл должен содержать следующие колонки с **точными названиями**:
                - **Модель насоса** - тип установки
                - **Глубина спуска, м** - глубина установки насоса
                - **% в** - обводненность продукции
                - **МРП** - межремонтный период (целевая переменная)
                - **Qж факт** - фактический дебит жидкости
                - **Qн факт** - фактический дебит нефти
                - **Причина остановки** - причина выхода из строя
                """)

                template_df = create_dataset_template()
                st.dataframe(template_df, use_container_width=True)

                csv = template_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать шаблон датасета (CSV)",
                    data=csv,
                    file_name='Dataset_for_ESP_template.csv',
                    mime='text/csv',
                    use_container_width=True
                )

                st.info(
                    "💡 **Совет:** Скопируйте структуру шаблона и замените данные на свои. Названия колонок должны совпадать **точно**!")

            st.stop()

        st.success("✅ Данные успешно загружены и прошли проверку!")

        with st.spinner("🔄 Выполняется анализ данных..."):
            # Предобработка
            df_encoded, df_fe = preprocess_data(df)

            # Подготовка данных
            X = df_encoded.drop("МРП", axis=1)
            y = df_encoded["МРП"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # --- Вкладки ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Данные",
                "📈 Распределение МРП",
                "🤖 Сравнение моделей",
                "⚡ Оптимизация",
                "📉 Анализ важности признаков"
            ])

            # Вкладка 1: Данные
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Количество записей", len(df))
                    st.metric("Количество признаков (до кодирования)", df_fe.shape[1])
                with col2:
                    st.metric("Количество признаков (после кодирования)", X.shape[1])
                    st.metric("Размер обучающей выборки", X_train.shape[0])
                    st.metric("Размер тестовой выборки", X_test.shape[0])

                st.subheader("Первые 5 строк данных")
                st.dataframe(df.head())

                st.subheader("Статистика числовых признаков")
                st.dataframe(df.describe())

            # Вкладка 2: Распределение МРП
            with tab2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(y, bins=30, kde=True, color='steelblue')
                ax.set_title('Распределение МРП (гистограмма)')
                ax.set_xlabel('МРП')
                ax.set_ylabel('Частота')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

                st.subheader("Статистики МРП")
                stats = {
                    "Среднее": y.mean(),
                    "Медиана": y.median(),
                    "Стандартное отклонение": y.std(),
                    "Асимметрия": y.skew(),
                    "Эксцесс": y.kurtosis(),
                    "25% квартиль": y.quantile(0.25),
                    "75% квартиль": y.quantile(0.75)
                }
                cols = st.columns(4)
                for i, (key, value) in enumerate(stats.items()):
                    cols[i % 4].metric(key, f"{value:.2f}")

            # Вкладка 3: Сравнение моделей
            with tab3:
                st.subheader("Обучение и сравнение моделей")

                with st.spinner("🔄 Обучение моделей..."):
                    results, trained_models = train_models(X_train, X_test, y_train, y_test)

                results_df = pd.DataFrame(results).T


                def highlight_min(s, props=''):
                    return np.where(s == np.min(s), props, '')


                styled_df = results_df.style.apply(
                    highlight_min,
                    props='background-color: #006400; color: white',
                    subset=['MAE']
                ).apply(
                    highlight_min,
                    props='background-color: #006400; color: white',
                    subset=['RMSE']
                )
                st.dataframe(styled_df)

                fig, ax = plt.subplots(figsize=(10, 6))
                results_df.plot(kind='bar', ax=ax)
                plt.title('Сравнение моделей по MAE и RMSE')
                plt.ylabel('Значение метрики')
                plt.xlabel('Модель')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Метрика')
                plt.tight_layout()
                st.pyplot(fig)

                best_mae = min(results, key=lambda k: results[k]['MAE'])
                best_rmse = min(results, key=lambda k: results[k]['RMSE'])

                col1, col2 = st.columns(2)
                col1.metric("Лучшая модель по MAE", best_mae, f"MAE = {results[best_mae]['MAE']:.2f}")
                col2.metric("Лучшая модель по RMSE", best_rmse, f"RMSE = {results[best_rmse]['RMSE']:.2f}")

                st.subheader(f"Факт vs Предсказание для лучшей модели ({best_mae})")
                best_model = trained_models[best_mae]
                pred = best_model.predict(X_test)
                fig = create_scatter_plot(y_test, pred, f"Лучшая модель: {best_mae}")
                st.pyplot(fig)

            # Вкладка 4: Оптимизация
            with tab4:
                st.subheader("Оптимизация гиперпараметров Random Forest")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("### GridSearchCV")
                    with st.spinner("🔄 Оптимизация через GridSearchCV..."):
                        start_time = time.time()
                        rf_grid_result = optimize_rf_grid(X_train, X_test, y_train, y_test)
                        grid_time = time.time() - start_time

                    st.write("**Лучшие параметры:**")
                    for key, value in rf_grid_result['best_params'].items():
                        st.write(f"- {key}: {value}")
                    st.metric("Лучшая MAE (кросс-валидация)", f"{rf_grid_result['best_score']:.2f}")
                    st.metric("MAE на тестовом наборе", f"{rf_grid_result['mae']:.2f}")
                    st.metric("RMSE на тестовом наборе", f"{rf_grid_result['rmse']:.2f}")
                    st.metric("Время выполнения", f"{grid_time:.1f} сек")

                with col2:
                    st.markdown("### RandomizedSearchCV")
                    with st.spinner("🔄 Оптимизация через RandomizedSearchCV..."):
                        start_time = time.time()
                        rf_rand_result = optimize_rf_random(X_train, X_test, y_train, y_test)
                        rand_time = time.time() - start_time

                    st.write("**Лучшие параметры:**")
                    for key, value in rf_rand_result['best_params'].items():
                        st.write(f"- {key}: {value}")
                    st.metric("Лучшая MAE (кросс-валидация)", f"{rf_rand_result['best_score']:.2f}")
                    st.metric("MAE на тестовом наборе", f"{rf_rand_result['mae']:.2f}")
                    st.metric("RMSE на тестовом наборе", f"{rf_rand_result['rmse']:.2f}")
                    st.metric("Время выполнения", f"{rand_time:.1f} сек")

                with col3:
                    st.markdown("### Optuna")
                    with st.spinner("🔄 Оптимизация через Optuna..."):
                        start_time = time.time()
                        rf_optuna_result = optimize_rf_optuna(X_train, X_test, y_train, y_test)
                        optuna_time = time.time() - start_time

                    st.write("**Лучшие параметры:**")
                    for key, value in rf_optuna_result['best_params'].items():
                        st.write(f"- {key}: {value}")
                    st.metric("Лучшая MAE (Optuna)", f"{rf_optuna_result['best_value']:.2f}")
                    st.metric("MAE на тестовом наборе", f"{rf_optuna_result['mae']:.2f}")
                    st.metric("RMSE на тестовом наборе", f"{rf_optuna_result['rmse']:.2f}")
                    st.metric("Время выполнения", f"{optuna_time:.1f} сек")

                st.subheader("Сравнение результатов оптимизации")

                original_rf = results['Random Forest']

                optimization_results = {
                    'Original RF': {'MAE': original_rf['MAE'], 'RMSE': original_rf['RMSE'], 'Time': 0},
                    'GridSearchCV': {'MAE': rf_grid_result['mae'], 'RMSE': rf_grid_result['rmse'], 'Time': grid_time},
                    'RandomizedSearchCV': {'MAE': rf_rand_result['mae'], 'RMSE': rf_rand_result['rmse'],
                                           'Time': rand_time},
                    'Optuna': {'MAE': rf_optuna_result['mae'], 'RMSE': rf_optuna_result['rmse'], 'Time': optuna_time}
                }

                opt_results_df = pd.DataFrame(optimization_results).T

                opt_styled = opt_results_df.style.apply(
                    highlight_min,
                    props='background-color: #006400; color: white',
                    subset=['MAE']
                ).apply(
                    highlight_min,
                    props='background-color: #006400; color: white',
                    subset=['RMSE']
                )
                st.dataframe(opt_styled)

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                metrics_df = opt_results_df[['MAE', 'RMSE']]
                metrics_df.plot(kind='bar', ax=axes[0])
                axes[0].set_title('Сравнение метрик')
                axes[0].set_ylabel('Значение метрики')
                axes[0].set_xlabel('Метод')
                axes[0].tick_params(axis='x', rotation=45)
                axes[0].legend(title='Метрика')

                time_df = opt_results_df[['Time']]
                time_df.plot(kind='bar', ax=axes[1], color='orange')
                axes[1].set_title('Время выполнения')
                axes[1].set_ylabel('Время (сек)')
                axes[1].set_xlabel('Метод')
                axes[1].tick_params(axis='x', rotation=45)
                axes[1].legend(['Time'])

                plt.tight_layout()
                st.pyplot(fig)

                st.subheader("Графики факт vs предсказание для оптимизированных моделей")

                tabs = st.tabs(["GridSearchCV", "RandomizedSearchCV", "Optuna"])

                with tabs[0]:
                    pred_grid = rf_grid_result['model'].predict(X_test)
                    fig = create_scatter_plot(y_test, pred_grid, "GridSearchCV")
                    st.pyplot(fig)

                with tabs[1]:
                    pred_rand = rf_rand_result['model'].predict(X_test)
                    fig = create_scatter_plot(y_test, pred_rand, "RandomizedSearchCV")
                    st.pyplot(fig)

                with tabs[2]:
                    pred_optuna = rf_optuna_result['model'].predict(X_test)
                    fig = create_scatter_plot(y_test, pred_optuna, "Optuna")
                    st.pyplot(fig)

            # Вкладка 5: Важность признаков
            with tab5:
                st.subheader("Анализ важности признаков")

                rf_model = trained_models['Random Forest']

                fig = create_feature_importance_plot(rf_model, X.columns)
                st.pyplot(fig)

                importances = rf_model.feature_importances_
                features_df = pd.DataFrame({
                    'Признак': X.columns,
                    'Важность': importances
                })
                features_df = features_df.sort_values('Важность', ascending=False)

                features_df_numeric = features_df[
                    ~features_df['Признак'].str.contains('Модель насоса|Причина остановки', regex=True)]

                st.subheader("Топ-10 числовых признаков по важности")
                st.dataframe(features_df_numeric.head(10).style.bar(subset=['Важность'], color='steelblue'))

                st.subheader("Корреляционная матрица")

                df_for_corr = df_fe.copy()
                df_for_corr = df_for_corr.select_dtypes(include=[np.number])

                fig = create_heatmap_with_highlight(df_for_corr)
                st.pyplot(fig)

    else:
        st.error("❌ Не удалось загрузить данные. Проверьте источник данных.")

else:
    # Начальное состояние
    st.info("👈 Настройте параметры в боковой панели и нажмите 'Запустить анализ'")

    st.subheader("📋 Описание датасета")
    st.markdown("""
    Датасет содержит данные по эксплуатации электроцентробежных насосов (ЭЦН) и включает следующие параметры:

    - **Модель насоса** - тип установки
    - **Глубина спуска, м** - глубина установки насоса
    - **% в** - обводненность продукции
    - **МРП** - межремонтный период (целевая переменная)
    - **Qж факт** - фактический дебит жидкости
    - **Qн факт** - фактический дебит нефти
    - **Причина остановки** - причина выхода из строя

    Приложение выполняет:
    1. Предобработку данных и создание новых признаков
    2. Визуализацию распределения целевой переменной
    3. Сравнение различных моделей машинного обучения
    4. Оптимизацию гиперпараметров Random Forest (GridSearchCV, RandomizedSearchCV, Optuna)
    5. Анализ важности признаков
    """)

    with st.expander("📋 Посмотреть пример формата данных", expanded=True):
        st.subheader("Образец датасета (Dataset_for_ESP.xlsx)")
        st.markdown("""
        **Важно:** Ваш файл должен называться **Dataset_for_ESP.xlsx** и содержать следующие колонки с **точными названиями**:
        """)

        template_df = create_dataset_template()
        st.dataframe(template_df, use_container_width=True)

        st.info("""
        **📌 Требования к данным:**
        - Имя файла: **Dataset_for_ESP.xlsx** (обязательно)
        - Все колонки должны иметь **точно такие же названия**
        - Колонка **МРП** - это целевая переменная (что мы прогнозируем)
        - Числовые значения должны быть в формате чисел (не текст)
        - Датасет не должен содержать пустых значений
        """)

        csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Скачать шаблон датасета (CSV)",
            data=csv,
            file_name='Dataset_for_ESP_template.csv',
            mime='text/csv',
            use_container_width=True
        )

    st.subheader("📊 Предварительный просмотр данных")
    try:
        url = 'https://raw.githubusercontent.com/TimurQuich/Machine-learning-for-determining-the-ESP-overhaul-period/main/Dataset_for_ESP.xlsx'
        preview_df = pd.read_excel(url)
        st.dataframe(preview_df.head(10))
    except:
        st.warning("Не удалось загрузить данные для предпросмотра")

# Footer
st.markdown("---")
st.caption("📌 Приложение для прогнозирования межремонтного периода ЭЦН")