"""
Главное приложение Streamlit для прогнозирования МРП ЭЦН
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import time

# Импорт модулей из src
from src.data_loader import load_data, validate_dataframe, create_dataset_template
from src.preprocessor import preprocess_data, get_features_and_target
from src.model_trainer import (
    train_models,
    optimize_rf_grid,
    optimize_rf_random,
    optimize_rf_optuna,
    split_data
)
from src.visualizer import (
    create_feature_importance_plot,
    create_scatter_plot,
    create_heatmap_with_highlight,
    plot_mrp_distribution,
    plot_grid_search_results
)

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


# Функция для выделения минимального значения (темно-зеленый)
def highlight_min(s, props=''):
    return np.where(s == np.min(s), props, '')


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

                st.info("💡 **Совет:** Скопируйте структуру шаблона и замените данные на свои. Названия колонок должны совпадать **точно**!")

            st.stop()

        st.success("✅ Данные успешно загружены и прошли проверку!")

        with st.spinner("🔄 Выполняется анализ данных..."):
            # Предобработка
            df_encoded, df_fe = preprocess_data(df)

            # Подготовка данных
            X, y = get_features_and_target(df_encoded)

            X_train, X_test, y_train, y_test = split_data(
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
                fig = plot_mrp_distribution(y)
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
                        rf_grid_result = optimize_rf_grid(X_train, X_test, y_train, y_test)

                    st.write("**Лучшие параметры:**")
                    for key, value in rf_grid_result['best_params'].items():
                        st.write(f"- {key}: {value}")
                    st.metric("Лучшая MAE (кросс-валидация)", f"{rf_grid_result['best_score']:.2f}")
                    st.metric("MAE на тестовом наборе", f"{rf_grid_result['mae']:.2f}")
                    st.metric("RMSE на тестовом наборе", f"{rf_grid_result['rmse']:.2f}")
                    st.metric("Время выполнения", f"{rf_grid_result['time']:.1f} сек")

                with col2:
                    st.markdown("### RandomizedSearchCV")
                    with st.spinner("🔄 Оптимизация через RandomizedSearchCV..."):
                        rf_rand_result = optimize_rf_random(X_train, X_test, y_train, y_test)

                    st.write("**Лучшие параметры:**")
                    for key, value in rf_rand_result['best_params'].items():
                        st.write(f"- {key}: {value}")
                    st.metric("Лучшая MAE (кросс-валидация)", f"{rf_rand_result['best_score']:.2f}")
                    st.metric("MAE на тестовом наборе", f"{rf_rand_result['mae']:.2f}")
                    st.metric("RMSE на тестовом наборе", f"{rf_rand_result['rmse']:.2f}")
                    st.metric("Время выполнения", f"{rf_rand_result['time']:.1f} сек")

                with col3:
                    st.markdown("### Optuna")
                    with st.spinner("🔄 Оптимизация через Optuna..."):
                        rf_optuna_result = optimize_rf_optuna(X_train, X_test, y_train, y_test)

                    st.write("**Лучшие параметры:**")
                    for key, value in rf_optuna_result['best_params'].items():
                        st.write(f"- {key}: {value}")
                    st.metric("Лучшая MAE (Optuna)", f"{rf_optuna_result['best_value']:.2f}")
                    st.metric("MAE на тестовом наборе", f"{rf_optuna_result['mae']:.2f}")
                    st.metric("RMSE на тестовом наборе", f"{rf_optuna_result['rmse']:.2f}")
                    st.metric("Время выполнения", f"{rf_optuna_result['time']:.1f} сек")

                # Сравнение оптимизации
                st.subheader("Сравнение результатов оптимизации")

                original_rf = results['Random Forest']

                optimization_results = {
                    'Original RF': {'MAE': original_rf['MAE'], 'RMSE': original_rf['RMSE'], 'Time': 0},
                    'GridSearchCV': {'MAE': rf_grid_result['mae'], 'RMSE': rf_grid_result['rmse'], 'Time': rf_grid_result['time']},
                    'RandomizedSearchCV': {'MAE': rf_rand_result['mae'], 'RMSE': rf_rand_result['rmse'], 'Time': rf_rand_result['time']},
                    'Optuna': {'MAE': rf_optuna_result['mae'], 'RMSE': rf_optuna_result['rmse'], 'Time': rf_optuna_result['time']}
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

                # Графики факт vs предсказание для оптимизированных моделей
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
        url = 'https://raw.githubusercontent.com/TimurQuich/ESP-MRP-Prediction/main/data/Dataset_for_ESP.xlsx'
        preview_df = pd.read_excel(url)
        st.dataframe(preview_df.head(10))
    except:
        st.warning("Не удалось загрузить данные для предпросмотра")

# Footer
st.markdown("---")
st.caption("📌 Приложение для прогнозирования межремонтного периода ЭЦН")