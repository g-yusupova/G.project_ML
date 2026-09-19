
# AI or Human essay detector

Детекция AI-сгенерированного текста. Финальный проект курса ML in Data Science. Задача: бинарная классификация — определить, написана ли студенческая работа человеком или сгенерирована/дописана с помощью AI.


## Датасет

AI vs Human Academic Writing Dataset (Kaggle) — 10 200 строк, 20 колонок. Содержит поведенческую телеметрию (скорость печати, время редактирования, число правок) и лингвистические характеристики текста (богатство словаря, пассивный залог, читаемость и др.). Таргет: Is_AI_Assisted (0 — человек, 1 — AI).

## Как запустить

1. Установка зависимостей

pip install -r requirements.txt

2. Ноутбук (EDA + обучение моделей)

Открой notebooks/analysis.ipynb в Jupyter и прогони все ячейки по порядку (Kernel → Restart & Run All).

3. Веб-интерфейс для теста модели

uvicorn app:app --reload

Открой в браузере:

http://127.0.0.1:8000/ - для ручного теста
http://127.0.0.1:8000/docs - Swagger UI (интерактивная документация API)
# Структура проекта

my-project/
│
├── notebooks/
│   └── analysis.ipynb        # EDA + модели + эксперименты
│
├── app.py                     
├── model.pkl                  # Сохранённая модель (joblib)
├── requirements.txt           # Зависимости
├── README.md                  # Описание проекта
│
├── data/
│   ├── raw/
            ai_writing_detection_dataset.csv   # Исходные данные
│   └── processed/ 
            processed_data.csv         # Обработанные данные
│
└── screenshots/               # Скриншоты для README
    └── demo.png## API Reference

Принимает данные одной студенческой работы, возвращает предсказания обеих моделей (двухмодельной системы).

**Пример запроса:**
```json
{
  "Submission_Type": "Essay",
  "Academic_Level": "Undergraduate",
  "Primary_Language": "Native",
  "Word_Count": 1000,
  "Average_Sentence_Length": 17,
  "Grammar_Errors": 5,
  "Vocabulary_Richness": 0.6,
  "Passive_Voice_Ratio": 0.2,
  "Reading_Level": 10,
  "Editing_Time": 60,
  "Revision_Count": 10,
  "Typing_Speed": 45,
  "Plagiarism_Score": 5,
  "Citation_Count": 3,
  "Punctuation_Density": 0.08,
  "Sentence_Complexity": 5,
  "Font_Family": "Arial",
  "Submission_Hour": 14
}
```

**Пример ответа:**
```json
{
  "full_model_prediction": "Human-written",
  "full_model_probability": 0.03,
  "linguistic_model_prediction": "Human-written",
  "disagreement": false,
  "needs_manual_review": false
}
```

| Поле ответа | Описание |
|---|---|
| `full_model_prediction` | Вердикт модели, обученной на всех фичах (включая телеметрию) |
| `full_model_probability` | Вероятность класса "AI-assisted" по полной модели |
| `linguistic_model_prediction` | Вердикт модели, обученной только на лингвистических фичах |
| `disagreement` | Разошлись ли предсказания двух моделей |
| `needs_manual_review` | Флаг: требуется ручная проверка (равен `disagreement`) |

### GET /docs
Интерактивная Swagger-документация (доступна только при запущенном сервере: `uvicorn app:app --reload`).


## EDA

10 200 строк, из которых 200 — полные дубликаты (удалены → 10 000 строк).

Дисбаланс таргета: 85.5% человеческих работ / 14.5% AI-ассистированных.

Пропуски в 5 колонках (8–15% строк) — не связаны с таргетом, импутированы медианой.

Категориальные фичи (Submission_Type, Academic_Level, Primary_Language, Font_Family) практически не различают классы и не несут предсказательной силы.

Сильнее всего с таргетом связаны поведенческие фичи: Revision_Count (r = -0.87), Typing_Speed (r = +0.81), Editing_Time (r = -0.54) — у AI-сдач почти нет правок и аномально высокая скорость ввода текста (фактически признак вставки текста, а не набора).
Лингвистические фичи (Vocabulary_Richness, Passive_Voice_Ratio, Grammar_Errors и др.) коррелируют умеренно (r ≈ 0.4–0.6), но стабильно.


## Модель

Выбранная финальная модель - RandomForest.

Сделана двухмодельная система с флагом расхождения.

Проблема: сильнейшие признаки (Typing_Speed, Revision_Count, Editing_Time) — это поведенческие признаки, а не характеристики самого текста. Если способ набора текста нетипичен для конкретного студента (например, текст написан от руки/в блокноте, а затем скопирован и вставлен), эти признаки могут ошибочно указывать на AI, что рискует ложным обвинением.

Решение: обучены две модели:

Полная модель — все признаки, включая телеметрию.
Лингвистическая модель — только признаки текста, без Typing_Speed, Revision_Count, Editing_Time.

Если предсказания расходятся — работа помечается как требующая ручной проверки, а не автоматического вердикта.


## Стек

Python, pandas, numpy, scikit-learn, XGBoost, FastAPI, matplotlib/seaborn.