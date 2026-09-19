from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.responses import HTMLResponse

app = FastAPI(title="AI Writing Detection API")

# грузим модели один раз при старте сервера, а не при каждом запросе
models = joblib.load('model.pkl')
full_model = models['full_model']
linguistic_model = models['linguistic_model']

# описываем, какие поля и какого типа ожидаем от пользователя —
# pydantic сам провалидирует входные данные и вернёт понятную ошибку, если что-то не так
class StudentSubmission(BaseModel):
    Submission_Type: str
    Academic_Level: str
    Primary_Language: str
    Word_Count: int
    Average_Sentence_Length: float
    Grammar_Errors: int
    Vocabulary_Richness: float
    Passive_Voice_Ratio: float
    Reading_Level: float
    Editing_Time: float | None = None
    Revision_Count: float | None = None
    Typing_Speed: float | None = None
    Plagiarism_Score: float | None = None
    Citation_Count: float | None = None
    Punctuation_Density: float
    Sentence_Complexity: float
    Font_Family: str
    Submission_Hour: int
    # инженерные фичи считаем сами внутри эндпоинта, пользователь их не вводит

@app.post("/predict")
def predict(submission: StudentSubmission):
    data = submission.model_dump()
    df = pd.DataFrame([data])

    # пересоздаём инженерные фичи так же, как в ноутбуке на Шаге 4
    df['Errors_per_Word'] = df['Grammar_Errors'] / df['Word_Count']
    df['Revisions_per_Word'] = df['Revision_Count'] / df['Word_Count']
    df['Citations_per_Word'] = df['Citation_Count'] / df['Word_Count']

    pred_full = int(full_model.predict(df)[0])
    proba_full = float(full_model.predict_proba(df)[0][1])
    pred_linguistic = int(linguistic_model.predict(df)[0])

    disagreement = pred_full != pred_linguistic

    return {
        "full_model_prediction": "AI-assisted" if pred_full == 1 else "Human-written",
        "full_model_probability": round(proba_full, 3),
        "linguistic_model_prediction": "AI-assisted" if pred_linguistic == 1 else "Human-written",
        "disagreement": disagreement,
        "needs_manual_review": disagreement
    }

@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>AI Writing Detection</title>
<style>
    body {
        font-family: -apple-system, Segoe UI, Roboto, sans-serif;
        max-width: 640px;
        margin: 40px auto;
        padding: 0 20px;
        background: #f7f7fb;
        color: #1a1a1a;
    }
    h1 { font-size: 1.5rem; margin-bottom: 4px; }
    p.subtitle { color: #666; margin-top: 0; margin-bottom: 24px; }
    .card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    label { display: block; font-size: 0.85rem; color: #444; margin-top: 12px; margin-bottom: 4px; }
    input, select {
        width: 100%;
        padding: 8px 10px;
        border: 1px solid #ddd;
        border-radius: 6px;
        font-size: 0.95rem;
        box-sizing: border-box;
    }
    button {
        margin-top: 20px;
        width: 100%;
        padding: 12px;
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 1rem;
        cursor: pointer;
    }
    button:hover { background: #1d4ed8; }
    #result { margin-top: 16px; padding: 16px; border-radius: 8px; display: none; }
    .result-human { background: #ecfdf5; border: 1px solid #10b981; }
    .result-ai { background: #fef2f2; border: 1px solid #ef4444; }
    .result-review { background: #fffbeb; border: 1px solid #f59e0b; }
    .result-row { margin: 4px 0; font-size: 0.9rem; }
    .badge { font-weight: 600; }
</style>
</head>
<body>
    <h1>🔍 AI Writing Detection</h1>
    <p class="subtitle">Заполни данные работы студента и проверь предсказание модели</p>

    <div class="card">
        <label>Submission Type</label>
        <select id="Submission_Type">
            <option>Essay</option><option>Research_Paper</option>
            <option>Lab_Report</option><option>Literature_Review</option>
        </select>

        <label>Academic Level</label>
        <select id="Academic_Level">
            <option>High_School</option><option>Undergraduate</option><option>Postgraduate</option>
        </select>

        <label>Primary Language</label>
        <select id="Primary_Language"><option>Native</option><option>Non_Native</option></select>

        <label>Font Family</label>
        <select id="Font_Family">
            <option>Arial</option><option>Calibri</option><option>Helvetica</option><option>Times New Roman</option>
        </select>

        <label>Word Count</label>
        <input type="number" id="Word_Count" value="1000">

        <label>Average Sentence Length</label>
        <input type="number" step="0.01" id="Average_Sentence_Length" value="17">

        <label>Grammar Errors</label>
        <input type="number" id="Grammar_Errors" value="5">

        <label>Vocabulary Richness (0-1)</label>
        <input type="number" step="0.01" id="Vocabulary_Richness" value="0.6">

        <label>Passive Voice Ratio (0-1)</label>
        <input type="number" step="0.01" id="Passive_Voice_Ratio" value="0.2">

        <label>Reading Level</label>
        <input type="number" step="0.01" id="Reading_Level" value="10">

        <label>Editing Time (минуты, можно пусто)</label>
        <input type="number" step="0.01" id="Editing_Time" value="60">

        <label>Revision Count (можно пусто)</label>
        <input type="number" id="Revision_Count" value="10">

        <label>Typing Speed (можно пусто)</label>
        <input type="number" step="0.01" id="Typing_Speed" value="45">

        <label>Plagiarism Score (можно пусто)</label>
        <input type="number" step="0.01" id="Plagiarism_Score" value="5">

        <label>Citation Count (можно пусто)</label>
        <input type="number" id="Citation_Count" value="3">

        <label>Punctuation Density</label>
        <input type="number" step="0.001" id="Punctuation_Density" value="0.08">

        <label>Sentence Complexity</label>
        <input type="number" step="0.01" id="Sentence_Complexity" value="5">

        <label>Submission Hour (0-23)</label>
        <input type="number" id="Submission_Hour" value="14">

        <button onclick="predict()">Проверить</button>
    </div>

    <div id="result" class="card"></div>

<script>
async function predict() {
    const payload = {
        Submission_Type: document.getElementById('Submission_Type').value,
        Academic_Level: document.getElementById('Academic_Level').value,
        Primary_Language: document.getElementById('Primary_Language').value,
        Font_Family: document.getElementById('Font_Family').value,
        Word_Count: Number(document.getElementById('Word_Count').value),
        Average_Sentence_Length: Number(document.getElementById('Average_Sentence_Length').value),
        Grammar_Errors: Number(document.getElementById('Grammar_Errors').value),
        Vocabulary_Richness: Number(document.getElementById('Vocabulary_Richness').value),
        Passive_Voice_Ratio: Number(document.getElementById('Passive_Voice_Ratio').value),
        Reading_Level: Number(document.getElementById('Reading_Level').value),
        Editing_Time: Number(document.getElementById('Editing_Time').value),
        Revision_Count: Number(document.getElementById('Revision_Count').value),
        Typing_Speed: Number(document.getElementById('Typing_Speed').value),
        Plagiarism_Score: Number(document.getElementById('Plagiarism_Score').value),
        Citation_Count: Number(document.getElementById('Citation_Count').value),
        Punctuation_Density: Number(document.getElementById('Punctuation_Density').value),
        Sentence_Complexity: Number(document.getElementById('Sentence_Complexity').value),
        Submission_Hour: Number(document.getElementById('Submission_Hour').value)
    };

    const res = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    });
    const data = await res.json();

    const box = document.getElementById('result');
    box.style.display = 'block';
    box.className = 'card ' + (data.needs_manual_review ? 'result-review' : (data.full_model_prediction === 'AI-assisted' ? 'result-ai' : 'result-human'));
    box.innerHTML = `
        <div class="result-row"><span class="badge">Полная модель:</span> ${data.full_model_prediction} (вероятность AI: ${data.full_model_probability})</div>
        <div class="result-row"><span class="badge">Лингвистическая модель:</span> ${data.linguistic_model_prediction}</div>
        <div class="result-row"><span class="badge">${data.needs_manual_review ? '⚠️ Модели разошлись — нужна ручная проверка' : '✅ Модели согласны'}</span></div>
    `;
}
</script>
</body>
</html>
"""
    return {"message": "Open /docs for the interactive test form."}