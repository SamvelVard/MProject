import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ================= НАСТРОЙКИ =================
DATA_DIR = Path('data')
TRAIN_PATH = DATA_DIR / 'train_data.csv'
TEST_PATH = DATA_DIR / 'test.csv'
SUBMISSION_TEMPLATE = DATA_DIR / 'sample_submission.csv'
OUTPUT_PATH = 'submission.csv'

RANDOM_STATE = 42
# =============================================

# ---------- 1. Загрузка данных ----------
print("Загружаю данные...")
# В train первая колонка — индекс без названия (используем index_col=0)
train = pd.read_csv(TRAIN_PATH, index_col=0)
# В test колонка 'id' задана явно
test  = pd.read_csv(TEST_PATH, index_col='id')

print(f"train: {train.shape}, test: {test.shape}")

# Удаляем возможную безымянную колонку (если осталась после index_col)
if '' in train.columns:
    train.drop(columns=[''], inplace=True)
if '' in test.columns:
    test.drop(columns=[''], inplace=True)

# ---------- 2. Определяем целевую переменную ----------
target_col = None
for col in ['target', 'target_variable', 'label', 'is_variable']:
    if col in train.columns:
        target_col = col
        break

if target_col is None:
    # Если имя не найдено, берём последний столбец как целевую переменную
    target_col = train.columns[-1]
    print(f"Целевая колонка не найдена по имени, использую последний столбец: '{target_col}'")
else:
    print(f"Целевая колонка: '{target_col}'")

y = train[target_col]
X = train.drop(columns=[target_col])

print(f"Распределение классов:\n{y.value_counts()}")

# ---------- 3. Инженерия признаков ----------
def add_features(df):
    df = df.copy()
    df['B-V'] = df['Bmag'] - df['Vmag']
    df['U-V'] = df['fuv_mag'] - df['Vmag']
    return df

X = add_features(X)
test = add_features(test)

# Убедимся, что признаки в train и test совпадают
common_cols = X.columns.intersection(test.columns)
X = X[common_cols]
test = test[common_cols]
print(f"Число используемых признаков: {len(common_cols)}")

# ---------- 4. Разделение на train/val ----------
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------- 5. Обучение модели (LightGBM) ----------
print("Обучаю модель...")
scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    verbose=-1
)

model.fit(X_tr, y_tr)

# ---------- 6. Оценка на валидации ----------
y_pred = model.predict(X_val)
print("\n=== Оценка на валидации ===")
print(classification_report(y_val, y_pred))
print(f"F1: {f1_score(y_val, y_pred):.4f}")

# ---------- 7. Предсказание на тесте и сохранение ----------
print("Делаю предсказание для тестовой выборки...")
test_pred = model.predict(test)

submission = pd.read_csv(SUBMISSION_TEMPLATE)
submission['0'] = test_pred   # имя колонки должно совпадать с шаблоном
submission.to_csv(OUTPUT_PATH, index=False)

print(f"Файл {OUTPUT_PATH} сохранён. Готово!")
