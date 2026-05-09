import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ================= НАСТРОЙКИ =================
DATA_DIR = Path('data')
TRAIN_PATH = DATA_DIR / 'train_data.csv'
TEST_PATH = DATA_DIR / 'test.csv'
SUBMISSION_TEMPLATE = DATA_DIR / 'sample_submission.csv'
OUTPUT_SUBMISSION = 'submission.csv'
OUTPUT_TRAIN_EVAL = 'train_evaluation.csv'

RANDOM_STATE = 42
# =============================================

# 1. Загрузка данных
print("Загружаю данные...")
train = pd.read_csv(TRAIN_PATH, index_col=0)
test  = pd.read_csv(TEST_PATH, index_col='id')

# Удаляем возможную безымянную колонку
if '' in train.columns:
    train.drop(columns=[''], inplace=True)
if '' in test.columns:
    test.drop(columns=[''], inplace=True)

# 2. Определение целевой переменной
target_col = None
for col in ['target', 'target_variable', 'label', 'is_variable']:
    if col in train.columns:
        target_col = col
        break
if target_col is None:
    target_col = train.columns[-1]
    print(f"Целевая колонка не найдена по имени, использую последний столбец: '{target_col}'")
else:
    print(f"Целевая колонка: '{target_col}'")

y = train[target_col]
X = train.drop(columns=[target_col])

print(f"Распределение классов:\n{y.value_counts()}")

# 3. Feature engineering
def add_features(df):
    df = df.copy()
    df['B-V'] = df['Bmag'] - df['Vmag']
    df['U-V'] = df['fuv_mag'] - df['Vmag']
    return df

X = add_features(X)
test = add_features(test)

# Оставляем только общие колонки
common_cols = X.columns.intersection(test.columns)
X = X[common_cols]
test = test[common_cols]
print(f"Число используемых признаков: {len(common_cols)}")

# 4. Разделение на train/val
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# 5. Baseline (Dummy)
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_tr, y_tr)
y_dummy = dummy.predict(X_val)
print("\n=== Dummy Baseline (most_frequent) ===")
print(classification_report(y_val, y_dummy))
print(f"F1 dummy: {f1_score(y_val, y_dummy):.4f}")

# 6. Модели для сравнения
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ),
    'SVM (LinearSVC)': make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=2000)
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
        random_state=RANDOM_STATE,
        verbose=-1
    )
}

# 7. Обучение и оценка всех моделей
best_f1 = -1
best_model_name = None

for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    print(classification_report(y_val, y_pred))
    print(f"{name} F1: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name

print(f"\nЛучшая модель по F1: {best_model_name} (F1={best_f1:.4f})")

# 8. Финальное обучение лучшей модели на всех train данных
if best_model_name == 'LightGBM':
    final_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=RANDOM_STATE,
        verbose=-1
    )
elif best_model_name == 'RandomForest':
    final_model = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
else:  # SVM
    final_model = make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=2000)
    )

final_model.fit(X, y)

# 9. Предсказание на тестовой выборке и сохранение submission.csv
test_pred = final_model.predict(test)

submission = pd.read_csv(SUBMISSION_TEMPLATE)
submission['0'] = test_pred
submission.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"\nФайл {OUTPUT_SUBMISSION} сохранён (колонки: id, 0).")

# 10. Оценка на полной обучающей выборке и сохранение train_evaluation.csv
#     Содержит истинные метки, предсказания и флаг совпадения
train_pred = final_model.predict(X)
train_eval = pd.DataFrame({
    'id': train.index,
    'true_label': y,
    'predicted_label': train_pred,
    'match': (y == train_pred).astype(int)
})
train_eval.to_csv(OUTPUT_TRAIN_EVAL, index=False)
print(f"Файл {OUTPUT_TRAIN_EVAL} сохранён (колонки: id, true_label, predicted_label, match).")
