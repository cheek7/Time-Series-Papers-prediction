# Установка необходимых библиотек
!pip install optuna lightgbm statsmodels catboost

# Импорт библиотек
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
import os

# 1. Загрузка данных
train = pd.read_csv("train.csv")
train['submitted_date'] = pd.to_datetime(train['submitted_date'])

# 2. Функции для обработки данных
def get_category_data(df: pd.DataFrame, category: str):
    category_data = df[df['category'] == category].copy()
    category_data = category_data.sort_values('submitted_date')
    category_data = category_data.set_index('submitted_date')
    return category_data.drop('category', axis=1)

def extend_dataset(category_data, last_train_date, future_weeks_num: int = 0):
    extended_data = category_data.copy()
    if future_weeks_num > 0:
        future_dates = pd.date_range(
            start=last_train_date + pd.Timedelta(days=1),
            periods=future_weeks_num * 7,
            freq='D'
        )
        future = pd.DataFrame(index=future_dates, data={'num_papers': 0})
        extended_data = pd.concat([extended_data, future])
    return extended_data

def get_rolling_features(cat_data: pd.DataFrame, global_data: pd.DataFrame = None):
    daily_features = pd.DataFrame(index=cat_data.index)
    
    # Временные признаки
    daily_features['dayofweek'] = daily_features.index.dayofweek
    daily_features['is_weekend'] = (daily_features['dayofweek'] >= 5).astype(int)
    daily_features['month'] = daily_features.index.month
    daily_features['rolling_3'] = cat_data['num_papers'].rolling(3).mean().shift(1)
    daily_features['rolling_7'] = cat_data['num_papers'].rolling(7).mean().shift(1)
    daily_features['expanding_mean'] = cat_data['num_papers'].expanding(2).mean().shift(1)

    # Rolling-статистики
    weekly_rolling = cat_data['num_papers'].rolling('7D', min_periods=1)
    monthly_rolling = cat_data['num_papers'].rolling('28D', min_periods=1)
    quarterly_rolling = cat_data['num_papers'].rolling('90D', min_periods=1)

    daily_features['rolling_sum_during_week'] = weekly_rolling.sum()
    daily_features['rolling_max_during_week'] = weekly_rolling.max()
    daily_features['rolling_min_during_month'] = monthly_rolling.min()
    daily_features['rolling_max_during_month'] = monthly_rolling.max()
    daily_features['rolling_std_28d'] = monthly_rolling.std()
    daily_features['rolling_median_28d'] = monthly_rolling.median()
    daily_features['rolling_mean_90d'] = quarterly_rolling.mean()

    # Сезонные признаки
    daily_features['week_of_year'] = daily_features.index.isocalendar().week
    daily_features['is_holiday_period'] = ((daily_features.index.month == 12) & (daily_features.index.day >= 25)) | \
                                         ((daily_features.index.month == 1) & (daily_features.index.day <= 5)).astype(int)

    # Темп роста
    daily_features['growth_rate_weekly'] = (daily_features['rolling_sum_during_week'] /
                                           daily_features['rolling_sum_during_week'].shift(7) - 1).fillna(0)

    # Глобальные признаки
    if global_data is not None:
        global_weekly = global_data.groupby(global_data.index)['num_papers'].sum().rolling('7D', min_periods=1).mean()
        daily_features['global_weekly_mean'] = global_weekly.reindex(daily_features.index, method='ffill')

    return daily_features

def add_lag_features(features: pd.DataFrame):
    new_features = features.copy()
    for col in features.columns:
        new_features[f'{col}_last_week'] = features[col].shift(freq=pd.Timedelta(days=7))
        new_features[f'{col}_4w_ago'] = features[col].shift(freq=pd.Timedelta(days=28))
        new_features[f'{col}_year_ago'] = features[col].shift(freq=pd.Timedelta(days=364))
    return new_features.dropna()

def build_weekly_features(features):
    daily_features = features.reset_index(names="day")
    daily_features['week'] = daily_features['day'] + pd.to_timedelta(6 - daily_features['day'].dt.weekday, unit='D')
    weekly_features = daily_features.groupby('week').last().reset_index()
    return weekly_features.drop('day', axis=1)

def build_targets(category_data, week_horizon: int):
    targets = category_data.resample('W').sum().shift(-week_horizon).num_papers.rename('target')
    targets.index.name = 'week'
    return targets

# 3. Подготовка данных
last_train_date = train['submitted_date'].max()
global_data = train.set_index('submitted_date')
dataset = []

progress_bar = tqdm(train.category.unique())
for category in progress_bar:
    category_data = get_category_data(train, category)
    extended_category_data = extend_dataset(category_data, last_train_date=last_train_date, future_weeks_num=1)
    rolling_features = get_rolling_features(cat_data=extended_category_data, global_data=global_data)
    lag_features = add_lag_features(rolling_features)
    weekly_features = build_weekly_features(lag_features)
    targets = build_targets(category_data=category_data, week_horizon=1)
    data = weekly_features.merge(targets, on='week')
    data['category'] = category
    dataset.append(data)

dataset = pd.concat(dataset)
dataset['category'] = dataset['category'].astype('category')
labeled_data = dataset[dataset.target.notnull()].reset_index(drop=True).dropna()

# 4. Разделение данных
n_valid_weeks = 4
valid_start_date = last_train_date - pd.Timedelta(days=7 * n_valid_weeks)
valid_dataset = labeled_data[labeled_data.week > valid_start_date]
train_dataset = labeled_data[labeled_data.week <= valid_start_date]
test_dataset = dataset[dataset.target.isnull()].reset_index(drop=True)

# 5. Подготовка данных для LightGBM
train_set = lgb.Dataset(train_dataset.drop(['week', 'target'], axis=1), label=train_dataset['target'])
valid_set = lgb.Dataset(valid_dataset.drop(['week', 'target'], axis=1), label=valid_dataset['target'])
test_set = test_dataset.drop(['week', 'target'], axis=1)

# 6. Определение кастомной метрики
def safe_mape_lgb(y_pred, dataset):
    y_true = dataset.get_label()
    denominator = pd.Series(y_true).abs().clip(lower=10.0)
    error = abs(y_pred - y_true) / denominator
    return 'safe_mape', error.mean(), False

# 7. Оптимизация гиперпараметров с Optuna
def objective(trial):
    params = {
        'objective': 'regression',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 1),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 1),
        'verbosity': -1,
        'metric': 'mape',
        'seed': 42
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        valid_names=['valid'],
        feval=safe_mape_lgb,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    y_pred = model.predict(valid_set.data)
    y_true = valid_set.label
    denominator = pd.Series(y_true).abs().clip(lower=10.0)
    error = abs(y_pred - y_true) / denominator
    return error.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_params = study.best_params

# 8. Обучение финальной модели
params = {
    'objective': 'regression',
    'learning_rate': 0.03,
    'depth': 6,
    'num_leaves': 40,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbosity': -1,
    'metric': 'mape',
    'seed': 42
}
params.update(best_params)

model = lgb.train(
    params,
    train_set,
    num_boost_round=1000,
    valid_sets=[valid_set],
    valid_names=['valid'],
    feval=safe_mape_lgb,
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=50)
    ]
)

# 9. Предсказания и постобработка
test_dataset['predicted'] = model.predict(test_set)
test_dataset['predicted'] = test_dataset['predicted'].clip(lower=0).round().astype(int)

# 10. Формирование submission
sample_submission = pd.read_csv("sample_submission.csv")
sample_submission['category'] = sample_submission['id'].apply(lambda x: x.split('__')[0])

submission_data = []
for week_id in range(1, 9):
    temp = test_dataset[['category', 'predicted']].copy()
    temp['week_id'] = week_id
    temp['id'] = temp['category'].astype(str) + '__' + temp['week_id'].astype(str)
    submission_data.append(temp[['id', 'predicted']])

submission = pd.concat(submission_data)
submission = submission.merge(sample_submission[['id']], on='id', how='right')
submission.rename(columns={'predicted': 'num_papers'}, inplace=True)

# 11. Проверка и сохранение
if submission['num_papers'].isnull().any():
    print("В submission есть пропуски! Проверьте совпадение категорий.")
else:
    print("Submission сформирован успешно!")

submission.to_csv('improved_submission.csv', index=False)
print(submission.shape)
print(submission.head())
