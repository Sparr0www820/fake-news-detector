# config.py
import os

# 数据路径
# 获取当前脚本所在的目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data") # 数据目录
FAKE_CSV_PATH = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV_PATH = os.path.join(DATA_DIR, "True.csv")

# 模型和向量化器保存路径
MODEL_DIR = os.path.join(BASE_DIR, "models") # 模型保存目录
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
LR_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.joblib")
DT_MODEL_PATH = os.path.join(MODEL_DIR, "decision_tree.joblib")
GBC_MODEL_PATH = os.path.join(MODEL_DIR, "gradient_boosting.joblib")
RFC_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.joblib")

# 训练参数
TEST_SET_SIZE = 0.25       # 测试集比例 (注意：如果数据量很小，可能导致测试集为空)
RANDOM_SEED = 42           # 随机种子，确保结果可复现
MAX_FEATURES = 10000       # TF-IDF 最大特征数
MAX_ITER_LR = 1000         # 逻辑回归最大迭代次数

# Paths for New Data for Retraining
NEW_FAKE_CSV_PATH = os.path.join(DATA_DIR, "New_Fake.csv") # Path for new fake news data
NEW_TRUE_CSV_PATH = os.path.join(DATA_DIR, "New_True.csv") # Path for new true news data

# 确保模型目录存在
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"配置加载完成: ")
print(f" - 数据目录: {DATA_DIR}")
print(f" - 模型保存目录: {MODEL_DIR}")
print(f" - 原始 Fake 数据路径: {FAKE_CSV_PATH}")
print(f" - 原始 True 数据路径: {TRUE_CSV_PATH}")
print(f" - 新 Fake 数据路径: {NEW_FAKE_CSV_PATH}")
print(f" - 新 True 数据路径: {NEW_TRUE_CSV_PATH}")
