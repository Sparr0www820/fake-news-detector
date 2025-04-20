# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib # 用于保存和加载模型/向量化器
import os

# 从配置文件导入常量
import config
# 从工具模块导入函数
from utils import clean_text

def load_and_prepare_training_data(fake_path, true_path):
    """加载数据，添加标签，合并，清理，打乱，仅用于训练"""
    try:
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)
        print(f"成功加载数据: {fake_path} ({len(df_fake)} 条) 和 {true_path} ({len(df_true)} 条)")
    except FileNotFoundError:
        print(f"错误：无法找到数据文件。请检查路径 '{fake_path}' 和 '{true_path}' 是否正确。")
        return None
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return None

    # 添加类别标签：0 表示虚假新闻，1 表示真实新闻
    df_fake["class"] = 0
    df_true["class"] = 1

    # 合并训练数据
    df_merged = pd.concat([df_fake, df_true], axis=0, ignore_index=True)
    print(f"合并后的总数据形状: {df_merged.shape}")

    # 选择需要的列 ('text', 'class')
    df = df_merged[['text', 'class']].copy()

    # 检查是否有缺失值
    print("\n缺失值检查:")
    print(df.isnull().sum())
    initial_rows = len(df)
    df.dropna(subset=['text'], inplace=True) # 删除 text 列为空的行
    if len(df) < initial_rows:
        print(f"处理缺失值：移除了 {initial_rows - len(df)} 行。")
    print(f"处理缺失值后的数据形状: {df.shape}")

    if df.empty:
        print("错误：数据为空或清理后为空。")
        return None

    # 清理文本数据
    print("\n开始清理文本数据...")
    df["text"] = df["text"].apply(clean_text)
    print("文本数据清理完成。")

    # 打乱数据顺序
    df = df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    print("数据已打乱顺序。")

    return df

def train_evaluate_and_save(df):
    """训练、评估分类模型并保存训练好的模型和向量化器"""
    if df is None or df.empty:
        print("错误：没有可用于训练的数据。")
        return

    # 定义特征 (X) 和目标 (y)
    X = df["text"]
    y = df["class"]

    # 检查数据量是否足够划分测试集
    if len(df) * config.TEST_SET_SIZE < 1:
        print("警告：数据量过小，无法划分出有效的测试集。将使用所有数据进行训练，跳过评估。")
        x_train, x_test, y_train, y_test = X, pd.Series(dtype='object'), y, pd.Series(dtype='int') # 创建空的测试集
        evaluate = False
    else:
        # 划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SET_SIZE, random_state=config.RANDOM_SEED, stratify=y # 添加 stratify 保证类别比例
        )
        print(f"\n数据集划分: 训练集 {len(x_train)} 条, 测试集 {len(x_test)} 条")
        evaluate = True


    # 文本向量化 (TF-IDF)
    print("开始进行 TF-IDF 向量化...")
    # 调整 min_df 可以过滤掉出现次数过少的词
    vectorizer = TfidfVectorizer(max_features=config.MAX_FEATURES, min_df=2)
    try:
        xv_train = vectorizer.fit_transform(x_train)
        if evaluate:
             xv_test = vectorizer.transform(x_test)
        print("TF-IDF 向量化完成。")
        print(f"特征维度: {xv_train.shape[1]}")
    except ValueError as e:
         print(f"TF-IDF 向量化错误（可能是所有词都被过滤了）: {e}")
         return
    except Exception as e:
        print(f"TF-IDF 向量化时发生未知错误: {e}")
        return

    # 保存向量化器
    print(f"正在保存 TF-IDF 向量化器到: {config.VECTORIZER_PATH}")
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    print("向量化器保存成功。")

    # 初始化模型
    models = {
        "逻辑回归": (LogisticRegression(random_state=config.RANDOM_SEED, max_iter=config.MAX_ITER_LR), config.LR_MODEL_PATH),
        "决策树": (DecisionTreeClassifier(random_state=config.RANDOM_SEED), config.DT_MODEL_PATH),
        "梯度提升": (GradientBoostingClassifier(random_state=config.RANDOM_SEED), config.GBC_MODEL_PATH),
        "随机森林": (RandomForestClassifier(random_state=config.RANDOM_SEED), config.RFC_MODEL_PATH)
    }

    # 训练、评估和保存每个模型
    print("\n开始训练、评估和保存模型...")
    for name, (model, save_path) in models.items():
        print(f"\n训练模型: {name}")
        try:
            model.fit(xv_train, y_train)
        except Exception as e:
            print(f"训练 {name} 模型时出错: {e}")
            continue # 跳过这个模型

        if evaluate:
            print(f"评估模型: {name}")
            try:
                predictions = model.predict(xv_test)
                accuracy = accuracy_score(y_test, predictions)
                # 检查测试集标签种类数，避免 classification_report 报错
                if len(np.unique(y_test)) > 1:
                     report = classification_report(y_test, predictions, target_names=['虚假新闻 (0)', '真实新闻 (1)'], zero_division=0)
                     print(f"{name} - 测试集准确率: {accuracy:.4f}")
                     print(f"{name} - 分类报告:\n{report}")
                else:
                     print(f"{name} - 测试集准确率: {accuracy:.4f}")
                     print(f"{name} - 分类报告: 测试集标签单一，无法生成详细报告。")

            except Exception as e:
                print(f"评估 {name} 模型时出错: {e}")

        # 保存模型
        print(f"正在保存 {name} 模型到: {save_path}")
        try:
            joblib.dump(model, save_path)
            print(f"{name} 模型保存成功。")
        except Exception as e:
             print(f"保存 {name} 模型时出错: {e}")


    print("\n所有模型训练、评估和保存完成")

# 主执行流程
if __name__ == "__main__":
    print("开始执行训练脚本")

    # 1. 加载和准备数据
    prepared_data = load_and_prepare_training_data(config.FAKE_CSV_PATH, config.TRUE_CSV_PATH)

    # 2. 训练、评估和保存
    if prepared_data is not None:
        train_evaluate_and_save(prepared_data)
    else:
        print("\n数据加载或准备失败，训练终止。")

    print("\n训练脚本执行完毕")
