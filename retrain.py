# retrain.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split # Optional: for evaluating on new data split
from sklearn.metrics import classification_report, accuracy_score # Optional: for evaluation

# 从配置文件导入常量
import config
# 从工具模块导入函数
from utils import clean_text

def load_existing_components():
    """加载已保存的向量化器和所有模型"""
    loaded_objects = {}
    print("--- 开始加载现有模型和向量化器 ---")

    # 加载向量化器
    try:
        vectorizer = joblib.load(config.VECTORIZER_PATH)
        loaded_objects['vectorizer'] = vectorizer
        print(f"向量化器加载成功: {config.VECTORIZER_PATH}")
    except FileNotFoundError:
        print(f"错误: 现有向量化器文件未找到: {config.VECTORIZER_PATH}. 无法继续 retraining。")
        return None
    except Exception as e:
        print(f"加载向量化器时出错: {e}")
        return None

    # 定义模型路径字典
    model_paths = {
        "逻辑回归": config.LR_MODEL_PATH,
        "决策树": config.DT_MODEL_PATH,
        "梯度提升": config.GBC_MODEL_PATH,
        "随机森林": config.RFC_MODEL_PATH
    }

    loaded_models = {}
    all_loaded = True
    # 加载模型
    for name, path in model_paths.items():
        try:
            model = joblib.load(path)
            loaded_models[name] = model
            print(f"现有 {name} 模型加载成功: {path}")
        except FileNotFoundError:
            print(f"警告: 现有模型文件未找到: {path}. 将跳过 retraining 该模型。")
            # 不标记 all_loaded 为 False，因为我们可以只 retrain 加载成功的
        except Exception as e:
            print(f"加载 {name} 模型时出错: {e}")
            # 同上，不阻止其他模型 retraining

    if not loaded_models:
         print("错误：未能加载任何现有模型。Retraining 终止。")
         return None

    loaded_objects['models'] = loaded_models
    print("--- 现有组件加载完成 ---")
    return loaded_objects

def load_and_prepare_new_data(new_fake_path, new_true_path):
    """加载、准备和清理 *新* 的训练数据"""
    all_data_loaded = True
    try:
        df_new_fake = pd.read_csv(new_fake_path)
        print(f"\n成功加载新数据: {new_fake_path} ({len(df_new_fake)} 条)")
    except FileNotFoundError:
        print(f"警告：无法找到新数据文件 '{new_fake_path}'。")
        df_new_fake = pd.DataFrame(columns=['text']) # 创建空 DataFrame
        all_data_loaded = False
    except Exception as e:
        print(f"加载文件 {new_fake_path} 时发生错误: {e}")
        df_new_fake = pd.DataFrame(columns=['text'])
        all_data_loaded = False

    try:
        df_new_true = pd.read_csv(new_true_path)
        print(f"成功加载新数据: {new_true_path} ({len(df_new_true)} 条)")
    except FileNotFoundError:
        print(f"警告：无法找到新数据文件 '{new_true_path}'。")
        df_new_true = pd.DataFrame(columns=['text']) # 创建空 DataFrame
        all_data_loaded = False
    except Exception as e:
        print(f"加载文件 {new_true_path} 时发生错误: {e}")
        df_new_true = pd.DataFrame(columns=['text'])
        all_data_loaded = False

    if not all_data_loaded and df_new_fake.empty and df_new_true.empty:
        print("错误：未能加载任何新数据文件。")
        return None

    # 添加类别标签
    if not df_new_fake.empty:
        df_new_fake["class"] = 0
    if not df_new_true.empty:
        df_new_true["class"] = 1

    # 合并新数据
    df_new_merged = pd.concat([df_new_fake, df_new_true], axis=0, ignore_index=True)
    if df_new_merged.empty:
         print("错误：合并后的新数据为空。")
         return None

    print(f"合并后的新数据形状: {df_new_merged.shape}")

    # 选择需要的列 ('text', 'class')
    if 'text' not in df_new_merged.columns or 'class' not in df_new_merged.columns:
        print("错误：新数据缺少 'text' 或 'class' 列。")
        # 尝试仅使用 'text' 列，如果存在
        if 'text' in df_new_merged.columns:
            print("警告：缺少 'class' 列，将尝试仅处理 'text' 列（无法用于训练）。")
            df_new = df_new_merged[['text']].copy()
            # 这种情况无法训练，需要用户修复数据
            return None # 或者只返回文本供其他用途
        else:
            return None


    df_new = df_new_merged[['text', 'class']].copy()

    # 检查并处理缺失值
    initial_rows = len(df_new)
    df_new.dropna(subset=['text'], inplace=True)
    if len(df_new) < initial_rows:
        print(f"处理新数据中的缺失值：移除了 {initial_rows - len(df_new)} 行。")
    print(f"处理缺失值后的新数据形状: {df_new.shape}")

    if df_new.empty:
        print("错误：新数据在处理缺失值后为空，无法进行 retraining。")
        return None

    # 清理文本数据
    print("开始清理新数据的文本...")
    df_new["text"] = df_new["text"].apply(clean_text)
    print("新数据文本清理完成。")

    # 打乱新数据（可选，但通常是好的做法）
    df_new = df_new.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    print("新数据已打乱顺序。")

    return df_new


def continue_training(loaded_components, new_data_df):
    """使用新数据继续训练加载的模型"""
    if not loaded_components or new_data_df is None or new_data_df.empty:
        print("错误：缺少现有组件或有效的新数据，无法继续训练。")
        return

    vectorizer = loaded_components.get('vectorizer')
    models = loaded_components.get('models')

    if not vectorizer or not models:
        print("错误：向量化器或模型未正确加载。")
        return

    print("\n--- 开始使用新数据进行 Retraining ---")

    # 准备新数据的特征和标签
    X_new = new_data_df["text"]
    y_new = new_data_df["class"]

    # 使用 *现有* 的向量化器转换新数据
    print("使用现有向量化器转换新文本数据...")
    try:
        # 检查向量化器是否为空（如果训练时未产生特征）
        if not hasattr(vectorizer, 'vocabulary_') or not vectorizer.vocabulary_:
             print("错误：加载的向量化器为空（没有词汇表），无法转换新数据。请重新运行 train.py。")
             return
        X_new_vec = vectorizer.transform(X_new)
        print(f"新数据向量化完成，形状: {X_new_vec.shape}")
    except ValueError as e:
         if "empty vocabulary" in str(e):
              print("错误：向量化器词汇表为空，无法转换。可能初始训练数据太少或无效。")
         else:
              print(f"新数据向量化失败: {e}")
         return
    except Exception as e:
        print(f"新数据向量化失败: {e}")
        return

    # --- Retrain 每个加载成功的模型 ---
    retrained_models = {}
    model_save_paths = { # 从 config 获取保存路径
        "逻辑回归": config.LR_MODEL_PATH,
        "决策树": config.DT_MODEL_PATH,
        "梯度提升": config.GBC_MODEL_PATH,
        "随机森林": config.RFC_MODEL_PATH
    }

    for name, model in models.items():
        print(f"\n--- Retraining 模型: {name} ---")
        try:
            # 注意：对于 scikit-learn 的标准分类器，再次调用 fit
            # 会使用新数据重新训练模型。
            model.fit(X_new_vec, y_new)
            print(f"{name} 模型已使用新数据进行 retraining。")

            # --- 保存更新后的模型 ---
            save_path = model_save_paths.get(name) # 获取模型保存路径
            if save_path:
                 print(f"正在保存更新后的 {name} 模型到: {save_path}")
                 joblib.dump(model, save_path)
                 print(f"更新后的 {name} 模型保存成功。")
                 retrained_models[name] = model # 更新字典中的模型对象
            else:
                 print(f"警告：无法找到 {name} 模型的保存路径配置，未保存。")

        except Exception as e:
            print(f"Retraining 或保存 {name} 模型时出错: {e}")

    print("\n--- 所有可用模型的 Retraining 和保存完成 ---")
    loaded_components['models'] = retrained_models # 更新主字典中的模型


# --- 主执行流程 ---
if __name__ == "__main__":
    print("--- 开始执行 Retraining 脚本 ---")

    # 1. 加载现有组件 (模型和向量化器)
    existing_components = load_existing_components()

    if existing_components:
        # 2. 加载和准备新数据
        new_data = load_and_prepare_new_data(config.NEW_FAKE_CSV_PATH, config.NEW_TRUE_CSV_PATH)

        if new_data is not None and not new_data.empty:
            # 3. 使用新数据继续训练 (Retrain)
            continue_training(existing_components, new_data)

            # 可选：在这里添加评估步骤
            # print("\n--- Retraining 后评估 (可选) ---")

        else:
            print("\n未能加载或准备有效的新数据，Retraining 过程未在模型上执行。")
    else:
        print("\n未能加载现有模型或向量化器，Retraining 终止。")

    print("\n--- Retraining 脚本执行完毕 ---")