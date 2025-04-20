# predict.py
import joblib
import os
import pandas as pd # 保留以便未来扩展

# 从配置文件导入常量
import config
# 从工具模块导入函数
from utils import clean_text

def load_components():
    """加载保存好的向量化器和所有模型"""
    loaded_objects = {}
    print("开始加载模型和向量化器")

    # 加载向量化器
    vectorizer = None
    try:
        vectorizer = joblib.load(config.VECTORIZER_PATH)
        # 检查向量化器是否有效
        if not hasattr(vectorizer, 'transform'):
             print(f"错误: 加载的对象不是有效的向量化器: {config.VECTORIZER_PATH}")
             return None
        if not hasattr(vectorizer, 'vocabulary_') or not vectorizer.vocabulary_:
             print(f"警告: 加载的向量化器词汇表为空，可能导致预测问题。")
             # 允许继续，但在预测时可能会失败

        loaded_objects['vectorizer'] = vectorizer
        print(f"向量化器加载成功: {config.VECTORIZER_PATH}")
    except FileNotFoundError:
        print(f"错误: 向量化器文件未找到: {config.VECTORIZER_PATH}. 请先运行 train.py。")
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
    all_loaded_successfully = True
    # 加载模型
    for name, path in model_paths.items():
        try:
            model = joblib.load(path)
            # 基本检查模型是否可预测
            if not hasattr(model, 'predict'):
                 print(f"警告: 加载的 {name} 模型缺少 predict 方法，路径: {path}")
                 # 可以选择跳过这个模型
                 # continue
            loaded_models[name] = model
            print(f"{name} 模型加载成功: {path}")
        except FileNotFoundError:
            print(f"警告: 模型文件未找到: {path}. 该模型将不可用。")
            all_loaded_successfully = False # 标记有问题，但不一定停止
        except Exception as e:
            print(f"加载 {name} 模型时出错: {e}. 该模型将不可用。")
            all_loaded_successfully = False

    if not loaded_models:
        print("错误：未能加载任何模型。无法进行预测。")
        return None
    # elif not all_loaded_successfully:
    #     print("警告：部分模型未能成功加载。")


    loaded_objects['models'] = loaded_models
    print("组件加载完成")
    return loaded_objects

def predict_news(news_text, loaded_components):
    """使用加载的模型和向量化器预测单条新闻的真伪"""
    if not loaded_components or 'vectorizer' not in loaded_components or 'models' not in loaded_components:
        print("错误：必需的组件（向量化器或模型）未加载，无法进行预测。")
        return

    vectorizer = loaded_components['vectorizer']
    models = loaded_components['models']

    if not models:
        print("错误：没有加载成功的模型可用于预测。")
        return

    if not news_text or not isinstance(news_text, str) or news_text.strip() == "":
        print("请输入有效的新闻文本。")
        return

    # 1. 清理输入文本
    cleaned_text = clean_text(news_text)
    if not cleaned_text:
        print("警告：清理后的文本为空，无法预测。")
        return

    # 2. 向量化文本 (使用加载的 vectorizer)
    try:
        # 再次检查向量化器是否为空
        if not hasattr(vectorizer, 'vocabulary_') or not vectorizer.vocabulary_:
             print("错误：向量化器为空，无法执行 transform。请确保 train.py 成功运行。")
             return
        vectorized_text = vectorizer.transform([cleaned_text])
    except ValueError as e:
         # 捕捉 'empty vocabulary' 等错误
         print(f"文本向量化时出错: {e}. 可能输入文本与训练数据差异过大或向量化器有问题。")
         return
    except Exception as e:
        print(f"文本向量化时发生未知错误: {e}")
        return

    # 3. 使用各个加载的模型进行预测
    print("\n新闻预测结果")
    has_prediction = False
    for name, model in models.items():
        try:
            # 检查模型是否真的加载了并且可用
            if model is None or not hasattr(model, 'predict'):
                 print(f"{name}: 模型未加载或无效，跳过预测。")
                 continue

            prediction = model.predict(vectorized_text)[0]
            # 4. 定义标签转换
            label = "真实新闻" if prediction == 1 else "虚假新闻"
            # 5. 打印结果
            print(f"{name}: {label}")
            has_prediction = True
        except AttributeError as e:
            # 特别捕捉 predict 方法不存在的情况 (虽然加载时已检查)
             print(f"使用 {name} 模型预测时出错: 模型对象可能已损坏或不完整 ({e})")
        except Exception as e:
            print(f"使用 {name} 模型预测时出错: {e}")

    if not has_prediction:
        print("未能从任何已加载的模型获得预测结果。")


# 主执行流程
if __name__ == "__main__":
    print("新闻真伪预测程序启动")

    # 1. 加载必要的组件
    components = load_components()

    if components and components.get('vectorizer') and components.get('models'):
        # 2. 进入手动测试模式
        print("\n进入手动测试模式")
        print("输入新闻文本进行预测，输入 '退出' 或 'exit' 来结束程序")
        while True:
            try:
                news_input = input("\n请输入新闻文本: ")
                # Strip input to handle empty lines or just spaces
                news_input_stripped = news_input.strip()
                if news_input_stripped.lower() == '退出' or news_input_stripped.lower() == 'exit':
                    print("程序结束")
                    break
                elif not news_input_stripped: # Handle empty input after stripping
                    print("输入不能为空，请输入新闻文本或 '退出' 或 'exit'")
                    continue
                predict_news(news_input, components) # Pass original input for cleaning later
            except EOFError: # 处理 Ctrl+D 或其他文件结束符
                print("\n检测到输入结束，程序退出。")
                break
            except KeyboardInterrupt: # 处理 Ctrl+C
                 print("\n操作中断，程序退出。")
                 break
            except Exception as e:
                print(f"\n处理输入时发生未知错误: {e}")
                print("请重试或输入 '退出' 或 'exit' 结束")
    else:
        print("\n必要组件加载失败，无法启动预测服务。请先运行 train.py 训练并保存模型")

    print("\n预测程序执行完毕")
