from langchain_core.example_selectors import BaseExampleSelector
from typing import Dict, List
import numpy as np

class CustomExampleSelector(BaseExampleSelector):

    def __init__(self, examples: List[Dict[str, str]]):
        self.examples = examples

    def add_example(self, example: Dict[str, str]) -> None:
        """添加新的例子"""
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """根据输入选择使用哪些例子"""
        return np.random.choice(self.examples, size=2, replace=False)

if __name__ == "__main__":
    examples = [
        {"foo": "1"},
        {"foo": 2},
        {"foo": "3"}
    ]
    # 初始化示例选择器
    example_selector = CustomExampleSelector(examples)

    # 选择例子
    print(example_selector.select_examples({"foo": "foo"}))

    # 添加新的例子
    example_selector.add_example({"foo": "4"})
   

    # 选择例子
    example_selector.select_examples({"foo": "foo"})
