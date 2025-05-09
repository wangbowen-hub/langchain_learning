from typing import List
from collections import deque
def add_window_to_document(
    document: List[dict], window_steps: int, window_size: int, window_index_name: str
):
    """在文档列表的每个文档中添加窗口化信息。

    参数:
        document (Document): 文档对象列表。
        window_steps (int): 窗口化的步长。
        window_size (int): 每个窗口的大小。
        window_index_name (str): 用于存储窗口索引的元数据键。
    """
    window_id = 0
    window_queue = deque()

    for index, doc in enumerate(document):
        if index % window_steps == 0 and index != 0 and index < len(document) - window_size:
            window_id += 1
        window_queue.append(window_id)

        if len(window_queue) > window_size:
            for _ in range(window_steps):
                window_queue.popleft()

        window = set(window_queue)
        doc[f"{window_index_name}_lower_bound"] = min(window)
        doc[f"{window_index_name}_upper_bound"] = max(window)

docs=[{},{},{},{},{},{},{},{},{},{}]
add_window_to_document(docs,2,4,"index_name")
print(docs)