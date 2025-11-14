import os
import csv

def append_to_csv(file_path, data=None, new_line=False):
    """
    向CSV文件追加数据，可选择是否换行
    
    参数:
        file_path (str): CSV文件路径
        data: 要添加的数据
        new_line (bool): 是否在添加数据后换行，默认为False
    """

    # 检查文件是否存在
    file_exists = os.path.exists(file_path)
    
    # 如果文件存在，读取所有内容
    rows = []
    if file_exists:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
    
    # 处理数据添加
    if rows:
        if data:
            rows[-1].append(str(data))
        if new_line:
            rows.append([])
    else:
        if data:
            rows = [[str(data)]]
        if new_line:
            rows.append([])
    
    # 将处理后的数据写回文件
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    
def sum_up_lines(file_path, last_lines_num: int):
    """
    对CSV文件中的每行进行求和
    
    参数:
        file_path (str): CSV文件路径
    """

    # 检查文件是否存在
    file_exists = os.path.exists(file_path)
    
    # 如果文件存在，读取所有内容
    rows = []
    if file_exists:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
    else:
        return
    
    res = []
    prior_rows = rows[:-last_lines_num-1]
    rows = rows[-last_lines_num-1:]
    for row in rows:
        if not row:
            continue
        f_row = [float(i) for i in row]
        total = sum(f_row)
        res.append([total])
    
    # 将处理后的数据写回文件
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(prior_rows + res)


# 使用示例
if __name__ == "__main__":
    # csv_path = "/home/austin/tools/utils/stats/test.csv"
    
    # append_to_csv(csv_path, data="数据1")
    # append_to_csv(csv_path, data="数据2")
    # append_to_csv(csv_path, data="数据3")

    # append_to_csv(csv_path, new_line=True)
    
    # append_to_csv(csv_path, data="数据4")
    # append_to_csv(csv_path, data="数据5")

    sum_up_lines("/home/austin/tools/utils/stats/SwiftTransformer_Metrics_copy.csv")
