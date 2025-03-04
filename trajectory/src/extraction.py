import json  
import re

def substitude(string):
    return re.sub(r'\\', r'\\\\', string)  

def jsonl_to_json_list(file_path):  
    json_list = []  
    current_json = ""  
    open_braces = 0  # 计数当前打开的花括号数量  

    with open(file_path, 'r', encoding='utf-8') as file:  
        for line in file:  
            # 清理行中的空白  
            line = line.strip()  
            if not line:  
                continue  # 跳过空行  
            line = substitude(line)
            current_json += line  # 累加当前行内容  
            open_braces += line.count('{')  # 统计打开的花括号  
            open_braces -= line.count('}')  # 统计关闭的花括号  

            # 如果当前对象的花括号匹配，表示一个完整的 JSON 对象已经构建完成  
            if open_braces == 0:  
                # 尝试解析 JSON 对象  
                try:  
                    json_object = json.loads(current_json)  
                    json_list.append(json_object)  # 添加到列表中  
                except json.JSONDecodeError as e:  
                    print(f"解析错误: {e}，内容: {current_json}")  # 错误处理  
                current_json = ""  # 重置当前 JSON 字符串  

    return json_list  

# 使用示例  
if __name__ == "__main__":  
    file_path = '../data/results/r1_result.jsonl'  # 替换为你的 JSONL 文件路径  
    json_list = jsonl_to_json_list(file_path)  
    for index, dic in enumerate(json_list):
        tmp_dic = {}
        if "TrajectoryDesign" in dic:
            tmp_dic["predecessor_work"] = dic["TrajectoryDesign"]["predecessor_work"]
            tmp_dic["objective"] = dic["TrajectoryDesign"]["objective"]
            tmp_dic["name"] = dic["TrajectoryDesign"]["name"]
        else:
            tmp_dic["predecessor_work"] = dic["predecessor_work"]
            tmp_dic["objective"] = dic["objective"]
            tmp_dic["name"] = dic["name"]

        json_list[index] = tmp_dic
    out_path = "../data/test/r1_test.jsonl"
    with open(out_path, "w") as f:
        for dic in json_list:
            dic = json.dumps(dic)
            f.write(dic + '\n')