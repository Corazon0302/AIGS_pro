from prompt.paper_read import *
from prompt.idea_gen import *
import json
import os
import openai
import argparse
from datetime import datetime


def main(args):
    if(args.type == "paper_read"):
    # Load the content of the paper and generate prompt
        dir_path = "/disks/disk6/private/lzh/AIGS_pro/trajectory/data/paper_md"
        files = os.listdir(dir_path)
        file_path = dir_path + "/" + files[args.index]
        with open(file_path, "r") as f:
            content = f.read()
        prompt_content = paper_read.render(paper=content)
    elif(args.type == "idea_gen"):
        file_path = f"/disks/disk6/private/lzh/AIGS_pro/trajectory/data/test/{args.model}_test.jsonl"
        contents = []
        with open(file_path, "r") as f:
            for line in f.readlines():
                line = json.loads(line)
                contents.append(line)
        item = contents[args.index]
        name = item.pop("name", None)
        prompt_content = idea_gen.render(data=json.dumps(item))

    client = openai.OpenAI()
    system_message = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question."
    response = client.chat.completions.create(
        model = "gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_content},
        ],
        temperature=0.7, max_tokens=4096, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    # print(content)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if(args.type == "paper_read"):
        with open("data/results/gpt_result.jsonl", 'a') as f:
            f.write(content + "\n")
    elif(args.type == "idea_gen"):
        content = json.loads(content)
        content["name"] = name
        content = json.dumps(content)
        with open(f"results/{args.model}_result_{current_time}.json", 'a') as f:
            f.write(content + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--type", type=str, default="paper_read")
    parser.add_argument("-i","--index", type=int, default=0)
    parser.add_argument("-m","--model", type=str, default="gpt")
    args = parser.parse_args()
    main(args)
