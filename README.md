## 需要自行设置 API key 和 Base Url
## 命令行参数说明

| 参数         | 选项    | 默认值 |  说明   |
|---------|---------|--------| -- |
| --model -m  | gpt/r1  | gpt | 表示使用的输入数据 是使用哪个模型生成的 |
| --type -t | read_paper/idea_gen | read_paper | 表示两种工作模式，从论文中总结信息生成输入数据，或通过输入数据生成工作轨迹 |
| --index -i | 非负整数 | 0 | 表示选取数据的索引，主要参考可测试数据的条数，以及待读论文的数目 | |
