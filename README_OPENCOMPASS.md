---
name: social_media_sentiment_analysis
desc: 从雪球投资平台上筛选的500条带情感标签的言论数据 
language:
- cn
dimension:
- understanding
sub_dimension:
- sentiment classification
website: https://github.com/Hoder-zyf/InternLM/blob/main/processed_data.json
github: https://github.com/Hoder-zyf/InternLM/blob/main/processed_data.json
paper: 
release_date: 2024-04-24
tag:
- text

download_url: https://github.com/Hoder-zyf/InternLM/blob/main/processed_data.json
---
## Introduction
从雪球投资平台上筛选的500条带情感标签的言论数据
## Meta Data
## Example
```json
{
        "id": "0",
        "question": "你是一个资讯情绪识别助手。请判断以下内容传达的情绪属于【积极、消极、中性】中的哪一类。请给出正确选项。\n个人评论： 这是个好行业，业绩做的不行就没话说了$三元股份(SH600429)$//@初善君:回复@麦客oo9:光明乳业这些年就证实了一件事，自己不行 ； 转发内容： $光明乳业(SH600597)$ 年报出来了，猛地一看，-40%的增长，吓尿。但细看发现，光明正在变好。（1）看营收依然200多亿，几乎持平。（2）新董事长上任一口气计提1.5个亿，加上这个光明18年是盈利5个亿的，负的并没有太多。轻装上阵了。（3）对于消费股，我只看营收，有营收就有保障，低市销率值得期待。（4）增值率下降，200亿会增厚6000万利润。（5）消费升级，国企混改，产业兴旺。",
        "A": "积极",
        "B": "消极",
        "C": "中性",
        "answer": "C"
    }
```
## Citation