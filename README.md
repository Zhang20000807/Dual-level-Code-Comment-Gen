# DLCoG: Dual-Level Code Comment Generation based on In-Context Learning

## Abstract

Code comments that describe the purpose of the code mainly include method comments and snippet comments. Method comments provide a brief description of the method's functionality, helping development team members quickly understand the method's purpose and interface. Snippet comments, on the other hand, detail the internal implementation of the method, helping developers understand the specific role of each code snippet, thus facilitating subsequent maintenance and modifications. 

We manually constructed a high-quality 6.9k Java dataset of <Method, Method Comment, \<Snippet, Snippet Comment>*> based on some open-source Java projects(**raw_dataset/manual_dataset.jsonl**). This dataset was used to train a classification model for identifying "code summaries" in comments, as well as an association model for finding the code corresponding to the comments. A total of 84k multi-level code annotation datasets were built in a larger range of open source projects(**raw_dataset/expansion_dataset.jsonl**).

## Manual Dataset

The original dataset can be found at https://huggingface.co/datasets/bigcode/the-stack/tree/main/data/java

The format of the dual-level code comment data is as follows:

```json
{
	id:0
	repo:"repo_owner/repo_name"
  path:"root/docA/.../file.java"
  raw_code:"public int fun_name(int arg){...}"
  code_summary:"This function make ..."
  code_snippets:[
  	{
  		sub_id:0
  		code_snippet:"int[] sublist = ..."
  		code_summary:"make a sublist ..."
  		place:(7,9) # Located on lines 7 to 9 of the source code
		},
		...
  ]
}
```

We recommend that you conduct research on this manual dataset because it is extracted by human annotations and is of higher quality than the expanded dataset.

The code to expand the dataset can be found in the **expand_dataset** folder.



## Dual-Level Code Comment Generation

You can perform code similarity search, build prompts, and use LLM to get two-level code comments according to the following process.
Encoding your code:

```
python S1_sentence_bert_encode.py
```

Get the similarit data for your code:

```
python s2_get_similarity_data.py
```

Build prompt. The prompt example is shown in the following figure:

```
python s3_build_prompt.py
```

Eval the comment by LLM:

```
python s4_eval.py
```

![image-20240803175709976](/Users/zhangzhiyang/Library/Application Support/typora-user-images/image-20240803175709976.png)

