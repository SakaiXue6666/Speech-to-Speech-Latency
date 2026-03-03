# from transformers import AutoTokenizer

# tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-ASR-0.6B", trust_remote_code=True)

# text_512 = "嗨，我是丽娜。我将介绍我们的工作，即在西班牙语中检测和同化借用词，并对语料库进行注释以及建模方法。所以，我们将介绍什么是词汇借用，我们提出的数据集的任务，我们发布的一些模型以及我们探索的模型。但首先，什么是词汇借用？为什么它作为 NLP 任务很重要？嗯，词汇借用基本上是将一个语言中的词融入另一个语言中。例如，在西班牙语中，我们使用来自英语的词。这里有几个例子，比如博客、应用程序、在线众筹，这些都是我们有时在西班牙语中使用的英语词。词汇借用是一种语言借用，基本上是在一种语言中重现其他语言的模式。借用和代码切换有时被比较，并描述为一个连续体。好的，切换是双语者同时混合两种语言。然而，词汇借用和代码切换之间存在一些差异。我们将专注于词汇借用。代码切换是双语者所做的事情。根据定义，代码切换没有融入到任何使用的语言中，而词汇借用也是单语者所做的事情。借词会符合接收语言的语法，而且借词最终可以融入到接收语言中。那么，为什么借代是一个有趣的现象呢？嗯，从语言学的角度来看，借代是语言变化和互动的一种表现，而且词汇借代也是新词的来源。这里有一些词汇借代的例子，它们已经被纳入西班牙语作为新词。在 NLP 方面，借词是词汇外单词的常见来源。事实上，自动检测词汇借词已被证明对 NLP 下游任务有用，例如解析文本到语音合成或机器翻译。人们对英语对其他语言的影响越来越感兴趣，特别是与英语词汇借用有关。这些借用有时被称为英语外来词。这里有一些关于自动检测这些语言中借用词的工作的例子。因此，我们提出的一项任务是检测西班牙语新闻通讯中未同化的词汇借用。这意味着我们有兴趣提取从其他语言中借用的单词，这些单词正在西班牙报纸中使用，但尚未融入或同化到接收语言中，也就是说还没有融入西班牙语。这里有一个例子，这是一句西班牙语。Las prendes bests, jin sellers, the San Panke motel, florales, animal print, or tales, double patchwork。正如你所看到的，有三个文本片段实际上是英语单词，比如 bests, jin seller, animal print 和 patchwork。这些是我们感兴趣提取和检测的文本片段类型。嗯，它由一个用于西班牙语新闻通讯的英语词检测 CRF 模型组成。这个模型达到了八十六的 F1 分数，但数据集和建模方法都存在一些局限性。所以数据集专注于单一来源的新闻，只包含标题，而且训练集和测试集之间存在重叠的借用词，这阻止了评估该建模方法是否能推广到以前未见过的借用词。因此，我们的目标是解决任务中的一些局限性，所以首先我们创建了一个新的数据集。这个新数据集是标注了词汇借用的，目的是创建一个尽可能困难的测试集。因此，训练集和测试集之间在单词和主题上的重叠会最小化。结果就是测试集来自来源和日期，这些在训练集中没有出现过。在这里你可以看到实际上没有重叠，测试集也非常借用密集，只是给你一些数字。如果训练集每千个词包含六个借用词，那么测试集每千个词就包含二十个借用词。测试包含尽可能多的词汇。事实上，测试集中的九十二的借词是 OOV，因此在训练期间没有看到。语料库基本上由一系列文本组成，这些文本来自不同的西班牙报纸来源，它是由人工标注的，使用了两个标签，一个是英语词汇借用，这是西班牙语中大多数词汇借用，然后是其他标签用于其他语言的借用。我们使用了康莫格式，我们使用了 BIO 编码，这样我们就可以对单个令牌借用进行编码，例如应用程序或多个令牌借用，例如机器学习。这些是语料库的数字。如你所见，它大约有三十七万个标记。这里你有被标记为英语的跨度数量和被标记为其他借用的跨度数量，以及其中有多少是唯一的。这里有几个数据集的例子，例如你可以看到在第一个例子中，我们有借代批量烹饪，这是一个多词借词。我们使用 BIO 编码进行了标注，所以 B-I-O 被用于西班牙语中的单词，而不是那些没有被借用的单词。在这个第二个例子中，你有 bench 和 crash，它们也被标记为英语中的借词。所以一旦我们有了数据集，我们就探索了几个模型来完成提取和检测这些词汇借用的任务。我们尝试的第一个模型是条件随机场模型，这是之前工作中使用过的模型。我们使用了相同的手工制作功能。从这项工作中，你可以看到这些是二进制特征，后者是使用相同的 CRF 模型和相同特征在不同的数据集上获得的结果，也是用于西班牙语词汇借用检测的。所以这证明了我们创建的数据集更难，我们需要探索更复杂的模型来完成这项任务。因此，我们测试了两个基于变压器的模型。我们使用了 Bert，这是一个为西班牙语训练的单语 BERT 模型，以及多语种 BART。这两个模型我们都通过 Hugging Face 的 Transformers 库来使用。这是我们得到的结果。如你所见，多语种 Bert 在开发集和测试集上，以及所有指标上都比 BERT 表现更好。这样我们就可以比较一下，CRF 模型获得了 82，今年 F1 分数为 55，而多语言 Bert 的分数为 82，这是一个很大的差距。所以一旦我们有了这些结果，我们就问了自己另一个问题，就是我们能不能找到一个 BiLSTM- CRF 模型，用不同类型的嵌入进行训练，直接嵌入编码了不同类型的语言信息，然后在性能上超过基于 Transformer 的模型所获得的结果。因此，为了做到这一点，我们进行了一些初步实验。我们使用 Flair 库运行了这个 BiLSTM- CRF 模型。我们尝试并试验了不同类型的嵌入，比如基于 Transformer 的，还有 FastText 字符嵌入等等。我们发现基于转换器的嵌入比非上下文化嵌入表现更好。英语 Bert 和西班牙语 Bert 嵌入的组合优于多语言 Bert 嵌入，而 BPE 嵌入产生的 F1 值更好，字符嵌入则产生了更好的召回率。考虑到这一点，这些是我们得到的最佳性能结果。两个模型都是使用 Flair 的 BiLSTM- CRF 模型，一个使用了 Bert 和 BART 嵌入以及 BPE，另一个使用了 Bert、BART 嵌入、BPE 以及字符嵌入。最后一个模型在测试集上产生了最高的 F1 分数，尽管开发集上的最高分数是由没有字符嵌入的那个模型获得的。请记住，我们用多语言 Bert 得到的最佳结果是获得了 76 分，在开发方面，测试集上是 82，所以这是一个改进。与这些结果相比，最后我们问自己另一个问题：词汇借用检测可以被描述为迁移学习吗？从语言识别和代码切换。因此，我们经过微调的嵌入，基于变压器的嵌入，已经在西班牙语简英语部分的 LIMEs 代码切换数据集上进行了语言识别的预训练。LIMEs 是一个关于代码切换的数据集，其中有一个关于西班牙语简英语代码切换的部分。我们用代码切换嵌入，以及可选的字符嵌入、BPE 嵌入等来训练我们的 BERT-STM-CLF。我们得到的最佳结果是 84.22，这是我们尝试的所有模型中最高的。在测试集上，尽管我们在开发集上得到的最佳 F1 分数是使用未适应的嵌入，因此我们从工作中得出了一些结论。我们已经制作了一个新的西班牙语新闻数据集，其中标注了未同化的词汇借用。这个数据集比之前的四元更密集、更丰富。我们探索了四种用于词汇借用检测的模型。从错误分析的角度来看，召回率是所有模型的一个弱点。如你所见，一些常见的假音性包括大写的借用词，例如在英语和西班牙语中都存在的单词。此外，有趣的是，BPE 嵌入似乎提高了 F1 分数，而字符嵌入似乎提高了召回率。这是一个有趣的发现，也许我们可以在未来的工作中进一步探讨。这就是我所有的内容。非常感谢大家的聆听。"
# # [ASR DEBUG] batch_start=0 item=0 real_len=512 truncated=True last_token_id=3837
# # [ASR DEBUG] batch_start=0 item=1 real_len=512 truncated=True last_token_id=3837
# # [ASR DEBUG] batch_start=0 item=2 real_len=512 truncated=True last_token_id=97639
# # [ASR DEBUG] batch_start=0 item=3 real_len=278 truncated=False last_token_id=151645

# text_1024 = "嗨，我是丽娜。我将介绍我们的工作，即在西班牙语中检测和同化借用词，并对语料库进行注释以及建模方法。所以，我们将介绍什么是词汇借用，我们提出的数据集的任务，我们发布的一些模型以及我们探索的模型。但首先，什么是词汇借用？为什么它作为 NLP 任务很重要？嗯，词汇借用基本上是将一个语言中的词融入另一个语言中。例如，在西班牙语中，我们使用来自英语的词。这里有几个例子，比如博客、应用程序、在线众筹，这些都是我们有时在西班牙语中使用的英语词。词汇借用是一种语言借用，基本上是在一种语言中重现其他语言的模式。借用和代码切换有时被比较，并描述为一个连续体。好的，切换是双语者同时混合两种语言。然而，词汇借用和代码切换之间存在一些差异。我们将专注于词汇借用。代码切换是双语者所做的事情。根据定义，代码切换没有融入到任何使用的语言中，而词汇借用也是单语者所做的事情。借词会符合接收语言的语法，而且借词最终可以融入到接收语言中。那么，为什么借代是一个有趣的现象呢？嗯，从语言学的角度来看，借代是语言变化和互动的一种表现，而且词汇借代也是新词的来源。这里有一些词汇借代的例子，它们已经被纳入西班牙语作为新词。在 NLP 方面，借词是词汇外单词的常见来源。事实上，自动检测词汇借词已被证明对 NLP 下游任务有用，例如解析文本到语音合成或机器翻译。人们对英语对其他语言的影响越来越感兴趣，特别是与英语词汇借用有关。这些借用有时被称为英语外来词。这里有一些关于自动检测这些语言中借用词的工作的例子。因此，我们提出的一项任务是检测西班牙语新闻通讯中未同化的词汇借用。这意味着我们有兴趣提取从其他语言中借用的单词，这些单词正在西班牙报纸中使用，但尚未融入或同化到接收语言中，也就是说还没有融入西班牙语。这里有一个例子，这是一句西班牙语。Las prendes bests, jin sellers, the San Panke motel, florales, animal print, or tales, double patchwork。正如你所看到的，有三个文本片段实际上是英语单词，比如 bests, jin seller, animal print 和 patchwork。这些是我们感兴趣提取和检测的文本片段类型。嗯，之前已经有一些关于英语词检测的工作。它由一个用于西班牙语新闻通讯的英语词检测 CRF 模型组成。这个模型达到了八十六的 F1 分数，但数据集和建模方法都存在一些局限性。所以数据集专注于单一来源的新闻，只包含标题，而且训练集和测试集之间存在重叠的借用词，这阻止了评估该建模方法是否能推广到以前未见过的借用词。因此，我们的目标是解决任务中的一些局限性，所以首先我们创建了一个新的数据集。这个新数据集是标注了词汇借用的，目的是创建一个尽可能困难的测试集。因此，训练集和测试集之间在单词和主题上的重叠会最小化。结果就是测试集来自来源和日期，这些在训练集中没有出现过。在这里你可以看到实际上没有重叠，测试集也非常借用密集，只是给你一些数字。如果训练集每千个词包含六个借用词，那么测试集每千个词就包含二十个借用词。测试包含尽可能多的词汇。事实上，测试集中的九十二的借词是 OOV，因此在训练期间没有看到。语料库基本上由一系列文本组成，这些文本来自不同的西班牙报纸来源，它是由人工标注的，使用了两个标签，一个是英语词汇借用，这是西班牙语中大多数词汇借用，然后是其他标签用于其他语言的借用。我们使用了康莫格式，我们使用了 BIO 编码，这样我们就可以对单个令牌借用进行编码，例如应用程序或多个令牌借用，例如机器学习。这些是语料库的数字。如你所见，它大约有三十七万个标记。这里你有被标记为英语的跨度数量和被标记为其他借用的跨度数量，以及其中有多少是唯一的。这里有几个数据集的例子，例如你可以看到在第一个例子中，我们有借代批量烹饪，这是一个多词借词。我们使用 BIO 编码进行了标注，所以 B-I-O 被用于西班牙语中的单词，而不是那些没有被借用的单词。在这个第二个例子中，你有 bench 和 crash，它们也被标记为英语中的借词。所以一旦我们有了数据集，我们就探索了几个模型来完成提取和检测这些词汇借用的任务。我们尝试的第一个模型是条件随机场模型，这是之前工作中使用过的模型。我们使用了相同的手工制作功能。从这项工作中，你可以看到这些是二进制特征，例如单词或标记是否为大写，是否为小写，是否为引号等。这些都是人们在命名实体识别任务中所期望的特征类型。这些是我们得到的结果。我们使用手工制作的特征，获得了五十五的 F1 分数，这与报告的八十六的 F1 分数相比是一个巨大的差异。后者是使用相同的 CRF 模型和相同特征在不同的数据集上获得的结果，也是用于西班牙语词汇借用检测的。所以这证明了我们创建的数据集更难，我们需要探索更复杂的模型来完成这项任务。因此，我们测试了两个基于变压器的模型。我们使用了 Bert，这是一个为西班牙语训练的单语 BERT 模型，以及多语种 BART。这两个模型我们都通过 Hugging Face 的 Transformers 库来使用。这是我们得到的结果。如你所见，多语种 Bert 在开发集和测试集上，以及所有指标上都比 BERT 表现更好。这样我们就可以比较一下，CRF 模型获得了 82，今年 F1 分数为 55，而多语言 Bert 的分数为 82，这是一个很大的差距。所以一旦我们有了这些结果，我们就问了自己另一个问题，就是我们能不能找到一个 BiLSTM- CRF 模型，用不同类型的嵌入进行训练，直接嵌入编码了不同类型的语言信息，然后在性能上超过基于 Transformer 的模型所获得的结果。因此，为了做到这一点，我们进行了一些初步实验。我们使用 Flair 库运行了这个 BiLSTM- CRF 模型。我们尝试并试验了不同类型的嵌入，比如基于 Transformer 的，还有 FastText 字符嵌入等等。我们发现基于转换器的嵌入比非上下文化嵌入表现更好。英语 Bert 和西班牙语 Bert 嵌入的组合优于多语言 Bert 嵌入，而 BPE 嵌入产生的 F1 值更好，字符嵌入则产生了更好的召回率。考虑到这一点，这些是我们得到的最佳性能结果。两个模型都是使用 Flair 的 BiLSTM- CRF 模型，一个使用了 Bert 和 BART 嵌入以及 BPE，另一个使用了 Bert、BART 嵌入、BPE 以及字符嵌入。最后一个模型在测试集上产生了最高的 F1 分数，尽管开发集上的最高分数是由没有字符嵌入的那个模型获得的。请记住，我们用多语言 Bert 得到的最佳结果是获得了 76 分，在开发方面，测试集上是 82，所以这是一个改进。与这些结果相比，最后我们问自己另一个问题：词汇借用检测可以被描述为迁移学习吗？从语言识别和代码切换。因此，我们运行相同的 BiLSTM- CRF 模型。我们使用 Flair 运行，但不是使用这些未被采用的基于变压器的 Bert 和 Bert 嵌入。我们使用代码切换嵌入。什么是代码切换嵌入？嗯，这些是。经过微调的嵌入，基于变压器的嵌入，已经在西班牙语简英语部分的 LIMEs 代码切换数据集上进行了语言识别的预训练。LIMEs 是一个关于代码切换的数据集，其中有一个关于西班牙语简英语代码切换的部分。我们用代码切换嵌入，以及可选的字符嵌入、BPE 嵌入等来训练我们的 BERT-STM-CLF。我们得到的最佳结果是 84.22，这是我们尝试的所有模型中最高的。在测试集上，尽管我们在开发集上得到的最佳 F1 分数是使用未适应的嵌入，因此我们从工作中得出了一些结论。我们已经制作了一个新的西班牙语新闻数据集，其中标注了未同化的词汇借用。这个数据集比之前的四元更密集、更丰富。我们探索了四种用于词汇借用检测的模型。从错误分析的角度来看，召回率是所有模型的一个弱点。如你所见，一些常见的假音性包括大写的借用词，例如在英语和西班牙语中都存在的单词。此外，有趣的是，BPE 嵌入似乎提高了 F1 分数，而字符嵌入似乎提高了召回率。这是一个有趣的发现，也许我们可以在未来的工作中进一步探讨。这就是我所有的内容。非常感谢大家的聆听。"
# # [ASR DEBUG] batch_start=0 item=0 real_len=522 truncated=False last_token_id=151645
# # [ASR DEBUG] batch_start=0 item=1 real_len=585 truncated=False last_token_id=151645
# # [ASR DEBUG] batch_start=0 item=2 real_len=570 truncated=False last_token_id=151645
# # [ASR DEBUG] batch_start=0 item=3 real_len=278 truncated=False last_token_id=151645

# print("512: ", len(text_512))
# print("1024: ", len(text_1024))

# import difflib

# def show_char_forks(a: str, b: str, ctx: int = 30, max_show: int = 200):
#     """
#     打印 a 和 b 的所有差异片段（replace/insert/delete），并带上下文。
#     ctx: 每处差异左右各显示多少字符上下文
#     max_show: 最多打印多少处差异
#     """
#     sm = difflib.SequenceMatcher(a=a, b=b)
#     k = 0
#     for tag, i1, i2, j1, j2 in sm.get_opcodes():
#         if tag == "equal":
#             continue
#         k += 1
#         if k > max_show:
#             print(f"...(diffs > {max_show}, stop)")
#             break

#         a_seg = a[i1:i2]
#         b_seg = b[j1:j2]

#         a_left = a[max(0, i1-ctx):i1]
#         a_right = a[i2:i2+ctx]
#         b_left = b[max(0, j1-ctx):j1]
#         b_right = b[j2:j2+ctx]

#         print(f"\n=== #{k} {tag} ===")
#         print(f"a[{i1}:{i2}] -> {repr(a_seg)}")
#         print(f"b[{j1}:{j2}] -> {repr(b_seg)}")
#         print(f"a_ctx: {repr(a_left)} >>> {repr(a_seg)} <<< {repr(a_right)}")
#         print(f"b_ctx: {repr(b_left)} >>> {repr(b_seg)} <<< {repr(b_right)}")

# # 用法
# show_char_forks(text_512, text_1024, ctx=40, max_show=300)



# # # print("tokens:", len(tok.encode(text)))
# # if tok is not None:
# #     # 1) 看“token 片段”(可能是子词/带前缀符号，不一定是完整汉字)
# #     piece = tok.convert_ids_to_tokens([151645])[0]
# #     print("[ASR DEBUG] id=3837 token_piece:", repr(piece))

# #     # 2) 看“解码后的字符串”(更接近你想要的“字/字符”)
# #     text = tok.decode([151645], skip_special_tokens=False)
# #     print("[ASR DEBUG] id=3837 decoded:", repr(text))
# # else:
# #     print("[ASR DEBUG] tokenizer not found in processor")



# 输入 jsonl (ref & hyp)：
# src	(string)	该段的源语言文本（日语，一句或一段）
# tgt	(string)	该段的目标语言文本（中文）：在 ref 里是ref，在系统文件里是hyp
# sys_id	(string)	系统ID：ref_A 表示参考，GPT-4、Claude-3.5 等表示模型名
# doc_id	(string)	文档 ID，jsonl 中有很多 doc_id；同一篇长文档的所有 segment 共用同一个 doc_id
# seg_id	(int)	段ID，在同一 jsonl 内从 1 开始递增（第 1 条通常不是正文）；一个 segment 有一到多 sentences

# import json
# import yaml
# import os

# def get_jsonl(src_text, tgt_ref_text, ref_segments, instances):
#     with open(src_text, "r", encoding="utf-8") as f:
#         src_text_lines = [line.rstrip("\n") for line in f]

#     with open(tgt_ref_text, "r", encoding="utf-8") as f:
#         tgt_ref_text_lines = [line.rstrip("\n") for line in f]

#     with open(ref_segments, "r", encoding="utf-8") as f:
#         src_info_lines = yaml.safe_load(f)

#     src_for_hyp_docs = []
#     ref_dicts = []
#     curr_doc = None

#     for i in range(len(src_info_lines)):
#         ref_dict = {
#             "src": src_text_lines[i],
#             "tgt": tgt_ref_text_lines[i],
#             "sys_id": None,
#             "doc_id": src_info_lines[i]['wav'],
#             "seg_id": i + 1
#         }
#         ref_dicts.append(ref_dict)
#         # -----------------------------------------------------------
#         if src_info_lines[i]['wav'] != curr_doc:
#             src_for_hyp_docs.append({"doc_id": src_info_lines[i]['wav'], "src": ""})
#             curr_doc = src_info_lines[i]['wav']
#         src_for_hyp_docs[-1]["src"] += " " + src_text_lines[i]

#     print(src_for_hyp_docs)
#     # -----------------------------------------------------------
#     with open(instances, "r", encoding="utf-8") as f:
#         instances_lines = [line.rstrip("\n") for line in f]
    
#     hyp_dicts = []
#     for j in range(len(instances_lines)):
#         instance_dict = json.loads(instances_lines[j])
#         hyp_text = instance_dict["prediction_text"]
#         for src_for_hyp_doc in src_for_hyp_docs:
#             if src_for_hyp_doc["doc_id"] == os.path.basename(str(instance_dict["source"] or "")).replace("\\", "/"):
#                 hyp_dict = {
#                     "src": src_for_hyp_doc["src"],
#                     "tgt": hyp_text,
#                     "sys_id": None,
#                     "doc_id": src_for_hyp_doc["doc_id"],
#                     "seg_id": j + 1
#                 }
#                 hyp_dicts.append(hyp_dict)
#                 break

#     return ref_dicts, hyp_dicts

# ref_dicts, hyp_dicts = get_jsonl(
#     "data/input/ACL.6060.dev.en-xx.en.txt", 
#     "data/input/ACL.6060.dev.en-xx.zh.txt", 
#     "data/input/ACL.ACLdev2023.en-xx.gold_segments.yaml", 
#     "data/output_qwen_asr/instances.log")

# # print(ref_dicts)
# print(hyp_dicts)


# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import nagisa
# import torch
# from qwen_asr.core.transformers_backend import (
#     Qwen3ASRConfig,
#     Qwen3ASRForConditionalGeneration,
#     Qwen3ASRProcessor,
# )
# from transformers import AutoConfig, AutoModel, AutoProcessor

# from .utils import (
#     AudioLike,
#     ensure_list,
#     normalize_audios,
# )


class Qwen3ForceAlignTokenizer():
    def __init__(self):
        ko_dict_path = os.path.join(os.path.dirname(__file__), "assets", "korean_dict_jieba.dict")
        ko_scores = {}
        with open(ko_dict_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                word = line.split()[0]
                ko_scores[word] = 1.0
        self.ko_score = ko_scores
        self.ko_tokenizer = None

    def is_kept_char(self, ch: str) -> bool:
        if ch == "'":
            return True
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            return True
        return False

    def clean_token(self, token: str) -> str:
        return "".join(ch for ch in token if self.is_kept_char(ch))

    def is_cjk_char(self, ch: str) -> bool:
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF   # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF  # Extension A
            or 0x20000 <= code <= 0x2A6DF  # Extension B
            or 0x2A700 <= code <= 0x2B73F  # Extension C
            or 0x2B740 <= code <= 0x2B81F  # Extension D
            or 0x2B820 <= code <= 0x2CEAF  # Extension E
            or 0xF900 <= code <= 0xFAFF    # Compatibility Ideographs
        )

    def tokenize_chinese_mixed(self, text: str) -> List[str]:
        tokens: List[str] = []
        current_latin: List[str] = []

        def flush_latin():
            nonlocal current_latin
            if current_latin:
                token = "".join(current_latin)
                cleaned = self.clean_token(token)
                if cleaned:
                    tokens.append(cleaned)
                current_latin = []

        for ch in text:
            if self.is_cjk_char(ch):
                flush_latin()
                tokens.append(ch)
            else:
                if self.is_kept_char(ch):
                    current_latin.append(ch)
                else:
                    flush_latin()

        flush_latin()

        return tokens

    def tokenize_japanese(self, text: str) -> List[str]:
        words = nagisa.tagging(text).words
        tokens: List[str] = []
        for w in words:
            cleaned = self.clean_token(w)
            if cleaned:
                tokens.append(cleaned)
        return tokens

    def tokenize_korean(self, ko_tokenizer, text: str) -> List[str]:
        raw_tokens = ko_tokenizer.tokenize(text)
        tokens: List[str] = []
        for w in raw_tokens:
            w_clean = self.clean_token(w)
            if w_clean:
                tokens.append(w_clean)
        return tokens

    def split_segment_with_chinese(self, seg: str) -> List[str]:
        tokens: List[str] = []
        buf: List[str] = []

        def flush_buf():
            nonlocal buf
            if buf:
                tokens.append("".join(buf))
                buf = []

        for ch in seg:
            if self.is_cjk_char(ch):
                flush_buf()
                tokens.append(ch)
            else:
                buf.append(ch)

        flush_buf()
        return tokens

    def tokenize_space_lang(self, text: str) -> List[str]:
        tokens: List[str] = []
        for seg in text.split():
            cleaned = self.clean_token(seg)
            if cleaned:
                tokens.extend(self.split_segment_with_chinese(cleaned))
        return tokens

    def encode_timestamp(self, text: str, language: str) -> List[str]:
        language = language.lower()

        if language.lower() == "japanese":
            word_list = self.tokenize_japanese(text)
        elif language.lower() == "korean":
            if self.ko_tokenizer is None:
                from soynlp.tokenizer import LTokenizer
                self.ko_tokenizer = LTokenizer(scores=self.ko_score)
            word_list = self.tokenize_korean(self.ko_tokenizer, text)
        else:
            word_list = self.tokenize_space_lang(text)
        
        return word_list
    

tok = Qwen3ForceAlignTokenizer()

text = "嗨，我是丽娜。 我将介绍我们的工作，即在西班牙语中检测和同化借用词，并对语料库进行注释以及建模方法。"
text = "我们得到的最佳结果是 84.22，这是我们尝试的所有模型中最高的"
tokens = tok.encode_timestamp(text, "Chinese")   # 或 "English"
print(tokens)