import json
import os
import unicodedata
from typing import Dict, List, Tuple


def is_kept_char(ch: str) -> bool:
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def norm_char_stream_with_mapping(text: str) -> Tuple[str, List[int]]:
    """
    把原文 text 转成规范化字符流，并记录：
    norm_text[i] 对应原文 raw_text 的哪个字符下标。

    返回:
        norm_text, norm_to_raw
    """
    raw = unicodedata.normalize("NFKC", text or "")
    norm_chars: List[str] = []
    norm_to_raw: List[int] = []

    for raw_idx, ch in enumerate(raw):
        if is_kept_char(ch):
            norm_chars.append(ch.lower())
            norm_to_raw.append(raw_idx)

    return "".join(norm_chars), norm_to_raw


def norm_unit_text(text: str) -> str:
    raw = unicodedata.normalize("NFKC", text or "")
    return "".join(ch.lower() for ch in raw if is_kept_char(ch))


def add_char_spans_to_asr_json(infile: str, outfile: str = None) -> str:
    context_chars = 80  ###

    with open(infile, "r", encoding="utf-8") as f:
        data = json.load(f)

    full_text = unicodedata.normalize("NFKC", data.get("text", "") or "")
    ts_list: List[Dict] = data.get("time_stamps", []) or []

    full_norm, norm_to_raw = norm_char_stream_with_mapping(full_text)

    norm_cursor = 0
    new_ts_list: List[Dict] = []

    # ++++++++++++++++++++++++++++++++++++++++++++++
    last_ok_idx = -1
    last_ok_raw_span = (-1, -1)
    last_ok_unit_text = ""

    basename = os.path.basename(infile)
    # ++++++++++++++++++++++++++++++++++++++++++++++

    for idx, ts in enumerate(ts_list):
        unit_text = ts.get("text", "") or ""
        unit_norm = norm_unit_text(unit_text)

        new_ts = dict(ts)

        if not unit_norm:
            new_ts["char_start"] = -1
            new_ts["char_end"] = -1
            new_ts_list.append(new_ts)
            # ++++++++++++++++++++++++++++++++++++++++++++++
            print(
                    f"[empty-unit] file={basename} idx={idx} "
                    f"unit_text={unit_text!r}"
                )
            # ++++++++++++++++++++++++++++++++++++++++++++++
            continue

        pos = full_norm.find(unit_norm, norm_cursor)

        if pos == -1:
            new_ts["char_start"] = -1
            new_ts["char_end"] = -1
            new_ts_list.append(new_ts)
            # ++++++++++++++++++++++++++++++++++++++++++++++
            # norm 全文上下文
            norm_left = max(0, norm_cursor - context_chars)
            norm_right = min(len(full_norm), norm_cursor + context_chars)
            norm_ctx = full_norm[norm_left:norm_right]

            # raw 全文上下文（通过 norm_cursor 尽量映射到 raw）
            if 0 <= norm_cursor < len(norm_to_raw):
                raw_cursor = norm_to_raw[norm_cursor]
            elif norm_to_raw:
                raw_cursor = norm_to_raw[-1] + 1
            else:
                raw_cursor = 0

            raw_left = max(0, raw_cursor - context_chars)
            raw_right = min(len(full_text), raw_cursor + context_chars)
            raw_ctx = full_text[raw_left:raw_right]

            print("=" * 80)
            print(f"[match-fail] file={basename} idx={idx}")
            print(f"unit_text      = {unit_text!r}")
            print(f"unit_norm      = {unit_norm!r}")
            print(f"norm_cursor    = {norm_cursor}")
            print(f"last_ok_idx    = {last_ok_idx}")
            print(f"last_ok_unit   = {last_ok_unit_text!r}")
            print(f"last_ok_span   = {last_ok_raw_span}")
            print(f"norm_ctx       = {_clip(norm_ctx, 200)!r}")
            print(f"raw_ctx        = {_clip(raw_ctx, 200)!r}")

            # 再给一个“从头搜”的参考，看看是不是 cursor 走偏了
            global_pos = full_norm.find(unit_norm)
            print(f"global_find_pos= {global_pos}")

            if global_pos != -1:
                g_start = norm_to_raw[global_pos]
                g_end = norm_to_raw[global_pos + len(unit_norm) - 1] + 1
                print(f"global_raw_span= ({g_start}, {g_end})")
                print(f"global_raw_txt = {_clip(full_text[g_start:g_end], 200)!r}")

            print("=" * 80)
            # ++++++++++++++++++++++++++++++++++++++++++++++
            continue

        norm_start = pos
        norm_end = pos + len(unit_norm)   # 开区间

        raw_start = norm_to_raw[norm_start]
        raw_end = norm_to_raw[norm_end - 1] + 1   # 开区间

        new_ts["char_start"] = raw_start
        new_ts["char_end"] = raw_end
        new_ts_list.append(new_ts)

        # ++++++++++++++++++++++++++++++++++++++++++++++
        last_ok_idx = idx
        last_ok_raw_span = (raw_start, raw_end)
        last_ok_unit_text = unit_text
        # ++++++++++++++++++++++++++++++++++++++++++++++

        norm_cursor = norm_end

    data["text"] = full_text
    data["time_stamps"] = new_ts_list

    if outfile is None:
        base, ext = os.path.splitext(infile)
        outfile = base + "_charspan" + ext

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return outfile

def add_char_spans_for_dir(asr_dir: str):
    for fn in os.listdir(asr_dir):
        if not fn.endswith("_asr.json"):
            continue
        infile = os.path.join(asr_dir, fn)
        out = add_char_spans_to_asr_json(infile)
        print("saved:", out)


# add_char_spans_to_asr_json(
#     "data/output_qwen_asr6/2022.acl-long.117_tgt_asr.json",
#     "try2.json"
# )

text = "Hi,我是叶莲娜。我将展示我们的工作:检测西班牙语中的为同化借用词,以及带注释的语料库和建模方法。我们将介绍什么是词汇借用,我们提出的任务,我们发布的数据集,以及我们探索的一些模型。但首先,什么是词汇借用,以及为什么它作为NLP任务很重要?词汇借用是基本上是从一种语言中融入单词到另一种语言。例如,在西班牙语中,我们使用来自英语的单词。这里有几个例子:上播客应用程序,在线众筹,所有这些都是英语单词,我们有时在西班牙语中使用。词汇借用是语言借用的一种类型,那基本上是在一种语言中重现其他语言的模式。借用和代码转换有时被比较并描述为一个连续体。代码转换是双语者所做的事情,他们同时混合使用两种语言。然而,词汇借用和代码转换之间存在一些差异。我们将重点关注词汇借用。代码转换是双语者的行为。根据定义,代码转换并未融入任何使用的语言中,而词汇借用也是单语者的行为。借用的词汇会符合目标语言的语法。借用的词汇最终可以融入目标语言。为什么借用是一个有趣的现象呢?从语言学的角度来看,借用是语言变化和相互作用的一种表现。此外,词汇借用是新词的一个来源。这里有一些词汇借用的例子,它们已作为新词融入西班牙语。在NLP中,借用是生词的常见来源。事实上,自动检测词汇借用已被证明对NLP下游任务有用,如解析文本、语音、句子或机器翻译。人们对英语对其他语言的影响越来越感兴趣,特别是与英语词汇借用有关的方面,有时被称为英语化现象。这里有一些关于自动检测借用的研究示例,涉及其中一些语言。我们提出的任务是。检测西班牙语新闻稿中的模拟借,这意味着我们关注从所有语言中提取借用的单词,这些语言中的西班牙报纸中使用,但还没有被整合或同化到目标语言中。所以这些单词还没有融入西班牙语。这里有一个例子,这是一句西班牙语句子。如你所见,有三个文本跨度,实际上是英语单词,如畅销书、动物印花和评估。这些是我们感兴趣的跨度类型,我们正在提取和检测它们。之前有关于英语外来词检测的研究,包括一个条件随机场模型,用于西班牙语新闻专线中的英语外来词检测。这个模型的F得分是86,但在数据集和建模方法上都存在一些局限性。数据集只关注一个新闻来源,只包含标题,而且训练集中出现的借用词也有重叠,测试集也有重叠,这妨碍了对建模方法是否能真正泛化的评估,使其能应用于此前未见的借用词。我们的目标是解决任务中的一些局限性。首先,我们创建了一个新的数据集。这个新数据集的目标是用词汇借用进行标注,目的是创建一个尽可能难的测试集。这样在词汇和主题方面就会有最小的重叠,在训练集和测试集之间,测试集来自于与训练集不同的来源和日期,在训练集中没有出现过。这里可以看到没有重叠,测试集也有很多借用的情况。给你一些数字,如果训练集包含六个借用,每千个token有六个借用,测试集每千个token有二十个借用,测试集包含尽可能多的生词。事实上,测试集中92%的借用词是O,所以在训练期间没有见过。语料库主要由来自不同来源的西班牙语报纸文本组成,它是手动注释,使用两个标签,一个用于英语词汇边界,这是西班牙语中大多数的词汇边界,另一个标签是其他用于其他语言的借词。我们使用GONO标签格式,我们使用BIO编码。这样我们就可以对单个词素的介词进行编码,比如像 f 这样的介词,或者对多词素的介词进行编码,比如机器学习。这些是语料库的数量。如你所见,大约有 31.7 万个 token,这里是跨度的数量,被标记为英语的跨度,以及被标记为其他借用语言的跨度,以及其中有多少是独特。这里有几个例子,这是数据集的集合。例如,在第一个例子中,我们有借用表达 batch cooking,这是一个多词借用。我们用 BIO 编码进行了标注。BIO 用于西班牙语中的单词,而不是用于非借用的单词。在第二个例子中,有 batch crush,它们也被标记为从英语借用的。一旦我们有了数据集,我们就探索了几个用于提取和检测这些词汇借用的模型。我们尝试的第一个是条件随机场模型,这是之前工作中使用过的模型。我们使用了相同的手工特征来自那项工作。如你所见,这些是特征,这些是二元特征,例如单词或大写标记,是否为标题大小写,是否为引号诸如词类。这些都是命名实体识别任务中预期的特征类型。这些是我们得到的结果。我们获得了 55 的 F1 平分,使用带有手工特征的 CRF 模型,这是一个巨大的差异,与报告的 F6 的 F1 平分相比。这是通过相同的 CRF 模型和相同的特征得到的结果,但在不同的数据集上,对于西班牙语检测墨西哥借用词,这证明了我们创建的数据集比我们的数据集更难。我们需要探索更复杂的模型来完成这项任务。所以我们测试了两个基于 Transformer 的模型。我们使用了 Bert,这是一个为西班牙语训练的单语 Bert 模型,还有多语言 Bert。这两个模型我们都是通过 Hugging Face Transformer 库使用的。这是我们得到的结果。如你所见,多语言 Bert 的表现比 Bert 更好,无论是在开发集还是测试集上,以及所有指标上都是如此。这样我们就能了解所得到的 CRF 模型与 82 和 55 的 CRF 模型相比,获得了 55 的 F1 分数,而多语言 BERT 获得了 82,这是一个相当大的差距。一旦我们有了这些结果,我们就会问自己另一个问题,即我们能否找到一个使用不同类型嵌入的双向 LSTM-CRF 模型,这些嵌入编码了不同类型的语言信息,并且比基于 Transformer 的模型表现更好。为了实现这一点,我们进行了一些初步实验。我们运行了这个双向 LSTM-CRF 模型,使用 Flair 库,并尝试了不同类型的嵌入,比如基于 Transformer 的嵌入,还有 FastText 字符嵌入等等。我们发现基于 Transformer 的嵌入比非上下文嵌入表现更好。英语 BERT 和西班牙语 BERT 嵌入的组合优于多语言 BERT 嵌入。BP 嵌入产生了更好的 F1 和字符嵌入。考虑到这一点,召回率也更高。这些是我们得到的最佳结果。两个模型都是双向 LSTM-CRF 模型,使用 Flair,一个输入 BERT 和 BERT 嵌入以及 BP,另一个输入 BERT 和 BERT 嵌入、BP 以及字符嵌入。最后一个在测试集上产生了最高的 F1 分数,尽管开发集上的最高分数是另一个模型得到的,是没有字符嵌入的模型。要记住,我们用多语言 BERT 取得的最佳结果,在开发集上获得了 76 的 F1 分数,在测试集上获得了 82 的分数。与那些结果相比,这是一种改进。最后,我们问了自己另一个问题,即词汇借用检测是否可以被定义为代码转换中语言识别的迁移学习。因此,我们运行了相同的双向 LSTM-CRF 模型,该模型曾用于 Flair,但没有使用这些调整后的基于 Transformer 的 BERT 和 BERT 嵌入。我们使用代码转换嵌入。什么是代码转换嵌入?这些是嵌入,已经进行了微调,是在 lint 代码转换数据集的西班牙语英语部分上针对语言识别进行了预训练的嵌入。Lint 是一个类似代码转换的数据集,其中有西班牙语英语部分。这是西班牙语英语代码转换。我们把 LSTM-CRF 输入进去,加入代码转换嵌入。以及可选的字符嵌入、BPE嵌入等等,我们得到的最佳结果是84.22,这是我们尝试过的所有模型中最高。在测试集上,尽管我们在开发集上得到的最佳分数是79,但低于双LSTM-CRF获得的最佳结果。该模型使用了适配的嵌入。我们的工作得出了一些结论。我们制作了一个新数据集,其中西班牙新闻段落已标注了同化的词汇借用。这个数据集的借用事件比OV覆盖的还要多,比我们之前探索的资源还要多。我们探索了四种模型用于词汇借用检测。在错误分析方面,召回率是所有模型的薄弱点。如你所见,一些常见的漏判包括大小写借用及在英语和西班牙语中都存在的单词,例如还有有趣的事儿。BPE嵌入似乎提高了F1分数,字符嵌入似乎提高了召回率。有趣的是,也许我们可以在未来的工作中进行探索。这就是我要分享的所有内容。非常感谢大家的聆听。"
# text = "Hi, this is Elena, and I'm going to present our work detecting unassimilated borrowings in Spanish and annotated corpus approaches to modeling. We're going to cover what lexical borrowing is, the tasks that we propose, the data set that we have released, and some models that we explored. But to begin with, what is lexical borrowing, and why it matters as an NLP task? Lexical borrowing is basically the incorporation of words from one language into another language. For instance, in Spanish, we use words that come from English, and here you have a few examples: words such as podcast, app, online, crowdfunding. All these are English words that we sometimes use in Spanish. Lexical borrowing is a type of linguistic borrowing, which is basically reproducing in one language patterns of other languages. Borrowing and code switching have sometimes been compared and described as a continuum. Code switching is a thing that bilinguals do, where they mix two languages at the same time. However, there are some differences between lexical borrowing and code switching. We're going to focus on lexical borrowing. Code switching is something that is done by by definition. The code switches are not integrated into any of the languages in use, whereas lexical borrowing is something that is also done by monolinguals. The borrowings will comply with the grammar of the recipient language, and borrowings can eventually be integrated into the recipient language. So why is borrowing an interesting phenomenon? Well, from the point of view of linguistics, borrowing is a manifestation of how languages change and how they interact, and also lexical borrowings are sources of new words. Here you have some examples of lexical borrowings that have been incorporated into the Spanish language as new words in terms of NLP. Borrowings are a common source of out-of-vocabulary words, and in fact, automatically detecting lexical borrowing has been proven to be useful for NLP downstream tasks such as parsing texts, split sentences, or machine translation. There has been a growing interest in the influence of English on other languages, particularly related to English lexical borrowings, which sometimes have been called anglicisms. Here you have some examples of work on automatic detection of borrowings. In some of these languages, the tasks that we propose.Is to detect assimilated lexical borrowings in Spanish news, right? Which means that we are interested in extracting words borrowed from all the languages that are being used in Spanish newspapers, but that have not been integrated or assimilated into the recipient language, so not yet integrated into Spanish. Here you have an example. This is a sentence in Spanish. As you can see, there are three spans of text which are actually English words, like bestseller, animal print, and patchwork. These are the type of spans that we are interested in. We are extracting and detecting them. There has been previous work on anglicism detection. It consisted of a CRF model for anglicism detection on Spanish news wire. This model achieved an F1 score of 86, but there were some limitations both in the dataset and the modeling approach. The dataset focused exclusively on one source of news, consisted only of headlines, and also there was an overlap in the borrowings that appear in the training set and the test set. So this prevented the assessment of whether the modeling approach could actually generalize to previously unseen borrowings. What we aim is to tackle some of these limitations. In the task, to begin with, we created a new dataset. The aim of the new dataset was annotated with lexical borrowings, and the aim was to create a test set that was as difficult as possible, so there would be minimal overlap in words and topics between the training set and the test set. And as a result, the test set comes from sources and dates that were not seen in the training set. Here you can see that there is no overlap. The test set is also very borrowing dense. Just to give you some numbers, if the training set contains six borrowings per each thousand tokens, the test set contains 20 borrowings per each thousand tokens. The test set contains as many out-of-vocabulary words as possible. In fact, 92% of the borrowings in the test set are out-of-vocabulary, so they were not seen during training. And the corpus consisted basically of a collection of texts that came from different sources of Spanish newspapers. It was annotated by hand using two tags: one for English lexical boundaries, which is the majority of lexical boundaries in Spanish, and then the label other for borrowings from other languages. We use GONAL formats, and we use BIO encoding.So that we could encode single-token borrowings, such as app, or multi-token borrowings, such as machine learning. These are the numbers of the corpus. As you can see, it amounts to roughly 317,000 tokens, and here you have the number of spans that were labeled as English, and the spans that were labeled as other borrowings, and how many of them were unique. Here you have a couple of examples of the set of the data set. As you can see, for instance, here we have in the first example we have the borrowing batch cooking, which is a multi-word borrowing, and we have annotated it using the BIO code. The BIO was used for words in Spanish, so not for words that were not borrowed. And here in the second example, you have benching and crash, which are also labeled as borrowings from English. So once we had the data set, we explored several models for the task of extracting and detecting these lexical borrowings. The first one that we tried was a conditional random field model. This was the model that had been used on previous work, and we used the same handcrafted features from that work. As you can see, these are the features. These are binary features, such as is the word or the token in uppercase, is it title case, is it a quotation mark, things like that, which are the type of features that one would expect in a named entity recognition task. These are the results that we got. We obtained a 55 of F1 score using the CRF model with handcrafted feature, which is a huge difference compared to the reported F1 score of 86, which was the result obtained with the same CRF model, same features, but on a different data set, also for Spanish lexical borrowing detection. So this proves that the data set that we created is more difficult, and that we needed to explore more sophisticated models for this task. So we tested two transformer-based models. We used BERT, which is a monolingual BERT model trained for Spanish, and also multilingual BERT. Both models, we used them through the transformers library by Hugging Face. These are the results that we got. As you can see, multilingual BERT performed better than BERT, both on the development set and on the test set, and across all metrics. Just so we have an idea to compare.The CRF model obtained 82. The CRF model obtained 55, obtained 55 of F1 score, whereas the multilingual bird obtained 82, which is a big difference. So once that we had those results, we asked ourselves another question, which is could we find a BiLSTM CRF model fed with different types of embeddings? Embeddings that encode different types of linguistic information and outperform the results obtained by transformer-based models. So in order to do so, we ran some preliminary experiments. We ran this BiLSTM CRF model using the flair library, and we tried and experimented with different types of embeddings, like transformer-based, but also fast text character embeddings, and so on. What we found out was that transformer-based embeddings perform better than non-contextualized embeddings, that the combination of English bird and Spanish bird embeddings outperform multilingual bird embeddings, and that BPE embeddings produce better F1 and character embeddings produce better recall. With that in mind, these were the best performing results that we got. Both models were BiLSTM CRF models using flair. One was fed with bird and bird embeddings and BPE, and the other one with bird embeddings, BPE, and also character embeddings. This last one was the one that produced the highest F1 score on the test set, although the highest score on the development set was obtained by the one without character embeddings. Just bear in mind that the best result that we got with multilingual bird obtained an F1 of 76 on the development set and 82 on the test set. So this is an improvement compared to those results. Finally, we asked ourselves another question, which was can lexical borrowing detection be framed as transfer learning from language identification in code switching? We run the same BiLSTM CRF model that we had run using flair, but instead of using these adapted transformer-based bird and bird embeddings, we use code switching embeddings. What are code switching embeddings? These are embeddings that have been fine-tuned on transformer-based embeddings that have been pre-trained for language identification on the Spanish English section of the LTC code switching dataset. LTC is a dataset on code switching that has a section on Spanish English Spanish English code switching. So we fed our BiLSTM CRF with code switching embeddings and optionally character embeddings, BPE embeddings, and so on.The best result that we got was 84.22, which is the highest across all the models that we tried on the test set. Although the best result F1 score that we got on the development set, which was 79, was lower than the best result obtained by the BiLSTM-CRF fed with unadapted embeddings. Some conclusions from our work: we have produced a new dataset of Spanish news wire that is annotated with and simulated lexical borrowings. This data set has more borrowing events and OOV reach than previous resources. We have explored four types of models for lexical borrowing detection. In terms of error analysis, recall was a weak point for all models. As you can see here, some frequent false negatives include uppercase borrowings, words that exist in both English and Spanish. For instance, also interesting, PP embeddings seem to improve F1 score, and character embeddings seem to improve recall, which is an interesting finding that perhaps we can explore in future work. This is everything that I have. Thank you so much for listening."
# text = "嗨。我们将讨论我们关于生成检索增强的反事实以用于问答任务的工作。这是我在谷歌研究实习期间完成的工作。当时我由 Matthew Lemo 和 Ian Tandy 指导。为了激励任务,要先定义一个反事实。在这项工作中,我们将反事实定义为输入文本的扰动。它以某种有意义的可控方式与原始文本不同,使我们能够推理结果或任务标签的变化。例如,将迷人这个词改为吸引人,或者预计我的麻木会改变对这部电影评论的看法。同样,将限定词女性添加到问题中会改变下面示例中问题的答案。与 NLP 模型相比,人类通常对这种扰动具有鲁棒性。为什么?数据集可能会被系统性的抽样,导致一个单独的决策边界被反事实所违反,如这个二维分类问题所示。Bryant 发现将反事实示例添加到训练数据中可以使模型对这种扰动具有鲁棒性。所以,如果反事实规则是有价值的,我们如何生成它们?这个任务对 NLP 来说特别困难,因为这里有三个来自不同 NLP 任务的例子。如您所见,违反结果之间决策边界的事例需要非常仔细地通过扰动文本的某些属性来精心制作。这些属性在此处被下划线标出。这可以通过人工注释来完成,但这是昂贵的和有偏见的。一些前期工作专注于使用语法树或语义角色标注,但是这些技术产生的扰动集受到语义框架的限制。最近的工作使用了大规模语言模型来填充文本的掩码部分以改变标签,但是要找出文本的哪些部分可以引用可能会很具挑战性。生成问题回答的反事实更具挑战性。具体来说,这项任务需要背景知识。例如,要提出原始问题是《夺宝奇兵》,你转盘是前传吗?我们需要了解该系列中的其他电影。要回答像印第安纳琼斯这样的问题,最后的圣战者是前传。此外,随机扰动可能导致无法用现有证据回答的问题或具有错误前提的问题。此外,一些问题扰动会导致与原始输入相比出现显著的语义漂移。例如,这个问题是关于《夺宝奇兵》中印第安纳琼斯在圣战奇兵中从事儿童奴隶时。我们提出了一种非常简单但有效的技术,称为检索生成过滤器或 RGF,以解决反事实问题的扰动。同时,也旨在解决所有其他上述挑战。而 D F 的核心直觉是,生成扰动所需的必要背景信息可能存在于问答模型做出的近似中。例如,最先进的模型 R N 对问题谁是李氏满足球俱乐部的队长给出了以下前 K 个答案。嗯,它确实恢复了原始的参考段落和答案。Chint Carting 做最上面的选择,它检索了额外的段落和答案,这些可以用来指导问题的提出。例如,它恢复了两个更多的答案,对应于同一俱乐部的预备队和女子队的队长。这可能会导致有趣的编辑。总结一下,R G F 首先检索出与参考答案和上下文不匹配的前 K 个最相关答案和上下文。在该步骤之后,问题生成模型会根据这些替代答案来生成与之对应的问。题最后,我们可以根据最小性或我们想要引入的语义扰动类型来过滤生成的问题。更详细的回顾每个步骤,为了检索,我们使用像 R N 这样的检索,然后阅读模型,它将原始问题作为输入,以及像维基百科这样的大型语料库。它有两个模块组成,检索模块在密集的段落索引上执行相似性搜索,以检索与问题最相关的前 K 个段落。然后,阅读器模块从每个段落中提取一个跨度作为潜在答案。在大多数情况下,R N 会检索黄金通道和答案。然而,在这项工作中,我们更感兴趣的是它在后面检索的答案和上下文。在下一步问题生成中,我们使用这些替代答案和上下文来生成与这些替代方案相对应的新问题。问题生成模型是一个预训练的文本到文本转换器,它在 N Q 数据上进行了微调,以生成一个答案的问题。该答案在上下文中被标记。在推理过程中,我们向问题生成模型提供我们在上一步中检索到的替代答案和上下文。例如,对于查询谁是李氏满足球俱乐部的队长,R G 会检索关于该俱乐部女子队的段落,尤其是肯尼迪担任队长。然后问题生成模型会生成查询谁是李氏满足球俱乐部第一支女子队的队长,其中包含特定的语义扰动。以类似的方式,我们还会得到诸如谁是李氏满威弗尔预备队的队长,或去年格雷厄姆尼甘特在总决赛中对阵谁这样的查询。最后,我们根据一些期望的特征过滤出生成查询的一个子集,正如我们之前所激励的。我们希望确保新问题与原始问题在语义上仍然接近。对于不需要额外监督的过滤技术,我们只需保留新问题,那些与原始问题有小标记及编辑距离的问题。例如,我们删除了问题,谁在大满贯中击败了格拉姆?我,因为它的编辑距离比原始问题更长。在我们的实验中,我们证明了这个简单的启发式方法可以用来增强和清理训练数据。我们还尝试了一种过滤策略,该策略基于语义扰动的类型。为此,我们使用了一个通用查询分解框架,称为 QED。QED 将问题分为两部分:谓词和参考。参考是问题中的名词短语,对应于上下文中的实体。谓词基本上是问题的剩余部分。例如,我们能够将查询谁领导了李世满的低职女子队分解为两个引用:李世满足球俱乐部女子队和谓词谁领导了 X。一个在参考谓词注释上训练的模型对于 NQ 给出了这个问题的分解。基于 QED 分解原始问题和生成的问题,使我们能够对生成的反事实进行分类以供评估。具体来说,我们得到了两组问题:那些在保留谓词的同时发生参考变化的问题,以及那些发生谓词变化并可选地添加参考的问题。例如,谁是李世满 VFL 预备队的队长是一个参考变化,而谁穿九号球衣俱乐部是一个谓词变化。现在,我们评估 RGF 扰动在训练数据增强时的有效性。因此,为了有效地评估反事实增强的有效性,我们特别对两个强大的数据增强基线进行了实验。第一个基线称为随机答案和问题生成,添加了与原始问题无关的数据集、段落和答案,只是从维基百科中随机采样的。这条基线基本上增加了更多看起来像 NQ 的数据。第二条线黄金答案和问题生成。我们专门更新了我们方法的检索部分,在这里备选答案只是从包含黄金答案的同一段落中选择的。而 GAF 增强的基线在阅读理解方面表现如何?其中,模型可以访问问题和上下文。我们对六个域外数据集进行了实验,结果如下:训练数据在增强中被加倍。我们发现两种数据增强基线都无法提高域外泛化能力。事实上,由六个在原始数据上训练的模型组成的集成似乎是竞争最激烈的基线。与该基线相比,我们发现 RGF 对抗事实能够提高域外性能,同时保持域内性能。这表明,通过对抗事实增强来填补模型的推理空白,比从训练分布中添加更多数据更有效。此外,我们发现使用检索来采样替代结果或答案对于有效的 CDA 很重要。我们还尝试了开放域 QA 设置,其中模型只看到问题。再一次,我们在四个域外数据集上进行评估。我们发现,基线模型对于域外泛化效果不佳。然而,使用 RGF 的数据增强显示了更显著的改进。我们在域内和 Q 数据集上甚至都得到了改进。我们假设反事实数据增强有助于模型学习更好的查询编码,用于非常相似的查询。最后,我们还评估了模型在原始问题的局部领域中提高一致性的能力。一致性衡量的是模型正确回答的问题的比例,其中原始查询和反事实查询都被正确回答。这明确的帮助我们衡量模型对原始输入领域中小扰动的鲁棒性。我们对五个数据集进行了实验。这些数据集包含语义上彼此接近的问题。对除了三个数据集 AQA、MBQA 和 CoREF 对比集之外,那些已经可用的。我们也评估了与原始 NQ 问题配对的 RGF 反事实,基于它们是否经历了谓词变化或引用变化。这些子集在内部进行了注释以消除噪声,并作为自然提供。所有基线都无法显著提高一致性,而集成模型仅小幅提高了。然而,RGF 反事实增强在一致性与准确性方面都有显著的提升,无论是在现有数据集上,还是在我们创建的两个子集上。对于参考和谓词扰动,注意增强的 RGF 数据不受扰动类型的影响,只有评估集受到影响。事实上,对生成的反事实类型的定性检查表明生成的问题包含几个不同的扰动,例如,这个关于明尼苏达州和泰国人口的原始问题在城镇、州、国家等不同维度上被扰动,以及沿着不同的谓词,如位置、贫困、学校数量。扰动的 RGF 是上下文特定的,例如,对于这个关于温布尔登单打比赛的其他问题,扰动是沿着游戏类型、比赛类型或游戏结果。最后的要点是,我们解决了反事实数据增强和扰动的任务,以应对信息查询。通过一种生成方法的逆向操作来解决其独特的挑战,使用模型的近似值进行过度生成,并根据扰动类型或最小化进行过滤。我们发现这种技术不需要额外的监督,而且示例被标记为增强。增强改善了域外泛化和领域一致性。我们发现 RGF 反事实在不引入偏差的情况下具有语义多样性。谢谢。"
# print(text[1882:1889])

List = [[468, 469], [470, 471], [471, 472], [472, 473], [473, 474], [474, 475], [475, 476], [476, 477], [477, 478], [478, 479], [479, 480], [480, 481], [481, 482], [483, 484], [484, 485], [485, 486], [486, 487]]
for span in List:
    print(span, text[span[0]:span[1]])