#from utils import inference_model_bart
# from inference_model_bart import cli_main
import re
import functools
import MeCab
from ja_sentence_segmenter.common.pipeline import make_pipeline
from ja_sentence_segmenter.concatenate.simple_concatenator import  concatenate_matching
from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation

def write_text_to_file(text):
    # ファイルを開き、全ての行を読み込んで空白と改行を削除
    # 全ての行を連結して、改行と空白を削除
    pre_text = text.replace("\n", "").replace(" ", "")
    cleaned_text = re.sub(r"\s+", "", pre_text)
    # 4つ以上の連続するアンダースコアを削除
    cleaned_text = re.sub(r"_{4,}", "", cleaned_text)

    # \s は空白文字全般（スペース、タブ、改行など）を指します
    print(cleaned_text)

    # テキストを「。」で分割
    sentences = cleaned_text.split("。")

    # 出力ファイルに一行ずつ書き込み
    with open('/home/yuki_ueda/AI_line_bot/line_bot/utils/output_llm.txt', 'w', encoding='utf-8') as file:
        for sentence in sentences:
            # 文が空でない場合にのみ書き込み、「。」を付け加えて出力
            if sentence.strip():
                file.write(sentence + "。\n")


# 使用例
#text = "いい天気で気分も晴れやかです。"
#out = tanomu(text)
#print(out)

# MeCabで形態素解析
def is_particle_or_auxiliary(word, pos):
    """形態素解析の結果から助詞・助動詞を判別"""
    return pos in {"助詞", "助動詞"}

def is_auxiliary(word, pos):
    """形態素解析の結果から助詞・助動詞を判別"""
    return pos == "助動詞"

def is_conjunction(word, pos):
    """形態素解析の結果から助詞・助動詞を判別"""
    return pos == "接続詞"

def is_particle(word, pos):
    """形態素解析の結果から助詞・助動詞を判別"""
    return pos == "助詞"


def mecab_tail_check(sentence):
    """文末が助詞・助動詞かどうかを確認"""
    mecab = MeCab.Tagger("-Owakati")
    mecab.parse("")  # バグ回避用
    node = mecab.parseToNode(sentence.strip())
    last_word = None
    while node:
        if node.surface:
            last_word = (node.surface, node.feature.split(",")[0])  # 表層形と品詞
        node = node.next
    return last_word and is_conjunction(*last_word) or is_particle(*last_word)



def mecab_head_check(sentence):
    """文頭が助詞・助動詞かどうかを確認"""
    mecab = MeCab.Tagger("-Owakati")
    mecab.parse("")  # バグ回避用
    node = mecab.parseToNode(sentence.strip())
    if node:
        node = node.next  # 最初のノードはBOSなのでスキップ
    if node and node.surface:
        first_word = (node.surface, node.feature.split(",")[0])  # 表層形と品詞
        return is_particle_or_auxiliary(*first_word) or is_conjunction(*first_word)
    return False

def split_on_auxiliary_with_space(sentence):
    """
    文中の助動詞の後に空白がある場合に分割
    """
    mecab = MeCab.Tagger("-Owakati")
    mecab.parse("")  # バグ回避用
    node = mecab.parseToNode(sentence.strip())
    result = []
    current_sentence = ""
    i = 0

    while node:
        if node.surface:
            current_sentence += node.surface
            # 助動詞で、次に空白がある場合
            if is_auxiliary(node.surface, node.feature.split(",")[0]):
                if i + len(node.surface) < len(sentence) and sentence[i + len(node.surface)] in {" ", "　"}:
                    # 分割
                    result.append(current_sentence.strip())
                    current_sentence = ""
        i += len(node.surface)  # 現在の位置を更新
        node = node.next

    # 最後の文を追加
    if current_sentence:
        result.append(current_sentence.strip())
    return result


def mecab_based_concatenator(sentences):
    """MeCabを利用して文の結合を行う（句点を優先して区切る）"""
    new_sentences = []
    prev_sentence = ""

    for sentence in sentences:
        # 文末に句点がある場合、強制的に区切る
        if prev_sentence.endswith(("。", "!", "?", "！", "？")):
            if sentence.strip().startswith(("!", "?", "！", "？")):
                prev_sentence += sentence
                continue
            new_sentences.append(prev_sentence)
            prev_sentence = sentence
        else:
            split_sentences = split_on_auxiliary_with_space(sentence)
            for split_sentence in split_sentences:
                if prev_sentence:
                    # 文末・文頭条件で結合
                    if mecab_tail_check(prev_sentence) or mecab_head_check(split_sentence):
                        prev_sentence += split_sentence
                    else:
                        new_sentences.append(prev_sentence)
                        prev_sentence = split_sentence
                else:
                    prev_sentence = split_sentence
    if prev_sentence:
        new_sentences.append(prev_sentence)
    return new_sentences

# 文区切り処理のパイプラインを作成
split_punc2 = functools.partial(split_punctuation, punctuations=r".。!?！？")
concat_decimal = functools.partial(concatenate_matching, former_matching_rule=r"^(?P<result>.+)(\d.)$", latter_matching_rule=r"^(\d)(?P<result>.+)$", remove_former_matched=False, remove_latter_matched=False)
concat_tail_comma_recursive = functools.partial(concatenate_matching, former_matching_rule=r"^(?P<result>.+)、$", latter_matching_rule=r"^(?P<result>.+)$", remove_former_matched=False, remove_latter_matched=False,)
segmenter = make_pipeline(
    #ormalize,
    split_newline,
    concat_tail_comma_recursive,
    concat_decimal,
    split_punc2,
)

def write_ocr_to_file(ocr_output_path):
    with open(ocr_output_path, "r", encoding="utf-8-sig") as file:
        content = file.read()
    content = re.sub(r"_{4,}", "", content)
    split_sentences = list(segmenter(content))
    final_sentences = mecab_based_concatenator(split_sentences)
    
    # 結果の確認と書き込み
    output_file_path = "/home/yuki_ueda/AI_line_bot/line_bot/utils/output_llm.txt"  # 書き込み先のファイル名を指定
    

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        count = 1
        for sentence in final_sentences:
            sentence = sentence.strip()  # 先頭や末尾の空白、改行、BOMを削除
            if sentence:  # 空でない行のみ書き込み
                output_file.write(f"{sentence}\n")
                count += 1

