from collections import namedtuple
import fileinput
import logging
import math
import sys
import time
import os
import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import encoders
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
import zenhan
from pyknp import Juman
import sentencepiece
import json

logging.basicConfig(
    format='%(name)s | %(message)s',
    level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    stream=sys.stdout,
)
logger = logging.getLogger('infer')

Batch = namedtuple('Batch', 'ids src_tokens src_lengths constraints')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


NG_FILE_PATH = "/home/yuki_ueda/AI_line_bot/line_bot/utils/ng_words.json"


def load_ng_phrases(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                print("NG発見")
                return data
            else:
                return {}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {}
        


def fill_empty_lines(file1, file2, file3, output_file):
    # ファイルを読み込む
    with open(file1, 'r', encoding='utf-8') as f1, \
         open(file2, 'r', encoding='utf-8') as f2, \
         open(file3, 'r', encoding='utf-8') as f3:
        
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        lines3 = f3.readlines()

    # 処理結果を書き込む
    with open(output_file, 'w', encoding='utf-8') as out:
        for i in range(max(len(lines1), len(lines2), len(lines3))):
            # 各ファイルの行を取得（範囲外の場合は空文字にする）
            line1 = lines1[i].strip() if i < len(lines1) else ''
            line2 = lines2[i].strip() if i < len(lines2) else ''
            line3 = lines3[i].strip() if i < len(lines3) else ''

            # 空行の優先順位を確認
            if line1:
                out.write(line1 + '\n')
            elif line2:
                out.write(line2 + '\n')
            elif line3:
                out.write(line3 + '\n')
            else:
                out.write('\n')



def filter_ng_phrases_by_keyword(ng_phrases, keyword):
    if ng_phrases is None:
        return []
    return ng_phrases.get(keyword, [])








def copy_file_content(src_file, dest_file):
    try:
        # 元ファイルを開いて内容を読み込む
        with open(src_file, 'r', encoding='utf-8') as src:
            content = src.read()

        # 読み込んだ内容を別のファイルに書き込む
        with open(dest_file, 'w', encoding='utf-8') as dest:
            dest.write(content)

        print(f"Content copied from {src_file} to {dest_file} successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")



def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, 'symbols_to_strip_from_output'):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]

    bos_token_id = task.source_dictionary.bos()
    tokens_with_bos = [torch.cat([torch.tensor([bos_token_id]), token]) for token in tokens]

    if args.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None
    lengths = [t.numel() for t in tokens_with_bos]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens_with_bos, lengths, constraints=constraints_tensor),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch['id']
        src_tokens = batch['net_input']['src_tokens']
        src_lengths = batch['net_input']['src_lengths']
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )


def main(args):
    start_time = time.time()
    total_translate_time = 0

    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = True

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    #logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        [args.path],
        arg_overrides=eval(args.model_overrides),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.prepare_for_inference_(args)
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(models, args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    jumanpp = Juman()
    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.bpe_model)
    #   return ' '.join([mrph.midasi for mrph in result.mrph_list()])
    def juman_split(line, jumanpp):
        result = jumanpp.analysis(line)
        return ' '.join([mrph.midasi for mrph in result.mrph_list()])

    def bpe_encode(line, spm):
        return ' '.join(spm.EncodeAsPieces(line.strip()))

    def encode_fn(x):
        x = x.strip()
        x = zenhan.h2z(x)
        x = juman_split(x, jumanpp)
        x = bpe_encode(x, spm)
        return x

    def decode_fn(x):
        x = x.translate({ord(i): None for i in ['▁', ' ']})
        return x

    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    if args.constraints:
        logger.warning("NOTE: Constrained decoding currently assumes a shared subword vocabulary.")

    if args.buffer_size > 1:
        logger.info('Sentence buffer size: %s', args.buffer_size)
    logger.info('NOTE: hypothesis and token scores are output in base 2')
    logger.info('Type the input sentence and press return:')
    start_id = 0

    # 入力用ファイルを指定する
    input_text = '/home/yuki_ueda/AI_line_bot/line_bot/utils/output_llm.txt'

    NG_phrases = load_ng_phrases(NG_FILE_PATH)

    

    # 出力用の配列
    output_texts = []
    output_texts_2 = []
    output_texts_3 = []


    
    with open(input_text, 'r', encoding='utf-8') as f:
        input_lines = [line.strip().replace(" ", "").replace("　", "") for line in f.readlines()]
    
    # 行数カウンター
    line_index = 0  # 現在処理中の行のインデックス
    

    for inputs in buffered_read(input_text, args.buffer_size):
        results = []
        for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            constraints = batch.constraints
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translate_start_time = time.time()
            translations = task.inference_step(generator, models, sample, constraints=constraints)
            translate_time = time.time() - translate_start_time
            total_translate_time += translate_time
            list_constraints = [[] for _ in range(bsz)]
            if args.constraints:
                list_constraints = [unpack_constraints(c) for c in constraints]
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                constraints = list_constraints[i]
                results.append((start_id + id, src_tokens_i, hypos,
                    {
                        "constraints": constraints,
                        "time": translate_time / len(translations)
                    }))

        # sort output to match input order
        for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                print(f'Inference time: {info["time"]:.3f} seconds')

            
            # 現在の行をキーワードとして取得
            current_keyword = input_lines[line_index]
            current_keyword = current_keyword.rstrip('。')
            line_index += 1  # 次の行のインデックスに進める

            # キーワードに基づいてNGワードを取得
            filter_ng_phrases = filter_ng_phrases_by_keyword(NG_phrases, current_keyword)
            print(current_keyword)
            print(filter_ng_phrases)
            





            # Process top predictions
            for hypo_i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                score = hypo['score'] / math.log(2)  # convert to base 2


                # NG文に該当しないかを確認
                if not any(ng in detok_hypo_str for ng in filter_ng_phrases):  # NGフレーズが含まれていない場合
                    if hypo_i == 0:
                        output_texts.append(detok_hypo_str)
                    elif hypo_i == 1:
                        output_texts_2.append(detok_hypo_str)
                    elif hypo_i == 2:
                        output_texts_3.append(detok_hypo_str)
                    print(f'Top {hypo_i+1} prediction score: {score}')
                else:
                    # NG文が含まれていた場合は、次の候補にスキップ
                    if hypo_i == 0:
                        output_texts.append("")
                        print("output_texts")
                    elif hypo_i == 1:
                        output_texts_2.append("")
                        print("output_texts_2")
                    elif hypo_i == 2:
                        output_texts_3.append("")
                        print("output_texts_3")
                    print(f"Skipped NG sentence: {detok_hypo_str}")


                """
                if hypo_i == 0:
                    output_texts.append(detok_hypo_str)
                if hypo_i == 1:
                    output_texts_2.append(detok_hypo_str)
                if hypo_i == 2:
                    output_texts_3.append(detok_hypo_str)

                print(f'Top {hypo_i+1} prediction score: {score}')
                """

        # update running id_ counter
        start_id += len(inputs)
    
    # 要約結果を１〜３番目までそれぞれ書き出し
    with open(f"/home/yuki_ueda/AI_line_bot/line_bot/utils/{SAVE_MODEL_NAME}_test_tgt.txt", 'w') as f:
        count = 0
        for d in output_texts:
            count+=1
            f.write("%s\n" % d)
    with open(f"/home/yuki_ueda/AI_line_bot/line_bot/utils/{SAVE_MODEL_NAME}_test_tgt_2.txt", 'w') as f:
        count = 0
        for d in output_texts_2:
            count+=1
            f.write("%s\n" % d)
    with open(f"/home/yuki_ueda/AI_line_bot/line_bot/utils/{SAVE_MODEL_NAME}_test_tgt_3.txt", 'w') as f:
        count = 0
        for d in output_texts_3:
            count+=1
            f.write("%s\n" % d)




SAVE_MODEL_NAME = "snow_matcha"
MODEL_NAME = "/home/yuki_ueda/bart-env/study_code/models/" + SAVE_MODEL_NAME

def cli_main():
    parser = options.get_interactive_generation_parser()
    parser.add_argument('--bpe_model', default='', required=True)
    parser.add_argument('--bpe_dict', default='', required=True)

    bpe_model = MODEL_NAME + "/sp.model"
    bpe_dict = MODEL_NAME + "/dict.txt"
    datasets_dir = MODEL_NAME + "/datasets"
    tuning_model = MODEL_NAME + "/save_snow_matcha/checkpoint_best.pt"

    input_args = [
        datasets_dir, 
        "--path", tuning_model,
        "--task", "translation_from_pretrained_bart",
        "--source-lang", "src",  # ソース言語
        "--target-lang", "tgt",
        "--max-sentences", "1",
        "--bpe_model", bpe_model,
        "--bpe_dict", bpe_dict,
        "--nbest", "3",
        "--beam", "10",
    ]
    args = options.parse_args_and_arch(parser,input_args)

    distributed_utils.call_main(args, main)






    file_path_output = "/home/yuki_ueda/AI_line_bot/line_bot/utils/last_output.txt"

    fill_empty_lines("/home/yuki_ueda/AI_line_bot/line_bot/utils/" + SAVE_MODEL_NAME + "_test_tgt.txt", "/home/yuki_ueda/AI_line_bot/line_bot/utils/" + SAVE_MODEL_NAME + "_test_tgt_2.txt", "/home/yuki_ueda/AI_line_bot/line_bot/utils/" + SAVE_MODEL_NAME + "_test_tgt_3.txt", file_path_output)

    with open('/home/yuki_ueda/AI_line_bot/line_bot/utils/get_img_or_text.txt', 'r', encoding='utf-8') as file:
        get_img_or_text = file.readline().strip()  # .strip() で前後の空白や改行を削除
    if get_img_or_text:
        img_or_text = int(get_img_or_text)



    with open(file_path_output, 'r') as file:
        all_lines = file.readlines()  # 各行をリストに格納
        if all_lines:
            if img_or_text == 0:
                last_text = ''.join([line for line in all_lines])
            else:
                last_text = ''.join([line.strip() for line in all_lines])  # 各行の改行を削除して一つの文字列に結合
        else:
            last_text = "<やさしい日本語に変換できませんでした>"
       


    return last_text

"""
    # ファイルを開いて、一行目を取得
    with open(file_path_output, 'r') as file:
        last_text = file.readline().strip()
    return last_text
"""
    

def create_single_text_message():
    output_text = cli_main()
    output_text = output_text.rstrip('\n')
    test_message = {
                    'type': 'text',
                    'text': output_text
                }
            
    return test_message

def create_all_text_message():
    cli_main()
    with open('/home/yuki_ueda/AI_line_bot/line_bot/utils/get_img_or_text.txt', 'r', encoding='utf-8') as file:
        get_img_or_text = file.readline().strip()  # .strip() で前後の空白や改行を削除
    if get_img_or_text:
        img_or_text = int(get_img_or_text)
    with open("/home/yuki_ueda/AI_line_bot/line_bot/utils/" + SAVE_MODEL_NAME + "_test_tgt.txt", 'r') as file:
        all_lines_1 = file.readlines()  # 各行をリストに格納
        print(all_lines_1)
        if all_lines_1:
            if img_or_text == 0:
                output_text_1 = ''.join([line for line in all_lines_1])
                output_text_1 = output_text_1.rstrip('\n')
            else:
                output_text_1 = ''.join([line.strip() for line in all_lines_1])  # 各行の改行を削除して一つの文字列に結合
        else:
            output_text_1 = 'NG文'
    with open("/home/yuki_ueda/AI_line_bot/line_bot/utils/" + SAVE_MODEL_NAME + "_test_tgt_2.txt", 'r') as file:
        all_lines_2 = file.readlines()
        if all_lines_2:
            if img_or_text == 0:
                output_text_2 = ''.join([line for line in all_lines_2])
                output_text_2 = output_text_2.rstrip('\n')
            else:
                output_text_2 = ''.join([line.strip() for line in all_lines_2])  # 各行の改行を削除して一つの文字列に結合
        else:
            output_text_2 = 'NG文'
    with open("/home/yuki_ueda/AI_line_bot/line_bot/utils/" + SAVE_MODEL_NAME + "_test_tgt_3.txt", 'r') as file:
        all_lines_3 = file.readlines()
        if all_lines_3:
            if img_or_text == 0:
                output_text_3 = ''.join([line for line in all_lines_3])
                output_text_3 = output_text_3.rstrip('\n')
            else:
                output_text_3 = ''.join([line.strip() for line in all_lines_3])  # 各行の改行を削除して一つの文字列に結合
        else:
            output_text_3 = 'NG文'
    test_message = {
                    'type': 'text',
                    'text': f"候補1：{output_text_1}\n候補2：{output_text_2}\n候補3：{output_text_3}"
                }
            
    return test_message