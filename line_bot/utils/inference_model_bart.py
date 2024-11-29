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

logging.basicConfig(
    format='%(name)s | %(message)s',
    level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    stream=sys.stdout,
)
logger = logging.getLogger('infer')

Batch = namedtuple('Batch', 'ids src_tokens src_lengths constraints')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


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

    # デバッグ用ログ: トークン化結果を確認
    for i, token in enumerate(tokens_with_bos):
        logger.info(f"Encoded token for line {i}: {token}")

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

        for i, src_token in enumerate(batch['net_input']['src_tokens']):
            # トークンのIDをログに出力 
            logger.info(f"Input token IDs for inference (line {i}): {src_token.tolist()}") # IDのリストに変換 
            # トークンを文字列に変換（必要に応じて） 
            src_str = task.source_dictionary.string(src_token, args.remove_bpe) 
            logger.info(f"Input text for inference (line {i}): {src_str}")


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
        logger.info(f"Juman++ tokenization result: {[mrph.midasi for mrph in result.mrph_list()]}")
        return ' '.join([mrph.midasi for mrph in result.mrph_list()])

    def bpe_encode(line, spm):
        encoded = spm.EncodeAsPieces(line.strip())
        logger.info(f"SentencePiece BPE encoding result: {encoded}")
        return ' '.join(encoded)

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
    input_text = '/home/yuki_ueda/bart-env/study_code/datasets/test_src.txt'
    dest_file = '/home/yuki_ueda/AI_line_bot/line_bot/utils/output_llm_copy.txt'

    # 出力用の配列
    output_texts = []
    output_texts_2 = []
    output_texts_3 = []

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

                # デバッグ: 復元された文を確認
                logger.info(f"Post-processed hypo_str: {hypo_str}")
                logger.info(f"Generated tokens: {hypo_tokens}")
                logger.info(f"Post-processed hypothesis string: {hypo_str}")

                detok_hypo_str = decode_fn(hypo_str)
                score = hypo['score'] / math.log(2)  # convert to base 2

                if hypo_i == 0:
                    output_texts.append(detok_hypo_str)
                if hypo_i == 1:
                    output_texts_2.append(detok_hypo_str)
                if hypo_i == 2:
                    output_texts_3.append(detok_hypo_str)

                print(f'Top {hypo_i+1} prediction score: {score}')

        # update running id_ counter
        start_id += len(inputs)
    
    # 要約結果を１〜３番目までそれぞれ書き出し
    with open(f"/home/yuki_ueda/bart-env/study_code/datasets/{SAVE_MODEL_NAME}_test_tgt.txt", 'w') as f:
        count = 0
        for d in output_texts:
            count+=1
            f.write("%s\n" % d)
    with open(f"/home/yuki_ueda/bart-env/study_code/datasets/{SAVE_MODEL_NAME}_test_tgt_2.txt", 'w') as f:
        count = 0
        for d in output_texts_2:
            count+=1
            f.write("%s\n" % d)
    with open(f"/home/yuki_ueda/bart-env/study_code/datasets/{SAVE_MODEL_NAME}_test_tgt_3.txt", 'w') as f:
        count = 0
        for d in output_texts_3:
            count+=1
            f.write("%s\n" % d)




SAVE_MODEL_NAME = "snow_prepro"
MODEL_NAME = "/home/yuki_ueda/bart-env/study_code/models/" + SAVE_MODEL_NAME

def cli_main():
    parser = options.get_interactive_generation_parser()
    parser.add_argument('--bpe_model', default='', required=True)
    parser.add_argument('--bpe_dict', default='', required=True)

    bpe_model = MODEL_NAME + "/sp.model"
    bpe_dict = MODEL_NAME + "/dict.txt"
    datasets_dir = MODEL_NAME + "/datasets"
    tuning_model = MODEL_NAME + "/save_snow_prepro/checkpoint_best.pt"

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



    with open('/home/yuki_ueda/AI_line_bot/line_bot/utils/i_number.txt', 'r', encoding='utf-8') as file:
        get_same_number = file.readline().strip()  # .strip() で前後の空白や改行を削除

    i = int(get_same_number)
    if i >= 3:
        i = (i % 3) - 1

    # ファイルを読み込みます
    with open(input_text, 'r', encoding='utf-8') as infile:
        input_content = infile.read()

    with open(dest_file, 'r', encoding='utf-8') as destfile:
        dest_content = destfile.read()

    # 内容を比較して同じであればiを+1
    if input_content == dest_content:
        i += 1
    else:
        i = 0
        copy_file_content(input_text, dest_file)

    with open('/home/yuki_ueda/AI_line_bot/line_bot/utils/i_number.txt', 'w', encoding='utf-8') as file:
        file.write(i)
    
    if i == 0:
        file_path_output = "/home/yuki_ueda/bart-env/study_code/datasets/{SAVE_MODEL_NAME}_test_tgt.txt"
    elif i == 1:
        file_path_output = "/home/yuki_ueda/bart-env/study_code/datasets/{SAVE_MODEL_NAME}_test_tgt_2.txt"
    else:
        file_path_output = "/home/yuki_ueda/bart-env/study_code/datasets/{SAVE_MODEL_NAME}_test_tgt_3.txt"
    
    



    # ファイルを開いて、一行目を取得
    with open(file_path_output, 'r') as file:
        last_text = file.readline().strip()
    return last_text