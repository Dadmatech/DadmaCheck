import argparse
import os
from tqdm import tqdm
import re
import time
import torch
import utils
from helpers import load_vocab_dict
from helpers import batch_iter, labelize, bert_tokenize_for_valid_examples
from helpers import  untokenize_without_unks, untokenize_without_unks2, get_model_nparams
from hazm import Normalizer
from models import SubwordBert
from utils import get_sentences_splitters

def model_inference(model, data, topk, DEVICE, BATCH_SIZE=16, vocab_=None):
    """
        model: an instance of SubwordBert
        data: list of tuples, with each tuple consisting of correct and incorrect
                sentence string (would be split at whitespaces)
        topk: how many of the topk softmax predictions are considered for metrics calculations
        """
    if vocab_ is not None:
        vocab = vocab_
    print("###############################################")
    inference_st_time = time.time()
    _corr2corr, _corr2incorr, _incorr2corr, _incorr2incorr = 0, 0, 0, 0
    _mistakes = []
    VALID_BATCH_SIZE = BATCH_SIZE
    valid_loss = 0.
    print("data size: {}".format(len(data)))
    data_iter = batch_iter(data, batch_size=VALID_BATCH_SIZE, shuffle=False)
    model.eval()
    model.to(DEVICE)
    results = []
    line_index = 0
    for batch_id, (batch_labels, batch_sentences) in tqdm(enumerate(data_iter)):
        torch.cuda.empty_cache()
        st_time = time.time()
        # set batch data for bert
        batch_labels_, batch_sentences_, batch_bert_inp, batch_bert_splits = bert_tokenize_for_valid_examples(batch_labels, batch_sentences)
        if len(batch_labels_) == 0:
            print("################")
            print("Not predicting the following lines due to pre-processing mismatch: \n")
            print([(a, b) for a, b in zip(batch_labels, batch_sentences)])
            print("################")
            continue
        else:
            batch_labels, batch_sentences = batch_labels_, batch_sentences_
        batch_bert_inp = {k: v.to(DEVICE) for k, v in batch_bert_inp.items()}
        # set batch data for others
        batch_labels_ids, batch_lengths = labelize(batch_labels, vocab)
        batch_lengths = batch_lengths.to(DEVICE)
        batch_labels_ids = batch_labels_ids.to(DEVICE)

        try:
            with torch.no_grad():
                """
                NEW: batch_predictions can now be of shape (batch_size,batch_max_seq_len,topk) if topk>1, else (batch_size,batch_max_seq_len)
                """
                batch_loss, batch_predictions = model(batch_bert_inp, batch_bert_splits, targets=batch_labels_ids, topk=topk)
        except RuntimeError:
            print(f"batch_bert_inp:{len(batch_bert_inp.keys())},batch_labels_ids:{batch_labels_ids.shape}")
            raise Exception("")
        valid_loss += batch_loss
        batch_lengths = batch_lengths.cpu().detach().numpy()
        if topk == 1:
            batch_predictions = untokenize_without_unks(batch_predictions, batch_lengths, vocab, batch_sentences)
        else:
            batch_predictions = untokenize_without_unks2(batch_predictions, batch_lengths, vocab, batch_sentences, topk=None)
        batch_clean_sentences = [line for line in batch_labels]
        batch_corrupt_sentences = [line for line in batch_sentences]
        batch_predictions = [line for line in batch_predictions]

        for i, (a, b, c) in enumerate(zip(batch_clean_sentences, batch_corrupt_sentences, batch_predictions)):
            results.append({"id": line_index + i, "original": a, "noised": b, "predicted": c, "topk": [], "topk_prediction_probs": [], "topk_reranker_losses": []})
        line_index += len(batch_clean_sentences)

        '''
        # update progress
        progressBar(batch_id+1,
                    int(np.ceil(len(data) / VALID_BATCH_SIZE)), 
                    ["batch_time","batch_loss","avg_batch_loss","batch_acc","avg_batch_acc"], 
                    [time.time()-st_time,batch_loss,valid_loss/(batch_id+1),None,None])
        '''
    print(f"\nEpoch {None} valid_loss: {valid_loss / (batch_id + 1)}")
    print("total inference time for this data is: {:4f} secs".format(time.time() - inference_st_time))
    print("###############################################")
    return results


def load_model(vocab):
    model = SubwordBert(3*len(vocab["chartoken2idx"]),vocab["token2idx"][ vocab["pad_token"] ],len(vocab["token_freq"]))
    print(model)
    print( get_model_nparams(model) )
    return model


def load_pretrained(model, checkpoint_path, optimizer=None, device='cuda'):
    if torch.cuda.is_available() and device != "cpu":
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    print(f"Loading model params from checkpoint dir: {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    max_dev_acc, argmax_dev_acc = checkpoint_data["max_dev_acc"], checkpoint_data["argmax_dev_acc"]

    if optimizer is not None:
        return model, optimizer, max_dev_acc, argmax_dev_acc
    return model


def load_pre_model(vocab_path, model_checkpoint_path):
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"loading vocab from {vocab_path}")
    vocab = load_vocab_dict(vocab_path)
    model = load_model(vocab)
    model = load_pretrained(model, model_checkpoint_path)
    return model, vocab, DEVICE


def spell_checking_on_sents(model, vocab, device, normalizer, txt):
    sents, splitters = get_sentences_splitters(txt)
    sents = [utils.space_special_chars(s) for s in sents]
    sents = list(filter(lambda txt: (txt != '' and txt != ' '), sents))
    test_data = [(normalizer.normalize(t), normalizer.normalize(t)) for t in sents]
    greedy_results = model_inference(model, test_data, topk=1, DEVICE=device, BATCH_SIZE=1, vocab_=vocab)

    corrected_text = ''.join([line['predicted'] + splitter for line, splitter in zip(greedy_results, splitters + ['\n'])])
    
    return corrected_text


def main():
    parser = argparse.ArgumentParser(description="Spell checking script.")
    parser.add_argument("--input-file", required=True, help="Path to the input text file for spell checking.")
    parser.add_argument("--output-file", default=None, help="Path to the output file. If not provided, the result will be printed.")
    parser.add_argument("--vocab-path", default="model/vocab.pkl", help="Path to the vocabulary file.")
    parser.add_argument("--model-checkpoint-path", default="model/model.pth.tar", help="Path to the model checkpoint.")
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as input_file:
        input_text = input_file.read()

    normalizer = Normalizer()
    vocab_path = os.path.join(args.vocab_path)
    model_checkpoint_path = os.path.join(args.model_checkpoint_path)
    model, vocab, device = load_pre_model(vocab_path=vocab_path, model_checkpoint_path=model_checkpoint_path)

    corrected_text = spell_checking_on_sents(model, vocab, device, normalizer, input_text)
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as output_file:
            output_file.write(corrected_text)
        print("Output successfully written to file!")
    else:
        print(corrected_text)

if __name__ == '__main__':
    main()
