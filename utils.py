import json
import torch
from nltk.translate.bleu_score import corpus_bleu


def calculate_bleu(out, img_names, dataset, vocab):
    references = []
    candidates = []
    for idx, out_caption in enumerate(out):
        decoded = vocab.decode_sentence(out_caption.tolist())
        all_caps = dataset.get_captions_for_img(img_names[idx])
        current_candidate = []
        for word in decoded.split(' '):
            if word in ['<end>', '<pad>']:
                break
            current_candidate.append(word)
        candidates.append(current_candidate)

        current_references = []
        for caption in all_caps:
            decoded = vocab.decode_sentence(caption.tolist())
            current_references.append([word for word in decoded.split(' ') if word not in ['<end>', '<pad>']])

        references.append(current_references)

    return corpus_bleu(references, candidates)


def load_json(file):
    with open(file, 'r') as f1:
        return json.loads(f1.read())


def save_model(model, name='saved_models/model'):
    torch.save(model.state_dict(), name)


def load_model(model, name='saved_models/model'):
    model.load_state_dict(torch.load(name))
