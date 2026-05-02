'''Trim mBART tokenizer + model down to the vocabulary actually present in the active dataset's subtitles.

The trim target language and mBART backbone are read from `config` (controlled by the DATASET env var):
    BOBSL    -> mbart-large-cc25, en_XX -> captioners/trimmed_tokenizer_bobsl{,_mbart}
    PHOENIX  -> mbart-large-50,   de_DE -> captioners/trimmed_tokenizer_phoenix{,_mbart}
    CSL      -> mbart-large-50,   zh_CN -> captioners/trimmed_tokenizer_csl{,_mbart}

Run this ONCE per dataset before training. It iterates the train split's VTT files (which for synth datasets
are the per-stream synth VTTs) and collects every subtitle token, then runs the standard hf-trim flow.
'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer, MBartForCausalLM
from hftrim.TokenizerTrimmer import TokenizerTrimmer
from hftrim.ModelTrimmers import MBartTrimmer

from loader import DVCDataset
from utils import parse_vtt
from config import VTT_DIR, MBART_NAME, TGT_LANG, TRIMMED_TOKENIZER_DIR, TRIMMED_MBART_DIR, DATASET


def main():
    print(f'Trimming for DATASET={DATASET}, MBART={MBART_NAME}, TGT_LANG={TGT_LANG}')
    print(f'Output dirs: {TRIMMED_TOKENIZER_DIR}, {TRIMMED_MBART_DIR}')
    subtitles = []
    for video_id in DVCDataset.load_subset('train'):
        vtt_path = VTT_DIR / f'{video_id}.vtt'
        try: subtitles.extend([sub['text'] for sub in parse_vtt(vtt_path)])
        except FileNotFoundError: print(f'missing {vtt_path}')
        
    print(f'Collected {len(subtitles)} training subtitles')
    if not subtitles: raise RuntimeError('No subtitles collected; cannot trim.')

    tokenizer = AutoTokenizer.from_pretrained(MBART_NAME, src_lang=TGT_LANG, tgt_lang=TGT_LANG, use_fast=False)
    tt = TokenizerTrimmer(tokenizer)
    tt.make_vocab(subtitles)
    tt.make_tokenizer()
    tt.trimmed_tokenizer.save_pretrained(TRIMMED_TOKENIZER_DIR)
    print(f'Saved trimmed tokenizer ({len(tt.trimmed_vocab_ids)} ids) to {TRIMMED_TOKENIZER_DIR}')

    model = MBartForCausalLM.from_pretrained(MBART_NAME)
    mt = MBartTrimmer(model, model.config, tt.trimmed_tokenizer)
    mt.make_weights(tt.trimmed_vocab_ids)
    mt.make_model()
    mt.trimmed_model.save_pretrained(TRIMMED_MBART_DIR)
    print(f'Saved trimmed mBART to {TRIMMED_MBART_DIR}')


if __name__ == '__main__':
    main()