import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer, MBartForCausalLM
from hftrim.TokenizerTrimmer import TokenizerTrimmer
from hftrim.ModelTrimmers import MBartTrimmer
from loader import DVCDataset
from utils import parse_vtt
from config import *


subtitles = []
for video_id in DVCDataset.load_subset('train'):
    vtt_path = VTT_DIR / f'{video_id}.vtt'
    subtitles.extend([sub['text'] for sub in parse_vtt(vtt_path)])

# Trim tokenizer
tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-cc25', src_lang='en_XX', tgt_lang='en_XX', use_fast=False)
tt = TokenizerTrimmer(tokenizer)
tt.make_vocab(subtitles)
tt.make_tokenizer()
tt.trimmed_tokenizer.save_pretrained('./captioners/trimmed_tokenizer')

# with open('./captioners/trimmed_tokenizer/trimmed_vocab_ids.txt', 'w') as f:
#     for tok_id in tt.trimmed_vocab_ids: 
#         f.write(f'{tok_id}\n')

# Trim model
model = MBartForCausalLM.from_pretrained('facebook/mbart-large-cc25')
mt = MBartTrimmer(model, model.config, tt.trimmed_tokenizer)
mt.make_weights(tt.trimmed_vocab_ids)
mt.make_model()
mt.trimmed_model.save_pretrained('./captioners/trimmed_mbart')