from transformers import AutoTokenizer
from hftrim.TokenizerTrimmer import TokenizerTrimmer
from loader import DVCDataset
from utils import parse_vtt
from config import *

subtitles = []
for video_id in DVCDataset.load_subset('train'):
    vtt_path = VTT_DIR / f'{video_id}.vtt'
    subtitles.extend([sub['text'] for sub in parse_vtt(vtt_path)])

tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-cc25', src_lang='en_XX', tgt_lang='en_XX')
tt = TokenizerTrimmer(tokenizer)
tt.make_vocab(subtitles)
tt.make_tokenizer()
tt.trimmed_tokenizer.save_pretrained('./trimmed_tokenizer')

# # Trim model
# model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')
# mt = MBartTrimmer(model, model.config, tt.trimmed_tokenizer)
# mt.make_weights(tt.trimmed_vocab_ids)
# mt.make_model()