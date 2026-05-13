"""
Written by KrishPro @ KP

filename: `vocab.py`
"""

import argparse
import pandas as pd

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers

def create_tokenizer(iter):

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]"])

    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.enable_padding()

    tokenizer.train_from_iterator(iter, trainer=trainer)

    return tokenizer


def main(csv_path: str, output_path: str):
    csv = pd.read_csv(csv_path)
    
    tokenizer = create_tokenizer(csv['sentence'])
                 
    tokenizer.save(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)

    args = parser.parse_args()
    main(csv_path=args.csv_path, output_path=args.output_path)