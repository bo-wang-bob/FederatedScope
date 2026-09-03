import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate local MDSent data and BERT model files.')
    parser.add_argument('--data-root',
                        default='data/sentiment',
                        help='Path to the Multi-Domain Sentiment dataset.')
    parser.add_argument('--model-dir',
                        default='pretrained_models/'
                        'nlptown_bert_base_multilingual_uncased_senti',
                        help='Path to the local BERT model directory.')
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    model_dir = Path(args.model_dir)

    required_domains = ['books', 'dvd', 'electronics', 'kitchen']
    for domain in required_domains:
        domain_dir = data_root / domain
        if not domain_dir.is_dir():
            raise FileNotFoundError(f'Missing domain directory: {domain_dir}')
        for filename in ['negative.review', 'positive.review']:
            review_file = domain_dir / filename
            if not review_file.is_file():
                raise FileNotFoundError(f'Missing review file: {review_file}')

    required_model_files = [
        'config.json',
        'model.safetensors',
        'special_tokens_map.json',
        'tokenizer_config.json',
        'vocab.txt',
    ]
    for filename in required_model_files:
        model_file = model_dir / filename
        if not model_file.is_file():
            raise FileNotFoundError(f'Missing model file: {model_file}')

    from transformers import AutoModel, AutoTokenizer

    AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    AutoModel.from_pretrained(str(model_dir), local_files_only=True)

    print('mdsent setup ok')


if __name__ == '__main__':
    main()
