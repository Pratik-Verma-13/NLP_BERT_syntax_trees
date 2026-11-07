from collections import Counter

def parse_conllu(file_path):
    print(f"Parsing dataset: {file_path}")
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            parts = line.split('\t')
            if len(parts) >= 2 and parts[0].isdigit():
                sentence.append(parts[1].lower())
        if sentence:
            sentences.append(sentence)
    return sentences

def build_vocabulary(sentences, min_freq=2):
    print(f"Building vocabulary (min_freq={min_freq})...")
    counts = Counter(w for sent in sentences for w in sent)
    vocab = [w for w, c in counts.items() if c >= min_freq]
    word2idx = {w: i for i, w in enumerate(vocab)}
    print(f"Vocabulary size: {len(word2idx)} unique words.")
    return word2idx, vocab