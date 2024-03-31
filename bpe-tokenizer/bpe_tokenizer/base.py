def get_stats(ids: list, counts=None):
    counts = {} if counts is None else counts
    # zip: Pythonic way to iterate consecutive elements, create tuple from
    # two list [i, ..] [i+1,...]
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
    newids = []
    i = 0
    while i < len(ids):
        # if we are not at the very last position AND the pair matches, replace it
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


class Tokenizer:
    """base class for Tokenizers"""

    def __init__(self) -> None:
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()
    
    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError
    
    def encode(self, text: str):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids: list):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError
    
    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocabs = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocabs[idx] = vocabs[p0] + vocabs[p1]
        for special, idx in self.special_tokens.items():
            vocabs[idx] = special.encode("utf-8")
        return vocabs