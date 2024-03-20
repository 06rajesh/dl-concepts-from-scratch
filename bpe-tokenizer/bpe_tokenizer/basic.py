

def get_stats(ids: list):
    counts = {}
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


class BasicTokenizer:
    def __init__(self):
        self.merges = {} # (int, int) -> int
        self.vocabs = {idx: bytes([idx]) for idx in range(256)}
    
    def train(self, text, vocab_size=276, verbose=False):
        num_merges = vocab_size - 256
        tokens = text.encode('utf-8')
        tokens = list(map(int, tokens))
        ids = list(tokens)

        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            if verbose:
                print(f"merging {pair} into a new token {idx}")
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
        
        self._compute_vocabs_after_merge()
        
        if verbose:
            print(f"tokens length: {len(tokens)}")
            print(f"ids lenght: {len(ids)}")
            print(f"compression ration: {len(tokens) / len(ids):.2f}X")
    
    def _compute_vocabs_after_merge(self):
        if len(self.merges) == 0:
            print("No merges found. Can not compute vocabs")
        for (p0, p1), idx in self.merges.items():
            self.vocabs[idx] = self.vocabs[p0] + self.vocabs[p1]

        
    
    def decode(self, ids: list):
        # given ids (list of integers), return Python string
        tokens = b"".join(self.vocabs[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def encode(self, text: str):
        # given a string, return list of integers (the tokens)\
        tokens = list(text.encode("utf-8"))
        
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        
        return tokens
    

if __name__ == "__main__":
    raw_text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. \
We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But \
Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes \
can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after \
Unicode’s inception." 

    tokenizer = BasicTokenizer()
    tokenizer.train(raw_text, verbose=True)

    text3 = "The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizers are a completely separate stage of the LLM pipeline: they have their own training sets, training algorithms (Byte Pair Encoding), and after training implement two fundamental functions: encode() from strings to tokens, and decode() back from tokens to strings."
    encoded = tokenizer.encode(text3)
    decoded = tokenizer.decode(encoded)
    assert text3 == decoded
    print(decoded)




