import heapq
import json
import mmap
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import regex

TokenId = int
TokenPair = tuple[TokenId, TokenId]
WordCountDict = defaultdict[tuple[TokenId, ...], int]

# First 256 ids are reserved for byte-level tokens (0x00-0xFF).
BASE_VOCAB_SIZE = 256
# GPT-2 style regex: splits on Unicode letters, numbers, whitespace, contractions.
GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
GPT2_REGEX = regex.compile(GPT2_PATTERN)


def _compile_special_tokens_pattern(
    special_tokens: dict[str, TokenId],
) -> regex.Pattern[str] | None:
    """Compile a regex that captures special token boundaries.

    Args:
        special_tokens: Mapping from special token text to token id.

    Returns:
        A compiled regex that captures any special token, or None when no
        special tokens are configured.
    """
    if not special_tokens:
        return None

    pattern = r"(" + "|".join([regex.escape(token) for token in special_tokens]) + r")"
    return regex.compile(pattern)


def _split_text_by_special_tokens(
    text: str, special_tokens_pattern: regex.Pattern[str] | None
) -> list[str]:
    """Split text into chunks around special token boundaries.

    Args:
        text: Input string that may contain special tokens.
        special_tokens_pattern: Compiled pattern for special tokens.

    Returns:
        List of chunks (alternating text and matched special tokens). If no
        pattern is provided, returns [text].
    """
    if not special_tokens_pattern:
        return [text]

    return special_tokens_pattern.split(text)


def _get_worker_segment_boundaries(
    file_path: Path, special_tokens: dict[str, TokenId], num_desired_chunks: int
) -> list[int]:
    """Find byte boundaries to split the corpus for parallel workers.

    Splits at newlines and/or special-token boundaries to avoid cutting
    special tokens across segment boundaries.

    Args:
        num_desired_chunks: Target number of segments (typically num_workers).

    Returns:
        List of byte offsets [b0, b1, ..., bn] where segments are
        (b0,b1), (b1,b2), ..., (b_{n-1}, bn). b0=0, bn=file_size.
    """
    # Match newlines; if special tokens exist, also match them.
    if not special_tokens:
        delimiter_pattern = regex.compile(b"\n")
    else:
        escaped = b"|".join(
            [regex.escape(token.encode("utf-8")) for token in special_tokens]
        )
        delimiter_pattern = regex.compile(b"\n|(?:" + escaped + b")")

    file_size = file_path.stat().st_size
    chunk_size = file_size // num_desired_chunks
    boundaries = [0]

    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for i in range(1, num_desired_chunks):
                target = i * chunk_size

                # Check if the previous search already jumped past this target.
                if target <= boundaries[-1]:
                    continue

                boundary = delimiter_pattern.search(mm, target)
                if boundary:
                    boundaries.append(boundary.end())
                else:
                    break  # No more delimiters; stop early.

    # Ensure last segment reaches end of file.
    if boundaries[-1] != file_size:
        boundaries.append(file_size)

    return boundaries


def _pretokenize_worker(args: tuple[Path, int, int, dict[str, int]]) -> WordCountDict:
    """Pretokenize a text segment and count all words in this segment

    Pool.map passes one argument per call; we pack
    (file_path, start, end, special_tokens) into a tuple for this reason.

    Args:
        args: (file_path, start_byte, end_byte, special_tokens).

    Returns:
        WordCountDict mapping (word_ids,) to frequency within this segment.
    """
    file_path, start, end, special_tokens = args
    special_tokens_pattern = _compile_special_tokens_pattern(special_tokens)

    word_counts: WordCountDict = defaultdict(int)
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            with memoryview(mm)[start:end] as segment:
                # Decode only this segment; no full-file read.
                segment_text = bytes(segment).decode("utf-8")
                chunks = _split_text_by_special_tokens(
                    segment_text, special_tokens_pattern
                )

                for chunk in chunks:
                    if chunk in special_tokens:
                        continue
                    words: list[str] = GPT2_REGEX.findall(chunk)
                    for word in words:
                        word_ids = tuple(word.encode("utf-8"))
                        word_counts[word_ids] += 1

    return word_counts


def _merge_word_counts(word_counts_list: list[WordCountDict]):
    """Merge word counts from multiple segments into parallel word list + freq list.

    Deduplicates words across segments and sums their counts. Returns two
    parallel lists: encoded words (as mutable list[TokenId]) and their freqs.

    Args:
        per_segment_word_counts: List of WordCountDict, one per worker segment.

    Returns:
        (encoded_words, word_freqs, pair_counts, pair_map) where:
        - encoded_words[i] is list[TokenId] for unique word i.
        - word_freqs[i] is the corpus frequency of encoded_words[i].
        - pair_counts[pair] is weighted occurrence count of pair across corpus.
        - pair_map[pair] is set of word indices that currently contain pair.
    """
    encoded_words: list[list[TokenId]] = []
    word_freqs: list[int] = []
    word_to_idx: dict[tuple[TokenId, ...], int] = {}

    for word_counts in word_counts_list:
        for word_ids, count in word_counts.items():
            if word_ids not in word_to_idx:
                word_to_idx[word_ids] = len(encoded_words)
                encoded_words.append(list(word_ids))
                word_freqs.append(count)
            else:
                idx = word_to_idx[word_ids]
                word_freqs[idx] += count

    # Initial counts of all pairs in the training text.
    pair_counts: defaultdict[TokenPair, int] = defaultdict(int)
    # Inverted index: pair -> set of word indices that contain this pair.
    pair_map: defaultdict[TokenPair, set[int]] = defaultdict(set)

    for i, (word, count) in enumerate(zip(encoded_words, word_freqs)):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += count
            pair_map[pair].add(i)

    return encoded_words, word_freqs, pair_counts, pair_map


def _find_best_pair(heap, pair_counts):
    best_pair = None
    while heap:
        neg_freq, pair = heapq.heappop(heap)
        if pair in pair_counts and pair_counts[pair] == -neg_freq:
            best_pair = pair
            break

    return best_pair


def _apply_merge_in_place(
    ids: list[TokenId], best_pair: tuple[TokenId, TokenId], new_id: TokenId
) -> None:
    i = 0
    write_idx = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i + 1] == best_pair[1]:
            ids[write_idx] = new_id
            i += 2
        else:
            ids[write_idx] = ids[i]
            i += 1
        write_idx += 1

    del ids[write_idx:]


class Tokenizer:
    def __init__(
        self,
        file_path: str | Path | None,
        vocab_size: int,
        special_tokens: dict[str, TokenId] | None = None,
        num_workers: int = 4,
    ):
        self.file_path = Path(file_path) if file_path else None
        self.vocab_size = vocab_size
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be >= {BASE_VOCAB_SIZE}, got {vocab_size}"
            )

        self.special_tokens = special_tokens if special_tokens else {}
        self.num_workers = num_workers

        # Rules to merge token in encode
        self.merge_rules: dict[TokenPair, TokenId] = {}

        # Rules to decode token id
        self.vocab = {id: bytes([id]) for id in range(BASE_VOCAB_SIZE)}

        # Inverse map for decode: token_id -> bytes for special tokens only.
        self.inverse_special_tokens: dict[TokenId, bytes] = {}
        for token, token_id in self.special_tokens.items():
            if token_id < self.vocab_size:
                raise ValueError(
                    f"special token id {token_id} must be >= vocab_size "
                    f"({self.vocab_size}); ids in [0, vocab_size) are reserved"
                )
            if token_id in self.inverse_special_tokens:
                raise ValueError(f"duplicate special token id {token_id}")
            self.inverse_special_tokens[token_id] = token.encode("utf-8")

        # Regex to split text on special token boundaries (capturing group).
        self.special_tokens_pattern = _compile_special_tokens_pattern(
            self.special_tokens
        )

        # Used by decode(); includes byte vocab (+ merges once learned)
        # and special tokens.
        self.unified_vocab: dict[TokenId, bytes] = {}
        self._merge_vocab()

        # FIFO cache for encode_word results; improves repeated encode calls.
        self.cache: dict[str, tuple[TokenId, ...]] = {}
        self.max_cache_size = 32768

    def _merge_vocab(self) -> None:
        """Build a unified vocab for fast decode lookup.

        Merges the regular vocab (byte tokens + learned merges) with the special
        token inverse map. Call this after train() and before decode().

        Returns:
            None. Updates self.unified_vocab in place.
        """
        self.unified_vocab = {**self.vocab, **self.inverse_special_tokens}

    def train(self) -> None:
        """Learn merge rules using parallel pretokenization and in-place merges.

        Splits corpus into segments, pretokenizes in parallel, merges counts,
        then iteratively applies BPE merges in place on the unified word list.

        Maintains two synchronized structures for fast updates:
        - pair_counts: weighted occurrence count of each pair.
        - pair_map: word-index membership for each pair.

        Returns:
            None. Populates self.merge_rules and self.vocab; calls merge_vocab().
        """
        if self.file_path is None:
            raise ValueError("file_path must be provided to train tokenizer")

        boundaries = _get_worker_segment_boundaries(
            self.file_path, self.special_tokens, self.num_workers
        )
        segments = [
            (self.file_path, start, end, self.special_tokens)
            for start, end in zip(boundaries, boundaries[1:])
        ]

        with Pool(self.num_workers) as p:
            word_counts_list = p.map(_pretokenize_worker, segments)

        encoded_words, word_freqs, pair_counts, pair_map = _merge_word_counts(
            word_counts_list
        )
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        num_merges = self.vocab_size - len(self.vocab)
        for _ in range(num_merges):
            if not pair_counts:
                break

            best_pair = _find_best_pair(heap, pair_counts)
            if best_pair is None:
                raise RuntimeError(
                    "Inconsistent heap state: pair_counts is non-empty but "
                    "no valid heap entry was found. pair_counts is the ground truth "
                    "and heap is lazily updated; this indicates missing/incorrect "
                    f"heap pushes. pair_counts_size={len(pair_counts)}, "
                    f"remaining_heap_entries={len(heap)}"
                )

            # Pick highest-frequency pair (deterministic tie-break by pair value).
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merge_rules[best_pair] = new_id

            # Only words containing best_pair can change after this merge.
            affected_indices = list(pair_map[best_pair])
            for idx in affected_indices:
                word_ids = encoded_words[idx]
                freq = word_freqs[idx]

                # Pair counts before the merge.
                old_pair_counts = Counter(zip(word_ids, word_ids[1:]))
                # Apply the merge in place.
                _apply_merge_in_place(word_ids, best_pair, new_id)
                # Pair counts after the merge.
                new_pair_counts = Counter(zip(word_ids, word_ids[1:]))

                # Get all pairs before and after the merge.
                all_changed_pairs = old_pair_counts.keys() | new_pair_counts.keys()
                for pair in all_changed_pairs:
                    old_count, new_count = old_pair_counts[pair], new_pair_counts[pair]
                    if old_count == new_count:
                        continue

                    if old_count == 0 and new_count > 0:
                        # Pair is new.
                        pair_map[pair].add(idx)
                    elif old_count > 0 and new_count == 0:
                        # Pair is gone.
                        pair_map[pair].remove(idx)
                    else:
                        # Pair is still there, but count has changed.
                        pass

                    pair_counts[pair] += (new_count - old_count) * freq
                    updated_count = pair_counts[pair]
                    if updated_count > 0:
                        heapq.heappush(heap, (-updated_count, pair))
                    else:
                        del pair_counts[pair]
                        pair_map.pop(pair)

        self._merge_vocab()

    def save(self, save_path: str | Path) -> None:
        """Save tokenizer state to disk.

        Args:
            save_path: Target JSON file path, typically under `weights/`.
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        merge_rules_serialized = [
            [pair[0], pair[1], token_id]
            for pair, token_id in sorted(
                self.merge_rules.items(), key=lambda item: item[1]
            )
        ]
        payload = {
            "format_version": 1,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "num_workers": self.num_workers,
            "merge_rules": merge_rules_serialized,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        load_path: str | Path,
        file_path: str | Path | None = None,
        num_workers: int | None = None,
    ) -> "Tokenizer":
        """Load a tokenizer checkpoint and reconstruct vocab/merge rules.

        Args:
            load_path: Path to tokenizer checkpoint created by `save()`.
            file_path: Optional corpus path for future re-training.
            num_workers: Optional override for workers when re-training.
        """
        path = Path(load_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError(
                f"Unsupported tokenizer format_version={payload.get('format_version')}"
            )

        resolved_num_workers = (
            num_workers if num_workers is not None else payload.get("num_workers", 4)
        )
        special_tokens = {
            token: int(token_id)
            for token, token_id in payload.get("special_tokens", {}).items()
        }
        tokenizer = cls(
            file_path=file_path,
            vocab_size=int(payload["vocab_size"]),
            special_tokens=special_tokens,
            num_workers=int(resolved_num_workers),
        )

        tokenizer.merge_rules = {}
        tokenizer.vocab = {id: bytes([id]) for id in range(BASE_VOCAB_SIZE)}
        for left, right, token_id in sorted(
            payload.get("merge_rules", []), key=lambda item: int(item[2])
        ):
            pair = (int(left), int(right))
            new_id = int(token_id)
            tokenizer.merge_rules[pair] = new_id
            tokenizer.vocab[new_id] = (
                tokenizer.vocab[pair[0]] + tokenizer.vocab[pair[1]]
            )

        tokenizer._merge_vocab()
        tokenizer.cache.clear()
        return tokenizer

    def _encode_word(self, word: str) -> tuple[TokenId, ...]:
        """Encode a single pre-tokenized word into token ids using merge rules.

        Starts with byte-level ids and greedily applies merges in order of
        rule precedence until no more apply.

        Args:
            word: A single "word" from GPT-2 style pre-tokenization (letters,
                numbers, punctuation, etc.). Must be valid UTF-8.

        Returns:
            Tuple of token ids representing the encoded word (cached).
        """
        if word in self.cache:
            return self.cache[word]

        # Start with byte-level ids (0-255).
        ids = list(word.encode("utf-8"))
        while len(ids) >= 2:
            # Pick the earliest applicable merge (by rule order).
            best_pair = min(
                (pair for pair in zip(ids, ids[1:]) if pair in self.merge_rules),
                key=lambda p: self.merge_rules[p],
                default=None,
            )

            if not best_pair:
                break

            _apply_merge_in_place(ids, best_pair, self.merge_rules[best_pair])

        # FIFO eviction when cache is full.
        if len(self.cache) >= self.max_cache_size:
            # Pop the first key (oldest) - Python dicts preserver order!
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        word_ids = tuple(ids)
        # Store in cache for future calls.
        self.cache[word] = word_ids
        return word_ids

    def _encode_chunk(self, chunk: str) -> list[TokenId]:
        """Encode a chunk of text (no special tokens inside) into token ids.

        Args:
            chunk: Text segment that does not contain any special token strings.

        Returns:
            List of token ids for the entire chunk.
        """
        ids: list[TokenId] = []
        words = GPT2_REGEX.findall(chunk)

        for word in words:
            word_ids = self._encode_word(word)
            ids.extend(word_ids)

        return ids

    def encode(self, text: str) -> list[TokenId]:
        """Encode text into token ids, preserving explicit special tokens.

        Splits on special token boundaries first; special token chunks are
        emitted as their fixed ids; other chunks are encoded normally.

        Args:
            text: Input string (may contain special token substrings).

        Returns:
            List of token ids for the full sequence.
        """
        chunks = _split_text_by_special_tokens(text, self.special_tokens_pattern)

        ids = []
        for chunk in chunks:
            # Chunk is either a special token (emit id) or normal text (encode).
            if chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
            else:
                ids.extend(self._encode_chunk(chunk))

        return ids

    def decode(self, token_ids: list[TokenId]) -> str:
        """Decode token ids back to a UTF-8 string.

        Args:
            token_ids: Sequence of token ids from encode().

        Returns:
            Decoded UTF-8 string.

        Raises:
            ValueError: If any token id is unknown (not in unified_vocab).
                Call merge_vocab() after train() before using decode().
        """
        try:
            text_bytes = b"".join([self.unified_vocab[id] for id in token_ids])
        except KeyError as e:
            raise ValueError(f"Invalid token id {e.args[0]}")

        text = text_bytes.decode("utf-8", errors="replace")
        return text
