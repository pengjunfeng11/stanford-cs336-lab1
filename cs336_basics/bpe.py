import regex as re
from tqdm import tqdm
import pickle
import os
from typing import Any, Optional

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # 创建反向词汇表：从bytes到token_id
        self.byte_to_id = {v: k for k, v in vocab.items()}

        # 创建merge规则的查找表
        self.merge_rules = {}
        for i, (first, second) in enumerate(merges):
            self.merge_rules[(first, second)] = first + second

        # 处理特殊token
        self.special_token_to_id = {}
        self.id_to_special_token = {}

        # 为特殊token分配ID（与GPT-2参考实现一致）
        # 特殊token的ID应该是len(vocab) - 1（如果vocab已经包含特殊token）
        # 或者len(original_vocab)（如果vocab不包含特殊token）
        # 从适配器代码来看，特殊token已经被添加到vocab中
        # 所以特殊token的ID应该是len(vocab) - 1
        for i, token in enumerate(self.special_tokens):
            token_id = len(self.vocab) - len(self.special_tokens) + i
            token_bytes = token.encode("utf-8")
            self.special_token_to_id[token] = token_id
            self.id_to_special_token[token_id] = token
            # 更新vocab和byte_to_id（替换原有的token）
            self.vocab[token_id] = token_bytes
            self.byte_to_id[token_bytes] = token_id

    def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
        """应用BPE合并规则"""
        if len(tokens) <= 1:
            return tokens

        # 重复应用合并规则直到没有更多合并可以进行
        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

            # 找到第一个可以合并的pair（按照merges的顺序）
            merge_found = False
            for merge_pair in self.merges:
                if merge_pair in pairs:
                    # 找到第一个出现的位置
                    idx = pairs.index(merge_pair)
                    # 执行合并
                    new_tokens = (
                        tokens[:idx]
                        + [merge_pair[0] + merge_pair[1]]
                        + tokens[idx + 2 :]
                    )
                    tokens = new_tokens
                    merge_found = True
                    break

            if not merge_found:
                break

        return tokens

    def encode(self, text: str) -> list[int]:
        """将文本编码为token ID列表"""
        # 如果没有特殊token，直接处理文本
        if not self.special_tokens:
            return self._encode_text(text)
        
        # 构建正则表达式来匹配特殊token
        escaped_tokens = [re.escape(token) for token in self.special_tokens]
        special_token_pattern = '(' + '|'.join(escaped_tokens) + ')'
        
        # 分割文本，保留特殊token
        parts = re.split(special_token_pattern, text)
        
        result = []
        for part in parts:
            if part in self.special_tokens:
                # 特殊token，直接添加其ID
                result.append(self.special_token_to_id[part])
            else:
                # 普通文本，正常编码
                if part:  # 跳过空字符串
                    result.extend(self._encode_text(part))
        
        return result
    
    def _encode_text(self, text: str) -> list[int]:
        """编码不包含特殊token的文本"""
        # 使用正则表达式进行预分词
        words = re.findall(PAT, text)
        
        token_ids = []
        for word in words:
            # 将单词转换为字节序列
            word_bytes = word.encode("utf-8")
            
            # 将字节序列分解为单个字节
            tokens = [bytes([b]) for b in word_bytes]
            
            # 应用BPE合并
            tokens = self._apply_merges(tokens)
            
            # 转换为token ID
            for token in tokens:
                if token in self.byte_to_id:
                    token_ids.append(self.byte_to_id[token])
                else:
                    # 如果token不在词汇表中，分解为字节
                    for byte_val in token:
                        token_ids.append(byte_val)
        
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """将token ID列表解码为文本"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_special_token:
                # 特殊token直接添加字符串
                tokens.append(self.id_to_special_token[token_id])
            elif token_id in self.vocab:
                tokens.append(self.vocab[token_id])
            else:
                # 未知token，跳过或使用替代字符
                continue

        # 将bytes连接并解码为字符串
        result_bytes = b""
        result_parts = []

        for token in tokens:
            if isinstance(token, str):
                # 特殊token，先处理之前的bytes
                if result_bytes:
                    try:
                        result_parts.append(
                            result_bytes.decode("utf-8", errors="replace")
                        )
                    except:
                        result_parts.append(
                            result_bytes.decode("utf-8", errors="ignore")
                        )
                    result_bytes = b""
                result_parts.append(token)
            else:
                result_bytes += token

        # 处理剩余的bytes
        if result_bytes:
            try:
                result_parts.append(result_bytes.decode("utf-8", errors="replace"))
            except:
                result_parts.append(result_bytes.decode("utf-8", errors="ignore"))

        return "".join(result_parts)

    def encode_iterable(self, iterable):
        """内存高效地编码一个字符串迭代器（如文件对象）"""
        for text in iterable:
            yield from self.encode(text)





def _get_gpt2_bytes_to_unicode():
    """返回GPT-2字节到Unicode字符的映射"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer from input text file using GPT-2 byte encoding.
    
    Args:
        input_path: Path to the training text file
        vocab_size: Target vocabulary size (including special tokens)
        special_tokens: List of special tokens to add
        
    Returns:
        tuple of (vocab, merges)
        vocab: dict mapping token_id -> bytes
        merges: list of (bytes, bytes) pairs
    """
    if special_tokens is None:
        special_tokens = []
    
    # GPT-2字节编码映射
    byte_encoder = _get_gpt2_bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    
    # 读取训练数据
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 预处理：分离特殊token和普通文本
    processed_segments = []
    if special_tokens:
        # 构建正则表达式来匹配特殊token
        escaped_tokens = [re.escape(token) for token in special_tokens]
        special_token_pattern = '(' + '|'.join(escaped_tokens) + ')'
        
        # 分割文本，保留特殊token
        parts = re.split(special_token_pattern, text)
        
        for part in parts:
            if part in special_tokens:
                # 特殊token，直接添加
                processed_segments.append(('special', part))
            else:
                # 普通文本，进行预分词
                words = re.findall(PAT, part)
                for word in words:
                    if word:
                        processed_segments.append(('text', word))
    else:
        # 没有特殊token，正常处理
        words = re.findall(PAT, text)
        for word in words:
            if word:
                processed_segments.append(('text', word))
    
    # 将文本转换为GPT-2字节编码（不要strip，保留空格）
    encoded_words = []
    for seg_type, word in processed_segments:
        if seg_type == 'special':
            # 特殊token，直接编码为单个单元
            encoded_word = ''.join(byte_encoder[b] for b in word.encode('utf-8'))
            encoded_words.append(encoded_word)
        else:
            # 普通文本
            encoded_word = ''.join(byte_encoder[b] for b in word.encode('utf-8'))
            encoded_words.append(encoded_word)
    
    # 构建初始的token频率统计
    token_dict = {}
    for word in encoded_words:
        token_dict[word] = token_dict.get(word, 0) + 1
    
    # 初始化词汇表（256个基础字节）
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    
    # 将文本转换为字节元组进行BPE训练
    byte_dict = {}
    for word, count in token_dict.items():
        # 将GPT-2编码字符转换回字节
        word_bytes = bytes(byte_decoder[c] for c in word)
        byte_tuple = tuple(bytes([b]) for b in word_bytes)
        byte_dict[byte_tuple] = count
    
    # 训练BPE merges
    merges = []
    num_merges = vocab_size - 256 - len(special_tokens)
    
    for _ in tqdm(range(num_merges), desc="BPE training"):
        # 统计所有相邻pair的频率
        pair_counts = {}
        for byte_tuple, count in byte_dict.items():
            byte_list = list(byte_tuple)
            for i in range(len(byte_list) - 1):
                pair = (byte_list[i], byte_list[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        
        if not pair_counts:
            break
        
        # 找到频率最高的pair
        max_count = max(pair_counts.values())
        best_pairs = [pair for pair, count in pair_counts.items() if count == max_count]
        
        # 按字节序选择第一个（与参考实现一致）
        best_pair = min(best_pairs)
        
        merges.append(best_pair)
        
        # 在整个byte_dict中应用这个merge
        new_byte_dict = {}
        for byte_tuple, count in byte_dict.items():
            byte_list = list(byte_tuple)
            i = 0
            new_byte_list = []
            while i < len(byte_list):
                if i < len(byte_list) - 1 and (byte_list[i], byte_list[i + 1]) == best_pair:
                    new_byte_list.append(byte_list[i] + byte_list[i + 1])
                    i += 2
                else:
                    new_byte_list.append(byte_list[i])
                    i += 1
            new_byte_dict[tuple(new_byte_list)] = count
        
        byte_dict = new_byte_dict
    
    # 构建最终的vocab
    final_vocab = {}
    for i in range(256):
        final_vocab[i] = bytes([i])
    
    for idx, merge_pair in enumerate(merges, start=256):
        final_vocab[idx] = merge_pair[0] + merge_pair[1]
    
    # 添加特殊token
    max_id = max(final_vocab.keys()) if final_vocab else -1
    for i, token in enumerate(special_tokens):
        token_id = max_id + 1 + i
        final_vocab[token_id] = token.encode('utf-8')
    
    return final_vocab, merges
