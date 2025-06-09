import regex as re
from tqdm import tqdm
import pickle
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

re.findall(PAT, "some text that i'll pre-tokenize")
word_dict = {}
byte_dict = {}
vocab: dict[int, bytes] = {}  # int: token_id, bytes:token bytes
merges: list[tuple[bytes, bytes]] = []
# 检查是否存在缓存的byte_dict
cache_path = "/mnt/d/WorkSpace/cs336/lab1/assignment1-basics/cs336_basics/byte_dict.pkl"
if os.path.exists(cache_path):
    print("发现缓存的byte_dict，直接加载...")
    with open(cache_path, "rb") as f:
        byte_dict = pickle.load(f)
else:
    # 读取文件并按照<|endoftext|>标签分割
    with open(
        "/mnt/d/WorkSpace/cs336/lab1/assignment1-basics/cs336_basics/data/TinyStoriesV2-GPT4-train.txt",
        "r",
        encoding="utf-8",
    ) as f:
        text = f.read().split("<|endoftext|>")
        # 移除空字符串
        text = [t.strip() for t in tqdm(text, desc="处理文本分段") if t.strip()]

    # 初始化vocab字典，将ASCII码0-255的每个字节作为token加入词表
    for i in tqdm(range(256), desc="初始化vocab"):
        vocab[i] = bytes([i])

    def split_to_words(dataset: list[list[str]]):
        temp_set = set()
        for sentence in tqdm(dataset, desc="处理句子"):
            for word in tqdm(re.findall(PAT, sentence), desc="处理单词", leave=False):
                word = word.strip()
                if word in temp_set:
                    word_dict[word] = word_dict[word] + 1
                else:
                    word_dict[word] = 1
                    temp_set.add(word)

    split_to_words(text)

    for word, value in tqdm(word_dict.items(), desc="构建byte_dict"):
        byte_dict.update({tuple([b for b in word]): value})
# 将byte_dict保存到本地
# 保存byte_dict到文件
with open(
    "/mnt/d/WorkSpace/cs336/lab1/assignment1-basics/cs336_basics/byte_dict.pkl", "wb"
) as f:
    pickle.dump(byte_dict, f)

merges = list()


def iter_byte_dict(iter_nums: int):
    selected_pair = ""
    for _ in tqdm(range(iter_nums), desc="迭代次数"):
        temp_dict = {}
        for byte_tuple, value in tqdm(
            byte_dict.items(), desc="处理byte_tuple", leave=False
        ):
            byte_list = list(byte_tuple)
            for i in range(len(byte_list)):
                if i + 1 >= len(byte_list):
                    break

                sub_word = byte_list[i] + byte_list[i + 1]

                if sub_word in temp_dict:
                    temp_dict[sub_word] = temp_dict[sub_word] + value
                else:
                    temp_dict[sub_word] = value

        max_value = max(temp_dict.values())
        max_keys = [k for k, v in temp_dict.items() if v == max_value]
        selected_pair = max(max_keys)
        merges.append(selected_pair)

        new_byte_dict = {}
        for byte_tuple, value in tqdm(
            byte_dict.items(), desc="更新byte_dict", leave=False
        ):
            byte_list = list(byte_tuple)
            i = 0
            _merged_temp_list = []
            while i < len(byte_list):
                if (
                    i < len(byte_list) - 1
                    and byte_list[i] + byte_list[i + 1] == selected_pair
                ):
                    _merged_temp_list.append(selected_pair)
                    i += 2
                else:
                    _merged_temp_list.append(byte_list[i])
                    i += 1
            new_byte_dict[tuple(_merged_temp_list)] = value
        byte_dict.clear()
        byte_dict.update(new_byte_dict)
        # print(f"替换后的byte_dict: {new_byte_dict}")


#################### 更新byte_dict中的byte tuple部分，将得到的selected_pair替换为其中的内容 ####################
iter_byte_dict(10000)
print(merges)

# 保存merges到文件
merges_path = "/mnt/d/WorkSpace/cs336/lab1/assignment1-basics/cs336_basics/merges.pkl"
with open(merges_path, "wb") as f:
    pickle.dump(merges, f)

# 构建并保存完整的vocab
final_vocab = {}
# 初始化基础字节
for i in range(256):
    final_vocab[i] = bytes([i])

# 根据merges更新vocab
for idx, merge_pair in enumerate(merges, start=256):
    final_vocab[idx] = merge_pair

# 保存vocab到文件
vocab_path = "/mnt/d/WorkSpace/cs336/lab1/assignment1-basics/cs336_basics/vocab.pkl"
with open(vocab_path, "wb") as f:
    pickle.dump(final_vocab, f)

print(f"Merges已保存至: {merges_path}")
print(f"Vocab已保存至: {vocab_path}")
print(f"Vocab大小: {len(final_vocab)}")
