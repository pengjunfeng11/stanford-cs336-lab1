import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

re.findall(PAT, "some text that i'll pre-tokenize")

text = ["low low low low low", "lower lower widest widest widest", " newest newest newest newest newest newest"]

word_dict = {}
byte_dict = {}
vocab:dict[int,bytes] = {} # int: token_id, bytes:token bytes
merges:list[tuple[bytes, bytes]] = []

# 初始化vocab字典，将ASCII码0-255的每个字节作为token加入词表
for i in range(256):
    vocab[i] = bytes([i])



def split_to_words(dataset:list[list[str]]):
    temp_set = set()
    for sentence in dataset:
        for word in re.findall(PAT, sentence):
            word = word.strip()
            if word in temp_set:
                word_dict[word] = word_dict[word] + 1
            else:
                word_dict[word] = 1
                temp_set.add(word)

split_to_words(text)

for word, value in word_dict.items():
    byte_dict.update({tuple([b for b in word]): value})
merges = list()
def iter_byte_dict(iter_nums:int):
    selected_pair = ""
    for _ in range(iter_nums):
        temp_dict = {}
        for byte_tuple, value in byte_dict.items():
            byte_list = list(byte_tuple)
            for i in range(len(byte_list)):
                if i+1 >= len(byte_list):
                    break
                
                sub_word = byte_list[i]+byte_list[i+1]
                
                if sub_word in temp_dict:
                    temp_dict[sub_word] = temp_dict[sub_word] + value
                else:
                    temp_dict[sub_word] = value
        # print(temp_dict)
        # 找出字典中value最大的key
        max_value = max(temp_dict.values())
        # 获取所有value等于max_value的key
        max_keys = [k for k, v in temp_dict.items() if v == max_value]
        # 如果有多个最大value的key，按字母表顺序排序并取第一个
        selected_pair = max(max_keys)
        merges.append(selected_pair)
        # print(f"选中的pair是: {selected_pair}, 出现次数为: {max_value}")
        # 替换所有byte_tuple中相邻等于selected_pair的部分
        new_byte_dict = {}
        for byte_tuple, value in byte_dict.items():
            byte_list = list(byte_tuple)
            i = 0
            merged = []
            while i < len(byte_list):
                if i < len(byte_list)-1 and byte_list[i]+byte_list[i+1] == selected_pair:
                    merged.append(selected_pair)
                    i += 2
                else:
                    merged.append(byte_list[i])
                    i += 1
            new_byte_dict[tuple(merged)] = value
        byte_dict.clear()
        byte_dict.update(new_byte_dict)
        print(f"替换后的byte_dict: {new_byte_dict}")


#################### 更新byte_dict中的byte tuple部分，将得到的selected_pair替换为其中的内容 ####################
iter_byte_dict(6)
print(merges)