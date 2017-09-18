import collections
import glob
import io
import itertools
import MeCab
import numpy as np
import unicodedata

# データセット型を定義
DataSet = collections.namedtuple('DataSet', ('sequences', 'labels'))

# 1行ごとに形態素解析
category_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
category_names_to_index = dict(zip(category_names, range(len(category_names))))
sequence_length = 50 # 一行あたりの単語数

def preprocess_single_line(line):
    # NKFC正規化(全角を半角に)
    # 小文字化
    # 末尾の改行除去
    return unicodedata.normalize('NFKC', line).lower().strip("\n")

def preprocess_dataset(paths, skip_lines=0, limit_lines=0):
    sequences = []
    labels = []
    count_line = 0
    for filename in sorted(glob.glob(paths)):
        with open(filename, "r", encoding="utf-8") as f:
            for text in f:
                # TSV処理
                tsv = text.split(",")
                label = tsv[1]
                line = ",".join(tsv[2:])

                # ラベル
                label = category_names_to_index.get(label, None)
                if label is None:
                    try:
                        # 本来IPCにない数字が含まれているが
                        # 数字として処理
                        label = int(label)
                    except:
                        label = 0

                # 行範囲取得
                if limit_lines != 0:
                    lines = line.strip("\n").split("。")
                    lines = lines[skip_lines:skip_lines+limit_lines]
                    line = "。".join(lines)

                # 1行処理
                line = preprocess_single_line(line)

                # 追加
                sequences.append(line)
                labels.append(label)

                # ログ
                if count_line % 10000 == 0:
                    print("preproces line: ", count_line)
                count_line += 1

    print("preproces line: ", count_line)
    return DataSet(sequences, labels)

def parse(mecab, text):
    token_list = []
    node = mecab.parseToNode(text)
    while node:
        feature = node.feature.split(',')

        # 品詞フィルタ・基本形補正を使わない(データセットが十分大きいため)
        #print(feature)
        pos_tag = feature[0]

        # BOS/EOS(Begin of sentence / End of sentence)のときは処理しない
        if pos_tag not in ["BOS/EOS"]:
            word = node.surface
            token_list.append(word)
        node = node.next
    return token_list


def parse_dataset(src_dataset):
    # MeCabを初期化
    mecab = MeCab.Tagger('-Ochasen')

    # node.surfaceを呼び出すとエラーになるMeCabのバグの回避策
    mecab.parse('')

    dst_dataset = DataSet([], [])
    count_line = 0
    for label, line in zip(src_dataset.labels, src_dataset.sequences):

        # 形態素
        tokens = parse(mecab, line)

        if not tokens:
            tokens = ["。"]

        # 追加
        dst_dataset.sequences.append(tokens)
        dst_dataset.labels.append(label)

        if count_line % 10000 == 0:
            print("parse line: ", count_line)
        count_line += 1
    
    print("parse line: ", count_line)
    return dst_dataset

def flatten(sequences):
    # 単語の配列にする
    return [word for words in sequences \
                 for word in words]

def create_word_index(dataset):
    padding_token = "<PAD>"

    # 頻度順にデータを取得(0番はpadding)
    word_counts = collections.Counter(
        flatten(dataset.sequences))
    vocabulary = [padding_token]
    vocabulary += [x[0] for x in word_counts.most_common()]

    # インデックス(単語→id)を作成
    word_index = {x: i for i, x in enumerate(vocabulary)}

    # 逆インデックス(id→単語)を作成
    reverse_index = {x: i for x, i in word_index.items()}
    return vocabulary, word_index, reverse_index

def dataset_shuffle(dataset):
    # 乱数を固定
    np.random.seed(0)
    indexes = np.random.permutation(len(dataset.sequences))
    sequences = [dataset.sequences[idx] for idx in indexes]
    labels = [dataset.labels[idx] for idx in indexes]
    return DataSet(sequences, labels)

def split_dataset(dataset, train_rate):
    train_sequences = []
    train_labels = []
    test_sequences = []
    test_labels = []
    
    for category_label in range(len(category_names)):
        # カテゴリで絞込
        category_indexes = [idx for idx, label in enumerate(dataset.labels) if label == category_label]
        categorywise_train_size = int(len(category_indexes) * train_rate)
        
        # 学習データ
        train_category_indexes = category_indexes[:categorywise_train_size]
        train_sequences += [dataset.sequences[idx] for idx in train_category_indexes]
        train_labels += [dataset.labels[idx] for idx in train_category_indexes]
        
        # 評価データ
        test_category_indexes = category_indexes[categorywise_train_size:]
        test_sequences += [dataset.sequences[idx] for idx in test_category_indexes]
        test_labels += [dataset.labels[idx] for idx in test_category_indexes]
        
    train = DataSet(train_sequences, train_labels)
    test = DataSet(test_sequences, test_labels)
    return dataset_shuffle(train), dataset_shuffle(test)
    
def upsampling(dataset):
    max_size = max(sum(1 for label in dataset.labels \
                if label == category_label) \
                for category_label in range(len(category_names)))
    sequences = []
    labels = []
    for category_label in range(len(category_names)):
        category_indexes = [idx for idx, label in enumerate(dataset.labels) if label == category_label]
        category_indexes = np.random.choice(category_indexes, max_size)
        
        sequences += [dataset.sequences[idx] for idx in category_indexes]
        labels += [dataset.labels[idx] for idx in category_indexes]
    
    return dataset_shuffle(DataSet(sequences, labels))

def cut_and_padding(ls, length, pad):
    # カット
    ls = ls[:length]
    # パディング
    if length > len(ls):
         ls += [pad] * (length - len(ls))
    return ls

def convert_word_to_index(dataset, word_index):
    new_sequences = []
    new_labels = []
    
    # データセットごとの処理
    for line, label in zip(dataset.sequences, dataset.labels):
        
        # 行ごとの処理
        # 1行分をword indexを用いて変換
        new_line = []
        for word in line:
            if word in word_index:
                new_line.append(word_index[word])

        # データがない場合はcontinue (全形態素が未知語の場合とか)
        if len(new_line) == 0:
            continue

        # カット & パディング
        new_line = cut_and_padding(
            new_line,
            sequence_length,
            pad=0)

        # 追加
        new_sequences.append(new_line)
        new_labels.append(label)
    return DataSet(np.array(new_sequences), np.array(new_labels))
