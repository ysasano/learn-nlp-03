import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import embed_sequence

def get_param(vocab_size,
    category_size,
    lstm,
    embed_dim=100,
    learning_rate=0.001):
    return {
        "vocab_size": vocab_size, # 語彙数
        "category_size": category_size, # カテゴリ数
        "lstm": lstm, # LSTMの使用有無
        "embed_dim": embed_dim, # 分散表現の次元数
        "learning_rate": learning_rate, # 学習率
    }

def model_fn(features, labels, mode, params):

    sequences = features["sequences"]

    # 分散表現を取得
    emb_sequences = embed_sequence(
        sequences,
        params["vocab_size"],
        params["embed_dim"],
        initializer = tf.random_uniform_initializer(-1,1))

    # 文章の長さを取得
    mask = tf.to_int32(
        tf.not_equal(sequences, tf.zeros_like(sequences)))
    length = tf.reduce_sum(mask, axis=-1)

    print(params)

    if params["lstm"] == 1:
        # RNN(LSTM / 双方向)を実行
        cell = tf.nn.rnn_cell.LSTMCell(num_units=params["embed_dim"])
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell,
            cell_bw=cell,
            inputs=emb_sequences,
            dtype=tf.float32,
            sequence_length=length)
        output_fw, output_bw = outputs
        states_fw, states_bw = states

        # 双方向の出力を結合
        output = tf.concat([output_fw, output_bw], axis=-1)
    else:
        output = emb_sequences

    # 出力の総和を取る(average pooling)
    mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
    length = tf.expand_dims(tf.cast(length, tf.float32), -1)
    logits = tf.reduce_sum(emb_sequences * mask, 1) / length
    logits = layers.dense(logits, params["category_size"])

    # 結果出力の準備 (結果出力モード)
    predictions = {
        "classes": tf.argmax(logits, axis=1), # 1位のカテゴリ
        "probabilities": tf.nn.softmax(logits, name="probabilities") # 識別確率
    }

    # 結果出力モード
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    # lossの計算 (学習モード / 評価モード)
    onehot_labels = tf.one_hot(
        indices=tf.to_int32(labels),
        depth=params["category_size"])
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels,
        logits=logits)

    # 学習モード
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)

    # 評価値の計算 (評価モード)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"])
    }

    # 評価モード
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops)
