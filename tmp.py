def decoder_inference(sentence):
    sentence = preprocess(sentence)
    sentence = tf.expand_dims(SOS + tokenizer.encode(sentence) + EOS, axis=0)
    output_sequence = tf.expand_dims(SOS, 0)
    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output_sequence], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, EOS[0]):
            break
        output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)
    return tf.squeeze(output_sequence, axis=0)


def sentence_generation(sentence):
    prediction = decoder_inference(sentence)
    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size]
    )
    print(f"🧑 : {sentence}")
    print(f"🤖 : {predicted_sentence}")
