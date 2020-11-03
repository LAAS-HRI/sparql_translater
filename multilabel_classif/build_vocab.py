#!/usr/bin/env python
import numpy as np
import argparse
import fasttext


def load_pretrained_model(pretrained_embeddings_file_path):
    print("Loading pre-trained word embeddings...")
    model = fasttext.load_model(pretrained_embeddings_file_path)
    print("Model loaded !")
    return model


def load_vocab(model, data_file_path, augmentation=False):
    word_to_vector = {}
    with open(data_file_path, "r") as file:
        for line in file:
            for word in line.split():
                if not word.startswith("__"):
                    if word not in word_to_vector:
                        if augmentation is True:
                            for item in model.get_nearest_neighbors(word):
                                score, similar_word = item
                                if similar_word not in word_to_vector:
                                    word_to_vector[similar_word] = model.get_word_vector(similar_word)
                        word_to_vector[word] = model.get_word_vector(word)
    return word_to_vector


def save_vocab(word_to_vector, output_vocab_file):
    print "Save vocab of {} words...".format(len(word_to_vector))
    with open(output_vocab_file, 'w') as vocab_file:
        for word, vector in word_to_vector.items():
            vector_str = str(vector).replace("[", "")
            vector_str = vector_str.replace("]", "")
            vector_str = "-".join(vector_str.split()).replace("-", " ")
            vocab_file.write("{} {}\n\r".format(word, vector_str))
    print("Done !")


def main(pretrained_embeddings_file_path, data_file_path="train.txt", output_vocab_file="vocab.txt"):
    print("Start building vocab...")
    model = load_pretrained_model(pretrained_embeddings_file_path)
    word_to_vector = load_vocab(model, data_file_path)
    save_vocab(word_to_vector, output_vocab_file)
    print ("Bye bye !")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a collection a static word vector from custom data and pretrained embeddings')
    parser.add_argument("--pretrained", type=str, default="./models/word_embeddings/wiki-news-300d-1M-subword.bin", help='The pretrained embeddings')
    parser.add_argument("--data", type=str, default="train.txt", help='The data file to use')
    parser.add_argument("--output", type=str, default="vocab.txt", help='The output file')

    args = parser.parse_args()
    print ("Start generating dataset...")
    main(args.pretrained, data_file_path=args.data, output_vocab_file=args.output)
