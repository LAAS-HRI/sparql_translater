#!/usr/bin/env python

import argparse
import csv
import yaml
import numpy as np
from numpy.random import shuffle

np.random.seed(123)  # for reproducibility


class Tokenizer(object):
    def __init__(self, sparql_tokens_file="sparql_tokens.yaml"):
        self.tokens_map = {}
        self.reverse_token_map = {}
        with open(sparql_tokens_file, "r") as file:
            self.tokens_map = yaml.load(file)
        for sparql, token in self.tokens_map.items():
            self.reverse_token_map[token] = sparql

    def tokenize(self, sentence):
        tokenized_sentence = sentence
        for word in sentence.split():
            if word in self.tokens_map:
                tokenized_sentence = tokenized_sentence.replace(word, self.tokens_map[word])
        return tokenized_sentence

    def reconstruct(self, sentence):
        reconstructed_sentence = ""
        for word in sentence.split():
            if word in self.reverse_token_map:
                reconstructed_sentence += self.reverse_token_map[word]
            else:
                reconstructed_sentence += word
        return reconstructed_sentence


def load_templates(templates_file_path):
    templates = []
    with open(templates_file_path, "r") as file:
        for line in file:
            line_token = line.split("|")
            templates.append((line_token[0], line_token[1]))
    return templates


def load_individuals(individuals_file_path):
    individuals_map = {}
    with open(individuals_file_path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        first_line = next(reader)
        for individuals_type in first_line:
            individuals_map["<"+individuals_type+">"] = []
    with open(individuals_file_path, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            for type in first_line:
                if type in row:
                    if row[type] != "":
                        individuals_map["<"+type+">"].append(row[type])
    return individuals_map


def generate_data(templates, individuals, nb_examples_per_template=1000, tokenizer=None):
    data = []
    source_already_generated = {}
    target_already_generated = {}
    for template_pair in templates:
        source_template, target_template = template_pair
        for j in range(0, nb_examples_per_template):
            source_template_filled = fill_template(source_template, individuals)
            target_template_filled = fill_template(target_template, individuals)
            if source_template_filled not in source_already_generated and target_template_filled not in target_already_generated:
                if tokenizer is None:
                    data.append((source_template_filled, target_template_filled))
                else:
                    data.append((source_template_filled, tokenizer.tokenize(target_template_filled)))
                source_already_generated[source_template_filled] = True
                target_already_generated[target_template_filled] = True
    shuffle(data)
    return data


def fill_template(template, individuals):
    for key in individuals.keys():
        index = pick_index(individuals[key])
        individual = individuals[key][index]
        template = template.replace(key, individual)
        template = template.replace("[none] ", "")
        template = template.replace(" [none] ", " ")
    return template


def pick_index(sequence):
    return int((np.random.random_sample() * len(sequence)) % len(sequence))


def save(data, training_file_path, validation_file_path):
    nb_train = 0
    nb_val = 0
    split_index = int(len(data)*0.8)
    train_data = data[:split_index]
    val_data = data[split_index:]
    with open(training_file_path+".en", 'w') as train_file_source:
        with open(training_file_path+".sparql", 'w') as train_file_target:
            for data_pair in train_data:
                source, target = data_pair
                train_file_source.write(source+"\n\r")
                train_file_target.write(target)
                nb_train += 1

    with open(validation_file_path+".en", 'w') as val_file_source:
        with open(validation_file_path+".sparql", 'w') as val_file_target:
            for data_pair in val_data:
                source, target = data_pair
                val_file_source.write(source+"\n\r")
                val_file_target.write(target)
                nb_val += 1

    print ("Saved "+str(nb_train)+" samples in training dataset")
    print ("Saved "+str(nb_val)+" samples in validation dataset")


def main(templates_file_path="templates.csv", individuals_file_path="individuals.csv", max_examples_per_template=600, train_file="seq2seq_train", val_file="seq2seq_val"):
    tokenizer = Tokenizer()
    templates = load_templates(templates_file_path)
    individuals = load_individuals(individuals_file_path)
    data = generate_data(templates, individuals, max_examples_per_template, tokenizer)
    save(data, train_file, val_file)
    print ("Bye bye !")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The dataset generator.')
    parser.add_argument("--templates", type=str, default="templates_seq2seq.txt", help='The templates file to use')
    parser.add_argument("--individuals", type=str, default="individuals.csv", help='The individuals to randomly pick')
    parser.add_argument("--max_per_template", type=int, default=50, help="The max number of examples to generate per template")

    args = parser.parse_args()
    print ("Start generating dataset...")
    main(templates_file_path=args.templates, individuals_file_path=args.individuals, max_examples_per_template=args.max_per_template)
