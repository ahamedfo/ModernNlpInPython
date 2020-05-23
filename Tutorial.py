import os
import codecs
import json
import pandas as pd
import itertools as it
import spacy
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

nlp = spacy.load('en')

businesses_filepath = "/Users/ahamedfofana/PycharmProjects/Modern NLP IN Python/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json"
review_txt_filepath = "/Users/ahamedfofana/PycharmProjects/Modern NLP IN Python/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json"
intermediate_directory = '/Users/ahamedfofana/PycharmProjects/untitled15/Intermediate'

def sample_simulator(file):
    with open(review_txt_filepath, encoding='utf_8') as f:
        sample_review = list(it.islice(f, 8, 9))[0]

        sample_review = sample_review.replace('\\n', '\n')
        print(sample_review)

def number_restaurants(file):
    ############################
    resturant_ids = set()

    with open(businesses_filepath, encoding='utf_8') as f:
        for business_json in f:
            business = json.loads(business_json)
            if not business[u'categories'] or u'Restaurants' not in business[u'categories']:
                continue

            resturant_ids.add(business[u'business_id'])

    resturant_ids = frozenset(resturant_ids)

    print ('{:,}'.format(len(resturant_ids)), u'restaurants in the dataset.')


def resturantReviews(file):
    if 0 == 1:
        return 1
    else:
        with open(review_txt_filepath, encoding='utf_8') as review_txt_file:
            for review_count, line in enumerate(review_txt_file):
                pass
        print( u'Text from {:,} restaurants reviews in the txt file.'.format(review_count + 1))

    # with open('/Users/ahamedfofana/PycharmProjects/untitled15/writing_file', "r",encoding='utf_8') as f:
    #     data = f.read()
    #     data = data.replace('\\n', '\n')
    #
    # parsed_review = nlp(data)
    #
    #
    #
    #
    # token_text = [token.orth_ for token in parsed_review]
    # token_pos = [token.pos_ for token in parsed_review]
    #
    # token_lemma = [token.lemma_ for token in parsed_review]
    # token_shape = [token.shape_ for token in parsed_review]
    #
    # token_entity_type = [token.ent_type_ for token in parsed_review]
    # token_entity_job = [token.ent_iob_ for token in parsed_review]
    #
    # print(pd.DataFrame(zip(token_text, token_entity_type, token_entity_job), columns=['token_text', 'entity_type', 'inside_outside_begin']))
    #
    # token_attributes = [(token.orth_,
    #                      token.prob,
    #                      token.is_stop,
    #                      token.is_punct,
    #                      token.is_space,
    #                      token.like_num,
    #                      token.is_oov)
    #                     for token in parsed_review]
    #
    # df = pd.DataFrame(token_attributes,
    #                   columns=['text',
    #                            'log_probability',
    #                            'stop?',
    #                            'punctuation?',
    #                            'whitespace?',
    #                            'number?',
    #                            'out of vocab.?'])
    #
    # df.loc[:, 'stop?':'out of vocab.?'] = (df.loc[:, 'stop?':'out of vocab.?']
    #                                        .applymap(lambda x: u'Yes' if x else u''))

def punct_space(token):
    return token.is_punct or token.is_space

def line_review(filename):

    with open(filename, encoding='utf_8') as f:
        for review in f:
            yield review.replace('\\n', '\n')

def lemmatized_sentence_corpus(filename):

    for parsed_review in nlp.pipe(line_review(filename), batch_size=10000, n_threads=4):

        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for token in sent if not punct_space(token)])


unigram_sentences_filepath = intermediate_directory + 'unigram_sentences_all.txt'


with open(unigram_sentences_filepath, 'w', encoding='utf_8') as f:
    for sentence in lemmatized_sentence_corpus(("/Users/ahamedfofana/PycharmProjects/untitled15/writing_file")):
        f.write(sentence + '\n')

unigram_sentences = LineSentence(unigram_sentences_filepath)

for sentence in it.islice(unigram_sentences, 1, 6):
        print(u' '.join(sentence))
        print(u' ')

bigram_model_filepath = intermediate_directory + 'bigram_model_all'

def link_twoWords(file):
    bigram_model = Phrases(unigram_sentences)
    bigram_model.save(bigram_model_filepath)
    bigram_model = Phrases.load(bigram_model_filepath)




if __name__ == '__main__':





