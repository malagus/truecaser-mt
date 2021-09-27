from flair.data import Sentence
from flair.models import SequenceTagger

from data import TruecaserSentence
from dataset.processor import character_tokenizer
from model.rnn.char import CharRNNTrueceser

tagger: CharRNNTrueceser = CharRNNTrueceser.load("result/truecase/2020-02-17 20:19:42.361040/final-model.pt")
sentence = Sentence('i love warsaw', use_tokenizer=character_tokenizer)
# sentence = TruecaserSentence(sentence)
r = tagger.predict(sentence)
print(sentence.to_dict(tag_type='case'))
