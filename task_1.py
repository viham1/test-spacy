import random
import spacy
from spacy.training import Example
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pos', type=str, help='Example Positional Argument') # will be accesible under args.POS

args = parser.parse_args()

print(args.pos)

LABEL = 'ANIMAL'
TRAIN_DATA = [
    ("Horses are too tall and they pretend to care about your feelings", {'entities': [(0, 6, LABEL)]}),
    ("Do they bite?", {'entities': []}),
    ("horses are too tall and they pretend to care about your feelings", {'entities': [(0, 6, LABEL)]}),
    ("horses pretend to care about your feelings", {'entities': [(0, 6, LABEL)]}),
    ("they pretend to care about your feelings, those horses", {'entities': [(48, 54, LABEL)]}),
    ("horses?", {'entities': [(0, 6, LABEL)]})
]
nlp = spacy.load('en_core_web_sm')  # load existing spaCy model
ner = nlp.get_pipe('ner')
ner.add_label(LABEL)
print(ner.move_names) # Here I see, that the new label was added
optimizer = nlp.create_optimizer()
# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):  # only train NER
    for itn in range(20):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
        print(losses)
# test the trained model # add some dummy sentences with many NERs

test_text = 'Do you like horses?'
doc = nlp(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent.label_, " -- ", ent.text)