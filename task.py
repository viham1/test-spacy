import spacy

spacy.cli.train.train("config.cfg", output_path="/model",overrides={"paths.train": "train.spacy", "paths.dev": "test.spacy"},use_gpu=0)
