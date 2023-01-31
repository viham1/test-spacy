import spacy

spacy.cli.train.train("config.cfg", output_path="./outputs/model",overrides={"paths.train": "train.spacy", "paths.dev": "test.spacy"},use_gpu=0)
