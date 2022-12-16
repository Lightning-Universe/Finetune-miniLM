Fork of https://github.com/Lightning-AI/LAI-Text-Classification-Component

Task: https://lightningai.atlassian.net/browse/ENG-2261

Sources:
- Embeddings forward: https://github.com/UKPLab/sentence-transformers/blob/v2.2.2/sentence_transformers/models/Transformer.py#L60-L79
- Sentence transformers example: https://www.sbert.net/docs/training/overview.html
- HF model outputs: https://huggingface.co/docs/transformers/main_classes/output
- Yelp review dataset: https://huggingface.co/datasets/yelp_review_full
- Model details: https://huggingface.co/microsoft/MiniLM-L12-H384-uncased
- Model details: https://github.com/microsoft/unilm/tree/master/minilm#english-pre-trained-models

## Run

### (Optional) Set up a clean environment

```bash
virtualenv venv
source venv/bin/activate
pip install lightning
```

### To share the code

Option 1:

1. Share the [app.py](/app.py) code
2. See instructions for running below

Option 2:

1. `git clone https://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@github.com/Lightning-AI/Finetune-miniLM`
2. See instructions for running below

### Running on the cloud

```bash
lightning run app app.py --setup --cloud
```

Don't want to use the public cloud? Contact us at `product@lightning.ai` for early access to run on your private cluster (BYOC)!


### Running locally

```bash
lightning run app app.py --setup
```


## Development notes

There's a gallery-ready copy of the entrypoint in https://github.com/Lightning-AI/Finetune-miniLM-gallery. Changes to the [app.py](/app.py) file should be applied there aswell.
