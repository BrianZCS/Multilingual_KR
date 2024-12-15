# Multilingual_KR

Changing tokenizer directly is not easy. Traditional, the model is trained on specific tokenizer. Thus, to change tokenizer, we may need to train the model from beginning. It may take a large amount of time and also even though some model like m-Bert, XLM is not trained on sentencepiece tokenizer, it has been realized in XLM-R, and has been test in the article (Jiang et al., 2020). Fortunately, recent papers (Gee et al., 2022) (Minixhofer et al., 2024) has shown way to change tokenizer in zero shot condition. Thus, we want to employ those method and then used on multi lingual knowledge of fact Q&A.

---
Requirments setup:
```
cd X-FACTR-Extend
conda create -n xfactr -y python=3.7 && conda activate xfactr && ./setup.sh
```
---
Steps to get default implementation result:
```
bash run_language.sh
```
---
Steps to train neural tokenizer:
```
python X-FACTR-Extend/scripts/neural_tokenizer_trainer.py --epoch 10
```