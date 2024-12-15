# Multilingual_KR

Changing tokenizer directly is not easy. Traditional, the model is trained on specific tokenizer. Thus, to change tokenizer, we may need to train the model from beginning. It may take a large amount of time and also even though some model like m-Bert, XLM is not trained on sentencepiece tokenizer, it has been realized in XLM-R, and has been test in the article (Jiang et al., 2020). Fortunately, recent papers (Gee et al., 2022) (Minixhofer et al., 2024) has shown way to change tokenizer in zero shot condition. Thus, we want to employ those method and then used on multi lingual knowledge of fact Q&A.

We reimplemented X-FACTR to establish baseline accuracy for the knowledge retrieval task. Subsequently, we applied recent tokenizer transfer methods [FVT, FOCUS, ZeTT, Vocab-free Tokenizer] from the literature to adapt them to the knowledge retrieval setting. 

While previous studies primarily focus on general tokenizer transfer, our work specifically targets the transfer of multilingual tokenizers in multilingual models to language-specific tokenizers. Using FVT and FOCUS, we achieved comparable results for English, Spanish, and Chinese after transferring to language-specific tokenizers. Further, continued pretraining significantly reduced the performance gap between the new and original tokenizers. Surprisingly, after continued training with just 1,000 examples, the accuracy for Chinese surpassed that of the original mBERT tokenizer paired with the mBERT model.

All the result accuracy can be found in `X-FACTR-Extent/*.log`

The detailed prediction results are in the specific folder named `evl_mbert_*`

---
Requirments setup:
```
cd X-FACTR-Extend
conda create -n xfactr -y python=3.7 && conda activate xfactr && ./setup.sh
```

### In the X-FACTR folder
Steps to get reimplemented X-FACTR benchmark result:
```
bash run_language.sh
```

### In the X-FACTR-Extend folder, run the following experiments
---
Steps to get naive substitution result:
```
bash run_language.sh
```
---
Steps to Run FVT evaluation in the Report
```
bash run_language_fvt.sh
```
---
Steps to Run Focus evaluation in the Report
```
bash run_language_focus.sh
```
---
Steps to Run FOCUS with training in the Report
```
./run_lms_focus.sh
```
Then run the evluation
```
bash run_language_focus_pretraining.sh
```
---
Steps to Run ZeTT in the Report
```
python scripts/probe_zztr.py --probe mlamaf --model xlmr_zztr --lang en --pred_dir prediction_folder
```
---
Steps to train neural tokenizer:
```
python X-FACTR-Extend/scripts/neural_tokenizer_trainer.py --epoch 10
```