# ccChatbot

A highly customizable chatbot which could be trained with custom corpus.



## Limitations

1. No matter which separator the corpus uses, it **must be a pair of statement and responses**

2. <u>*preprocess.py depends on the format and the separator of the corpus*</u> 

   You could write your own parser for corpuses with other format into **.tsv format without header**

   

### Technical Limitations

* [x] Better text cleaning
* [ ] Better documentation
* [ ] To filter text by MAX_LENGTH instead of clamping
* [ ] To save a subclassed model with optimizer state instead of only weights



## Reference

#### GRU

- https://github.com/zhaoyingjun/chatbot



#### Transformer

- https://arxiv.org/abs/1706.03762
- https://www.tensorflow.org/text/tutorials/transformer
- https://tensorflow.google.cn/tutorials/text/transformer
- https://github.com/jadore801120/attention-is-all-you-need-pytorch



#### Corpuses

- https://github.com/codemayq/chinese_chatbot_corpus