# ccChatbot

A highly customizable chatbot which could be trained with custom corpus.



## Limitations

1. No matter which separator the corpus uses, it **must be a pair of statement and responses**

2. <u>*Currently only corpuses with .CONV format with E and M separator are officially supported.*</u> 

   You could write your own parser for other format of corpuses as long as the parser converts the processed file into **.tsv format without header**

   

### Technical Limitations

* [ ] Better text cleaning
* [ ] Better documentation
* [ ] To filter text by MAX_LENGTH instead of clamping
* [ ] To save a subclassed model with optimizer state instead of only weights



## Reference

#### GRU

- https://github.com/zhaoyingjun/chatbot



#### Transformer

- https://arxiv.org/abs/1706.03762
- https://www.tensorflow.org/text/tutorials/transformer
- https://github.com/jadore801120/attention-is-all-you-need-pytorch



#### Other resources

- **Deep Learning: CS 182 Spring 2021 - online lecture offered by UC Berkeley**

  https://youtube.com/playlist?list=PL_iWQOsE6TfVmKkQHucjPAoRtIJYt8a5A
