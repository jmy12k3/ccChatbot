# ccChatbot

A highly customizable chatbot which could be trained with custom corpus.



## Limitations

1. No matter which separator the corpus uses, it **must be a pair of statement and responses**

2. <u>Currently only corpuses with .CONV format with E and M separator are officially supported.</u> 

   You could write your own parser for other format of corpuses as long as the parser converts the processed file into .tsv format without header.

   

### Technical Limitations:

* [ ] Filter text by MAX_LENGTH instead of clamp
* [ ] Save model with optimizer state instead of only weights



## Reference:

- https://github.com/zhaoyingjun/chatbot
