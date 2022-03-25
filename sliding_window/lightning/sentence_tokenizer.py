import re
import nltk
import re
import torch

nltk.download('punkt', quiet=True)

class SentenceProcessor():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.CLS_TOKEN = self.tokenizer.cls_token_id
        self.PAD_TOKEN = self.tokenizer.pad_token_id
        self.SEP_TOKEN = self.tokenizer.sep_token_id
        
        self.num_prev_sentences = 2
        self.max_block_size = 510
        
    def preprocess(self, text):
        # clean the text by removing unnecessary characters
        text = re.sub(r'\s+', ' ', text)
        
        # split text into sentences
        sentences = nltk.tokenize.sent_tokenize(text)
        
        # convert sentences into tokens
        sentences_ids = [self.tokenizer.encode_plus(
                                    sentence, 
                                    add_special_tokens=False, 
                                    padding=False, 
                                    truncation=False,
                                    return_token_type_ids=False,
                                    return_attention_mask=False,
                                    return_tensors='pt',
                                    verbose=False)['input_ids'].tolist()[0] for sentence in sentences]
        
        ### split tokens into blocks of 512 tokens ###
        
        input_ids_blocks = []
        attention_blocks = []
        
        def add_to_input_ids_blocks(input_ids):
            assert len(input_ids) <= self.max_block_size
            
            pad_len = self.max_block_size - len(input_ids)
            
            input_ids.insert(0, self.CLS_TOKEN)
            input_ids.append(self.SEP_TOKEN)
            
            attention_mask = [1] * len(input_ids)
            
            # add padding length to make all the blocks the same size
            if pad_len > 0:
                input_ids.extend([self.PAD_TOKEN] * pad_len)
                attention_mask.extend([0] * pad_len)
            
            input_ids_blocks.append(input_ids)
            attention_blocks.append(attention_mask)
        
        
        current_input_ids = []
        for i, sentence_ids in enumerate(sentences_ids):
            if (len(current_input_ids) + len(sentence_ids) <= self.max_block_size):
                # if current block has enough space for the current sentence
                current_input_ids.extend(sentence_ids)
            else:
                # if the current block doesn't have enough space for the current sentence
                
                # clear the current block
                if current_input_ids:
                    add_to_input_ids_blocks(current_input_ids)
                    current_input_ids = []
                
                # if the sentence is too long to be less than max token size, trucate it
                if not (len(sentence_ids) <= self.max_block_size):
                    current_input_ids = sentence_ids[:self.max_block_size]
                    add_to_input_ids_blocks(current_input_ids)
                    current_input_ids = []
                    continue
                
                current_input_ids.extend(sentence_ids)
                
                # add the previous sentences to the current block if it is less than 512 tokens
                for j in range(min(self.num_prev_sentences, i)):
                    prev_sentence = sentences_ids[i-j-1]
                    if len(current_input_ids) + len(prev_sentence) <= self.max_block_size:
                        current_input_ids[:0] = prev_sentence
                    else:
                        # retain some of the first previous sentence for context learning
                        if j == 0:
                            diff = self.max_block_size - len(current_input_ids)
                            current_input_ids[:0] = prev_sentence[len(prev_sentence) - diff:len(prev_sentence)]
                            add_to_input_ids_blocks(current_input_ids)
                            current_input_ids = []
                        break
            
            if i == len(sentences_ids) - 1 and current_input_ids:
                add_to_input_ids_blocks(current_input_ids)
                current_input_ids = []
        
        return {
            'input_ids': torch.IntTensor(input_ids_blocks),
            'attention_mask': torch.IntTensor(attention_blocks)
        }