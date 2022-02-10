import numpy as np
import torch

from lib.pooling import transform_text_to_model_input


class Preprocessor():
    '''
    An abstract class for text preprocesssors. Preprocessor takes an array of strings and transforms it to an array of data compatible with the model
    '''

    def __init__(self):
        pass

    def preprocess(self, array_of_texts):
        raise NotImplementedError(
            "Preprocessing is implemented for subclasses only")


class BERTTokenizerPooled(Preprocessor):
    def __init__(self, tokenizer, size, step, minimal_length):
        self.tokenizer = tokenizer
        self.text_splits_params = [size, step, minimal_length]

    def preprocess(self, array_of_texts):
        array_of_preprocessed_data = tokenize_pooled(
            array_of_texts, self.tokenizer, *self.text_splits_params)
        return array_of_preprocessed_data


def tokenize_pooled(texts, tokenizer, size, step, minimal_length):
    '''
    Tokenizes texts and splits to chunks of 512 tokens

    Parameters:
    texts - list of strings
    tokenizer - object of class transformers.PreTrainedTokenizerFast

    size - size of text chunk to tokenize (must be <= 510)
    step - stride of pooling
    minimal_length - minimal length of a text chunk

    Returns:
    array_of_preprocessed_data - array of the length len(texts)
    '''
    model_inputs = [
        transform_text_to_model_input(
            text,
            tokenizer,
            size,
            step,
            minimal_length) for text in texts]
    input_ids = [model_input[0] for model_input in model_inputs]
    attention_mask = [model_input[1] for model_input in model_inputs]
    tokens = {'input_ids': input_ids, 'attention_mask': attention_mask}
    return tokens
