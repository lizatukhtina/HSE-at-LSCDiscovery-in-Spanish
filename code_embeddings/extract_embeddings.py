import os
import pickle
import torch
import torch.nn.functional as padder
import tqdm
from abc import ABC, abstractmethod
from argument_parser import (Argument, CustomArgumentParser, BooleanArgument, OptionalArgument,
                             SemicolonSeparatedArgument)
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from typing import List, Tuple


class Checkpoint:
    def __init__(self, data_path: str, ckpt_path: str = '', corpora_processed: list = None, last_line: int = 0,
                 data: dict = None, frequency: int = 0):
        self.data_path = data_path
        self.ckpt_path = ckpt_path
        self.corpora_processed = corpora_processed or ['<None>']
        self.last_line = last_line
        self.data = data
        self.frequency = frequency or 0

    @classmethod
    def init_from_file(cls, output_path: str, frequency: int = 0):
        try:
            with open(f'{output_path}.ckpt', 'rb') as f:
                checkpoint = pickle.load(f)
                return cls(data_path=output_path, ckpt_path=checkpoint.ckpt_path,
                           corpora_processed=checkpoint.corpora_processed, last_line=checkpoint.last_line,
                           data=checkpoint.data, frequency=frequency)
        except FileNotFoundError:
            return cls(data_path=output_path, ckpt_path=f'{output_path}.ckpt', frequency=frequency)


class BertBatch:
    def __init__(self, tensor: torch.Tensor, texts: List[list]):
        self.tensor = tensor
        self.texts = texts

    @classmethod
    def init_from_text_batch(cls, text_batch: List[str], tokenizer, use_gpu=False):
        """Add special tokens to sequences from the corpus and pad them to have the same length."""
        batch_tensors = []
        texts = []
        max_len = 0
        for batch_line in text_batch:
            text = ['[CLS]'] + tokenizer.tokenize(batch_line) + ['[SEP]']
            max_len = max(max_len, len(text))
            indexed_tokens = tokenizer.convert_tokens_to_ids(text)
            if use_gpu:
                batch_tensors.append(torch.tensor([indexed_tokens]).cuda())
            else:
                batch_tensors.append(torch.tensor([indexed_tokens]))
            texts.append(text)
        batch_tensors = [padder.pad(i, (0, max_len - len(i[0]))) for i in batch_tensors]  # zero padding
        return cls(torch.cat(batch_tensors), texts)


class BaseExtractor(ABC):
    def __init__(self, layers: str, model_path: str, model_name: str, targets_path: str, corpora_paths: list,
                 batch_size: int, checkpoint: Checkpoint, concat_layers=False, use_gpu=False):
        print('Initializing extractor')
        if not layers:
            self.layers = [i for i in range(0, 12)]
        else:
            self._parse_layers()
        self.concat_layers = concat_layers
        self.use_gpu = use_gpu
        self.model_path = model_path
        self.model_name = model_name
        self.targets_path = targets_path
        self.corpora_paths = corpora_paths
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.target_to_spellings = self.generate_target_spellings()
        self.spelling_to_target = {spelling: target for target in self.target_to_spellings for spelling in
                                   self.target_to_spellings[target]}
        try:
            fine_tuned_model = torch.load(self.model_path)
        except RuntimeError:
            fine_tuned_model = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.model = self.load_model(fine_tuned_model)
        self.tokenizer = self.load_tokenizer(fine_tuned_model)
        self.tokenizer.save_pretrained('/home/aishein_1/.cache/huggingface/transformers')
        if self.use_gpu:
            self.model.cuda()
            self.tokenizer.cuda()
        self.model.eval()
        os.makedirs(os.path.dirname(self.checkpoint.ckpt_path), exist_ok=True)
        if not self.checkpoint.data:
            self.checkpoint.data = {i: {} for i in self.corpora_paths}  # embeddings for all datasets here.
            # Filled in get_bert_embeddings
        print('Extractor initialized')

    def _parse_layers(self) -> None:
        self.layers = []
        spl = [i.strip() for i in args.bert_layers.split(',')]
        for part in spl:
            subpart = [i.strip() for i in part.split('-')]
            try:
                self.layers.extend([i for i in range(int(subpart[0]), int(subpart[-1]) + 1)])
            except ValueError:
                print('--bert_layers argument could not be parsed. Please refer to --help')
                exit(1)
        if not self.layers or any(i < 0 or i > 11 for i in self.layers):
            print('--bert_layers argument could not be parsed. Please refer to --help')
            exit(1)

    @abstractmethod
    def load_model(self, state):
        pass

    @abstractmethod
    def load_tokenizer(self, state):
        pass

    @staticmethod
    @abstractmethod
    def check_for_split_tokens(tokens: List[str]) -> List[Tuple[str, List[int]]]:
        pass

    def generate_target_spellings(self) -> dict:
        translator = str.maketrans("äöü", "aou")
        with open(self.targets_path, encoding='utf-8-sig') as f:
            return {word.strip(): {word.strip(),
                                   word[:word.find('_')].strip(),
                                   word.translate(translator).strip(),
                                   word.lower().strip(),
                                   word.title().strip()}
                    for word in f.readlines()}

    def get_bert_embeddings_for_batch(self, bert_batch: BertBatch, corpus_name: str):
        """Calculate embeddings for target words and write them to total_embeddings"""
        if self.use_gpu:
            segments_tensors = torch.ones(len(bert_batch.tensor), len(bert_batch.tensor[0]), dtype=torch.long).cuda()
        else:
            segments_tensors = torch.ones(len(bert_batch.tensor), len(bert_batch.tensor[0]), dtype=torch.long)
        with torch.no_grad():
            try:
                outputs = self.model(bert_batch.tensor, segments_tensors)
            except (RuntimeError, IndexError) as e:
                print(e)
                return
            hidden_states = outputs.hidden_states[1:]  # the first state is the input state, we don't need it
        token_embeddings = torch.stack(hidden_states, dim=0)
        # initial order: layer; batch number; token (word); embeddings dimension
        token_embeddings = token_embeddings.permute(1, 2, 0, 3)  # permuting dimensions for easy iterating
        # after permuting: batch number; token (word); layer; embeddings dimension
        for i_batch, batch in enumerate(token_embeddings):  # iterating over batches
            tokens_with_indices = self.check_for_split_tokens(bert_batch.texts[i_batch])
            for token_indices_tuple in tokens_with_indices:
                word = token_indices_tuple[0]
                indices = token_indices_tuple[1]
                if word in self.spelling_to_target:
                    average_embeddings = sum(token_embeddings[i_batch][indices]) / len(indices)
                    # If the word is split and has embeddings for each of its part, we calculate the average of its part
                    # embeddings for each layer separately
                    if self.concat_layers:
                        sum_vec = torch.concat(tuple(average_embeddings[self.layers]), dim=0)
                    else:
                        sum_vec = torch.sum(average_embeddings[self.layers], dim=0)
                    if self.spelling_to_target[word] not in self.checkpoint.data[corpus_name]:
                        self.checkpoint.data[corpus_name][self.spelling_to_target[word]] = sum_vec  # initial embedding
                    else:  # we just stack the new embedding at the bottom of the previous ones
                        self.checkpoint.data[corpus_name][self.spelling_to_target[word]] = torch.vstack(
                            [self.checkpoint.data[corpus_name][self.spelling_to_target[word]], sum_vec])

    def extract(self):
        print(f"The following BERT layers will be extracted: {self.layers}. ", end="")
        if self.concat_layers:
            print("The layers will be concatenated.")
        else:
            print("The layers will be summed.")
        for corpus in self.corpora_paths:
            if corpus in self.checkpoint.corpora_processed[:-1]:
                print(f'Skipping file: {corpus}\n')
                continue
            print(f'Processing file: {corpus}\n')
            current_batch = []
            batch_counter = 0
            with tqdm.tqdm(total=os.path.getsize(corpus)) as progress_bar:
                with open(corpus, encoding='utf-8-sig') as f:
                    file_iter = enumerate(f)
                    if self.checkpoint.corpora_processed[-1] == corpus:
                        while True:
                            line_num, line = next(file_iter)
                            progress_bar.update(len(line))
                            if line_num >= self.checkpoint.last_line:
                                print(f'Skipped first {line_num} lines')
                                break
                    since_checkpoint = 0
                    for line_num, line in file_iter:
                        since_checkpoint += 1
                        progress_bar.update(len(line))
                        for target in self.spelling_to_target:
                            if target in line.split():
                                current_batch.append(line.strip())
                                batch_counter += 1
                                if not batch_counter % self.batch_size:
                                    bert_batch = BertBatch.init_from_text_batch(text_batch=current_batch,
                                                                                tokenizer=self.tokenizer,
                                                                                use_gpu=self.use_gpu)
                                    self.get_bert_embeddings_for_batch(bert_batch=bert_batch, corpus_name=corpus)
                                    current_batch = []
                                    batch_counter = 0
                                    if since_checkpoint > self.checkpoint.frequency != 0:
                                        with open(self.checkpoint.ckpt_path, 'wb') as f_ckpt:
                                            if corpus not in self.checkpoint.corpora_processed:
                                                self.checkpoint.corpora_processed.append(corpus)
                                            self.checkpoint.last_line = line_num
                                            since_checkpoint = 0
                                            pickle.dump(self.checkpoint, f_ckpt)
                                break
        with open(self.checkpoint.data_path, 'wb') as f_data:
            pickle.dump(self.checkpoint.data, f_data)


class RobertaExtractor(BaseExtractor):
    def __init__(self, layers: str, model_path: str, model_name: str, targets_path: str, corpora_paths: list,
                 batch_size: int, checkpoint: Checkpoint, concat_layers=False, use_gpu=False):
        super().__init__(layers=layers, concat_layers=concat_layers, model_path=model_path, model_name=model_name,
                         batch_size=batch_size, use_gpu=use_gpu, targets_path=targets_path, corpora_paths=corpora_paths,
                         checkpoint=checkpoint)

    def load_model(self, state):
        return RobertaModel.from_pretrained(self.model_name, output_hidden_states=True, state_dict=state,
                                            local_files_only=True)

    def load_tokenizer(self, state):
        return RobertaTokenizer.from_pretrained(self.model_name, local_files_only=True)

    @staticmethod
    def check_for_split_tokens(tokens: List[str]) -> List[Tuple[str, List[int]]]:
        """Here we analyze a tokenized line and scan it for split words (for example, [Ġcont, ineo]).
            We glue such words together and memorize their indices in the original tokenized line.
            The result is a list of tuples of two elements. The first element is a word,
            the second element is a list of this word's positions int the batch"""
        tokens_with_indexes = []
        current_index = len(tokens) - 1
        # We move in reverse direction, from the last token to the first one
        while current_index >= 0:
            current_token = tokens[current_index]
            if (current_token.startswith('Ġ') or current_token in ['[SEP]', '[CLS]'] or
                    tokens[current_index - 1] == '[CLS]'):
                # Then it's not a split word, we just add it to the result with its index
                tokens_with_indexes.append((current_token.strip('Ġ'), [current_index]))
                current_index -= 1
            else:
                # It's a split word. We are now standing at the tail of this word
                token_indexes = [current_index]  # First we add the tail index of this word to the list
                full_token = current_token  # Then we initialize the full token with its tail
                sub_index = current_index - 1  # Move one step towards the beginning of the line
                while not tokens[sub_index].startswith('Ġ'):
                    # The word can be split into multiple parts.
                    # This loop runs only if the word is split in more than two parts
                    # This loop processes all middle parts of the word (non-head and non-tail)
                    full_token = tokens[sub_index] + full_token  # glue the part to the front of the word
                    token_indexes.append(sub_index)
                    sub_index -= 1
                # Finally, we process the head of the split word. It has to start with 'Ġ' token
                full_token = tokens[sub_index].strip('Ġ') + full_token
                token_indexes.append(sub_index)
                token_indexes.reverse()  # Because we moved from the tail towards the head (done mostly for convenience)
                tokens_with_indexes.append((full_token, token_indexes))
                current_index = sub_index - 1
        tokens_with_indexes.reverse()  # Also for convenience
        return tokens_with_indexes


class BertExtractor(BaseExtractor):
    def __init__(self, layers: str, model_path: str, model_name: str, targets_path: str, corpora_paths: list,
                 batch_size: int, checkpoint: Checkpoint, concat_layers=False, use_gpu=False):
        super().__init__(layers=layers, concat_layers=concat_layers, model_path=model_path, model_name=model_name,
                         use_gpu=use_gpu, targets_path=targets_path, corpora_paths=corpora_paths, batch_size=batch_size,
                         checkpoint=checkpoint)

    def load_model(self, state):
        return BertModel.from_pretrained(self.model_name, output_hidden_states=True, state_dict=state,
                                         local_files_only=True)

    def load_tokenizer(self, state):
        return BertTokenizer.from_pretrained('/home/aishein_1/.cache/huggingface/transformers', local_files_only=True)

    @staticmethod
    def check_for_split_tokens(tokens: List[str]) -> List[Tuple[str, List[int]]]:
        """Here we analyze a tokenized line and scan it for split words (for example, [ab, ##bauen]).
        We glue such words together and memorize their indices in the original tokenized line."""
        tokens_with_indexes = []
        current_index = len(tokens) - 1
        # We move in reverse direction, from the last token to the first one
        while current_index >= 0:
            current_token = tokens[current_index]
            if not current_token.startswith('##'):
                # Then it's not a split word, we just add it to the result with its index
                tokens_with_indexes.append((current_token, [current_index]))
                current_index -= 1
            else:
                # It's a split word. We are now standing at the tail of this word
                token_indexes = [current_index]  # First we add the tail index of this word to the list
                full_token = current_token.strip(
                    '##')  # Then we strip the word part of “##” and add this part to a string
                sub_index = current_index - 1  # Move one step towards the beginning of the line
                while tokens[sub_index].startswith('##'):
                    # The word can be split into multiple parts.
                    # This loop runs only if the word is split in more than two parts
                    # This loop processes all middle parts of the word (non-head and non-tail)
                    full_token = tokens[sub_index].strip('##') + full_token  # glue the part to the front of the word
                    token_indexes.append(sub_index)
                    sub_index -= 1
                # Finally we process the head of the split word. It doesn't have "##"
                full_token = tokens[sub_index] + full_token
                token_indexes.append(sub_index)
                token_indexes.reverse()  # Because we moved from the tail towards the head (done mostly for convenience)
                tokens_with_indexes.append((full_token, token_indexes))
                current_index = sub_index - 1
        tokens_with_indexes.reverse()  # Also for convenience
        return tokens_with_indexes
        # So this list contains tuples of full words and indices of their parts in BERT-tokenized line


arg_list = [Argument(name='model_path', type_=str, help_message='Path to a .bin PyTorch model',
                     absent_message='Please specify a path to PyTorch model'),
            Argument(name='model_name', type_=str, help_message='Name of a model used',
                     absent_message='Please specify model name'),
            Argument(name='targets_path', type_=str, help_message='Path to a .txt file with target words',
                     absent_message='Please specify a path to target words'),
            SemicolonSeparatedArgument(name='corpora_paths', type_=str,
                                       help_message='Paths to corpora separated with ;',
                                       absent_message='Please specify a path to corpora separated with ;'),
            OptionalArgument(name='corpora_language', type_=str,
                             help_message='not needed, kept for backward compatibility'),
            Argument(name='output_path', type_=str, help_message='Path to a result file with embeddings',
                     absent_message='Please specify the output path'),
            OptionalArgument(name='bert_layers', type_=str,
                             help_message='Bert layers to extract (from 0 to 11). Possible ways to provide:'
                                          ' 1) separated with commas (0,4,5);'
                                          ' 2) separated with dash (3-5 same as 3,4,5).'
                                          ' If this argument is omitted, all 12 layers will be extracted'),
            BooleanArgument(name='concat_layers', action='store_true',
                            help_message='Pass this argument if you want to concatenate the layers'
                                         ' rather than sum them. Disabled by default'),
            BooleanArgument(name='gpu', action='store_false', help_message='Pass this argument to use GPU'),
            OptionalArgument(name='checkpoint_frequency', type_=int,
                             help_message='Make checkpoints every <VALUE> lines. '
                                          'If omitted, checkpoints are not created')]
arg_parser = CustomArgumentParser()
args = arg_parser.parse(arg_list)
if 'roberta' in args.model_name:
    extractor = RobertaExtractor(layers=args.bert_layers, concat_layers=args.concat_layers, model_path=args.model_path,
                                 model_name=args.model_name, batch_size=32, targets_path=args.targets_path,
                                 corpora_paths=args.corpora_paths.split(';'),
                                 checkpoint=Checkpoint.init_from_file(output_path=args.output_path,
                                                                      frequency=args.checkpoint_frequency))
else:
    extractor = BertExtractor(layers=args.bert_layers, concat_layers=args.concat_layers, model_path=args.model_path,
                              model_name=args.model_name, batch_size=32, targets_path=args.targets_path,
                              corpora_paths=args.corpora_paths.split(';'),
                              checkpoint=Checkpoint.init_from_file(output_path=args.output_path,
                                                                   frequency=args.checkpoint_frequency))
extractor.extract()
