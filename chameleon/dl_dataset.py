import torch
from types import NoneType
from typing import Type, Union, List
from itertools import repeat
from collections import defaultdict
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Vocabulary(object):
    # pre-defined token idx
    PAD, BOS, EOS, UNK = 0, 1, 2, 3

    def __init__(
        self,
        min_freq=1,
        max_vocab=99999,
    ):
        # Default Vocabulary
        self.itos = {Vocabulary.PAD: "<PAD>", Vocabulary.BOS: "<BOS>", Vocabulary.EOS: "<EOS>", Vocabulary.UNK: "<UNK>"}
        self.stoi = {token: idx for idx, token in self.itos.items()}

        self.min_freq = min_freq
        self.max_vocab = max_vocab

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text, delimiter):
        return [tok.strip() for tok in text.split(delimiter)]

    def build_vocab(self, sents, delimiter):
        # bag of words
        bow = defaultdict(int)

        for sent in sents:
            words = self.tokenizer(sent, delimiter=delimiter)
            for word in words:
                bow[word] += 1

        # limit vocab by removing low frequence word
        bow = {word: freq for word, freq in bow.items() if freq >= self.min_freq}
        bow = sorted(bow.items(), key=lambda x: -x[1])

        # limit size of the vocab
        bow = dict(bow[: self.max_vocab - len(self.itos)])

        # create vocab
        start_idx = len(self.itos)

        for word in bow.keys():
            self.stoi[word] = start_idx
            self.itos[start_idx] = word
            start_idx += 1

        print("Number of vocabularies: ", len(self))

    def encode(self, text, delimiter):
        """
        Encode text input. Support batch input.
        Return list.
        """

        encoded_text = []

        if isinstance(text, list):
            # |text| = [text1, text2, ...]
            tokenized_text = list(map(self.tokenizer, text, repeat(delimiter)))
            # |tokenized_text| = [[token1, token2, ...], [token1, token2, ...]]
            for tokens in tokenized_text:
                encoded_text += [
                    [self.stoi[token] if token in self.stoi.keys() else self.stoi["<UNK>"] for token in tokens]
                ]
                # |encoded_text| = [[token_idx1, token_idx2], [token_idx1, token_idx2]]
        else:
            # |text| = str
            tokenized_text = self.tokenizer(text, delimiter=delimiter)
            # |tokenized_text| = [token1, token2, ...]
            encoded_text += [
                self.stoi[token] if token in self.stoi.keys() else self.stoi["<UNK>"] for token in tokenized_text
            ]
            # |encoded_text| = [token_idx1, token_idx2, ...]

        return encoded_text

    def decode(self, indice, delimiter, removed_indice=[BOS, EOS, PAD]):
        """
        Decode indice input. Support batch input.
        Return list.
        """

        decoded_indice = []

        # check if indice is batch input
        if isinstance(indice, torch.Tensor):
            is_nested = indice.ndim > 1
            indice = indice.tolist()
        else:
            is_nested = any(isinstance(elm, list) for elm in indice)

        if is_nested:
            # |indice| = (batch_size, length)
            # |indice| = [[idx1, idx2, ...], [idx1, idx2, ...]]
            for encoded_text in indice:
                decoded = []
                for idx in encoded_text:
                    if idx in self.itos.keys() and idx not in removed_indice:
                        decoded += [self.itos[idx]]
                    elif idx in removed_indice:
                        continue
                    else:
                        decoded += [self.itos[Vocabulary.UNK]]

                decoded_indice += [delimiter.join(decoded).strip()]

        else:
            # |indice| = (length, )
            # |indice| = [idx1, idx2, ...]
            decoded = []
            for idx in indice:
                if idx in self.itos.keys() and idx not in removed_indice:
                    decoded += [self.itos[idx]]
                elif idx in removed_indice:
                    continue
                else:
                    decoded += [self.itos[Vocabulary.UNK]]

            decoded_indice += [delimiter.join(decoded).strip()]

        return decoded_indice


class TranslationDataset(Dataset):
    """
    Args:
        srcs (list): Sources to be used as the input data.
            Note. Sources must be tokenized before putting into Dataset.
        tgts (list): Targets to be used as the target data.
            Note. Targets must be tokenized before putting into Dataset.
        min_freq (int): Minimum frequency to be included in the vocabulary. Defaults to 1.
        max_vocab (int): Maximum size of vocabulary. Defaults to 99999.
        src_delimiter (str): Delimiter to tokenize the srcs and tgts.
        src_vocab (Vocabulary): Vocabulary to encode or decode the srcs of the validation_set and test_set.
            Defaults to None.
        tgt_vocab (Vocabulary): Vocabulary to encode or decode the tgts of the validation_set and test_set.
            Defaults to None.
        with_text (bool): Whether to include raw text in the output when calling __getitem__ method.
            It is used in evaluation and reinforcement learning. Defaults to False.
        is_dual (bool): Whether to make dataloader for dual learning.
    """

    def __init__(
        self,
        srcs: List[str],
        tgts: List[str],
        min_freq: int = 1,
        max_vocab: int = 99999,
        src_delimiter: str = " ",
        tgt_delimiter: str = " ",
        src_vocab: Union[Type[Vocabulary], NoneType] = None,
        tgt_vocab: Union[Type[Vocabulary], NoneType] = None,
        with_text: bool = False,
        is_dual: bool = False,
    ):
        # Originally, srcs and tgts both must have been tokenized using BPE before.
        # But in agri translation model, tgts were tokenized with custom tokenization.
        # Instead, tgts have to be delimited by tgt_delimiter before.
        self.srcs, self.tgts = srcs, tgts
        self.src_delimiter, self.tgt_delimiter = src_delimiter, tgt_delimiter

        # If with_text is True, not only the encoded_src and encoded_tgt,
        # the raw src and tgt text would be returned together when __getitem__ is called.
        self.with_text = with_text
        self.is_dual = is_dual

        # If the Dataset is train_dataset, it has to build its vocabulary.
        if src_vocab is None or tgt_vocab is None:
            # Initialize vocabulary of sources and targets
            self.src_vocab = Vocabulary(min_freq=min_freq, max_vocab=max_vocab)
            self.tgt_vocab = Vocabulary(min_freq=min_freq, max_vocab=max_vocab)
            # Build vocabulary of sources and targets
            self.src_vocab.build_vocab(self.srcs, delimiter=src_delimiter)
            self.tgt_vocab.build_vocab(self.tgts, delimiter=tgt_delimiter)
        else:
            # If the Dataset is validation or test_dateset, it has to use the vocabulary originated from train_dataset.
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, idx):
        src, tgt = self.srcs[idx], self.tgts[idx]

        # encode src
        # In dual learning, src must have BOS and EOS token at the beginning and the end.
        encoded_src = self.src_vocab.encode(src, delimiter=self.src_delimiter)
        if self.is_dual:
            encoded_src.insert(0, Vocabulary.BOS)
            encoded_src.append(Vocabulary.EOS)

        # In seq2seq structure, tgt must have BOS and EOS token at the beginning and the end.
        encoded_tgt = self.tgt_vocab.encode(tgt, delimiter=self.tgt_delimiter)
        encoded_tgt.insert(0, Vocabulary.BOS)
        encoded_tgt.append(Vocabulary.EOS)

        return_value = {"src": torch.tensor(encoded_src), "tgt": torch.tensor(encoded_tgt)}

        # src_txt and tgt_txt would be used in inference and evaluation
        if self.with_text:
            return_value["src_text"] = src
            return_value["tgt_text"] = tgt

        return return_value


class TranslationCollator:
    def __init__(self, pad_idx: int, eos_idx: int, max_length: int, with_text: bool = False, is_dual: bool = False):
        """
        Usages:
            It is used as a parameter in DataLoader.
            Collate batch srcs or tgts and process it to make batch loader.
            Add length of each src and tgt, and add pad token according to the length of batch.
        Args:
            pad_idx (int): Index of pad_token.
            eos_idx (int): Index of eos_token.
            max_length (list): Max length of the encoded_srcs or encoded_tgts .
            with_text (bool): Whether to include raw text in the output.
            is_dual (bool): Whether it is dual learning or not.
        """
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.max_length = max_length
        self.with_text = with_text
        self.is_dual = is_dual

    def truncate_sample(self, sample):
        src, tgt = sample["src"][: self.max_length], sample["tgt"][: self.max_length]
        if self.is_dual:
            if src[-1] != self.eos_idx:
                src[-1] = self.eos_idx

        if tgt[-1] != self.eos_idx:
            tgt[-1] = self.eos_idx
        return src, tgt

    def __call__(self, batch):
        # |batch| = [{"src": tensor[], "tgt": tensor[]}, {"src": tensor[], "tgt": tensor[]}...]

        srcs, tgts = [], []

        # If there are raw text passed from batch, include them in the returned value
        # If length of src or target is larger than max_length, truncate it.
        # Be careful not to exclude EOS token when truncating the sentence.
        if self.with_text:
            srcs_texts, tgts_texts = [], []

            for sample in batch:
                src, tgt = self.truncate_sample(sample)
                srcs.append((src, len(src)))
                tgts.append((tgt, len(tgt)))

                srcs_texts.append(" ".join(sample["src_text"].split(" ")[: self.max_length - 2]))
                tgts_texts.append(" ".join(sample["tgt_text"].split(" ")[: self.max_length - 2]))

        else:
            for sample in batch:
                src, tgt = self.truncate_sample(sample)
                srcs.append((src, len(src)))
                tgts.append((tgt, len(tgt)))

        # |srcs| = [(src_ids, src_length), (src_ids, src_length) ...]
        # |srcs_texts| = [src_text, src_text, ...]

        # Pad Sequence with pad token according to the length
        srcs, srcs_lengths = zip(*srcs)
        tgts, tgts_lengths = zip(*tgts)
        # |srcs| = [[src_ids], [src_ids] ...]
        # |srcs_lenghts| = [src_length, src_length]

        srcs = pad_sequence(srcs, batch_first=True, padding_value=self.pad_idx)
        tgts = pad_sequence(tgts, batch_first=True, padding_value=self.pad_idx)
        # |srcs| = (batch_size, batch_max_length)

        srcs = (srcs, torch.LongTensor(srcs_lengths))
        tgts = (tgts, torch.LongTensor(tgts_lengths))

        return_value = {
            "input_ids": srcs,
            "output_ids": tgts,
        }

        if self.with_text:
            return_value["input_texts"] = srcs_texts
            return_value["output_texts"] = tgts_texts

        return return_value
