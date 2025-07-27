import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import pickle
import os
from typing import List, Tuple, Dict, Optional


class Vocabulary:
    """Vocabulary management for source and target languages"""
    
    def __init__(self, min_freq: int = 2, max_size: int = 50000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.word_freq = Counter()
        
    def add_word(self, word: str):
        """Add a word to vocabulary"""
        self.word_freq[word] += 1
        
    def build_vocab(self):
        """Build vocabulary from collected words"""
        # Sort words by frequency
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Add words that meet minimum frequency
        for word, freq in sorted_words:
            if freq >= self.min_freq and len(self.word2idx) < self.max_size:
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    
    def encode(self, sentence: List[str]) -> List[int]:
        """Encode sentence to indices"""
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in sentence]
        
    def decode(self, indices: List[int]) -> List[str]:
        """Decode indices to sentence"""
        return [self.idx2word.get(idx, '<UNK>') for idx in indices]
        
    def __len__(self):
        return len(self.word2idx)


class QEDataset(Dataset):
    """Dataset class for Quality Estimation"""
    
    def __init__(self, src_sentences: List[List[str]], tgt_sentences: List[List[str]], 
                 quality_scores: List[float], src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                 max_length: int = 100):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.quality_scores = quality_scores
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.src_sentences)
        
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        quality_score = self.quality_scores[idx]
        
        # Encode sentences
        src_encoded = self.src_vocab.encode(src_sentence)
        tgt_encoded = self.tgt_vocab.encode(tgt_sentence)
        
        # Truncate or pad to max_length
        src_encoded = src_encoded[:self.max_length]
        tgt_encoded = tgt_encoded[:self.max_length]
        
        src_length = len(src_encoded)
        tgt_length = len(tgt_encoded)
        
        return {
            'src_tokens': torch.tensor(src_encoded, dtype=torch.long),
            'tgt_tokens': torch.tensor(tgt_encoded, dtype=torch.long),
            'src_length': torch.tensor(src_length, dtype=torch.long),
            'tgt_length': torch.tensor(tgt_length, dtype=torch.long),
            'quality_score': torch.tensor(quality_score, dtype=torch.float)
        }


class DataProcessor:
    """Data processing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
        
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple tokenization"""
        return DataProcessor.clean_text(text).split()
        
    @staticmethod
    def load_wmt_format(filepath: str) -> Tuple[List[str], List[str], List[float]]:
        """Load data in WMT QE format"""
        src_sentences = []
        tgt_sentences = []
        quality_scores = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                src_sentences.append(parts[0])
                tgt_sentences.append(parts[1])
                # Convert quality score to 0-1 scale
                score = float(parts[2])
                quality_scores.append(score)
                
        return src_sentences, tgt_sentences, quality_scores
        
    @staticmethod
    def create_synthetic_data(num_samples: int = 1000) -> Tuple[List[str], List[str], List[float]]:
        """Create synthetic data for testing when real data is not available"""
        src_sentences = []
        tgt_sentences = []
        quality_scores = []
        
        # Sample sentences
        sample_src = [
            "the cat sat on the mat",
            "i love machine learning",
            "this is a test sentence",
            "quality estimation is important",
            "neural networks are powerful",
            "the weather is nice today",
            "i enjoy reading books",
            "translation quality varies",
            "deep learning models work well",
            "language processing is complex"
        ]
        
        sample_tgt = [
            "le chat s'est assis sur le tapis",
            "j'aime l'apprentissage automatique",
            "c'est une phrase de test",
            "l'estimation de qualité est importante",
            "les réseaux de neurones sont puissants",
            "il fait beau aujourd'hui",
            "j'aime lire des livres",
            "la qualité de traduction varie",
            "les modèles d'apprentissage profond fonctionnent bien",
            "le traitement du langage est complexe"
        ]
        
        for i in range(num_samples):
            idx = i % len(sample_src)
            src_sentences.append(sample_src[idx])
            tgt_sentences.append(sample_tgt[idx])
            
            # Generate quality score based on length similarity and some noise
            src_len = len(sample_src[idx].split())
            tgt_len = len(sample_tgt[idx].split())
            length_ratio = min(src_len, tgt_len) / max(src_len, tgt_len)
            noise = np.random.normal(0, 0.1)
            score = max(0, min(1, length_ratio + 0.3 + noise))
            quality_scores.append(score)
            
        return src_sentences, tgt_sentences, quality_scores
        
    @staticmethod
    def build_vocabulary(sentences: List[str], vocab: Vocabulary):
        """Build vocabulary from sentences"""
        for sentence in sentences:
            for word in DataProcessor.tokenize(sentence):
                vocab.add_word(word)
        vocab.build_vocab()


def collate_fn(batch):
    """Custom collate function for batching"""
    src_tokens = [item['src_tokens'] for item in batch]
    tgt_tokens = [item['tgt_tokens'] for item in batch]
    src_lengths = [item['src_length'] for item in batch]
    tgt_lengths = [item['tgt_length'] for item in batch]
    quality_scores = [item['quality_score'] for item in batch]
    
    # Pad sequences
    src_tokens = torch.nn.utils.rnn.pad_sequence(src_tokens, batch_first=True, padding_value=0)
    tgt_tokens = torch.nn.utils.rnn.pad_sequence(tgt_tokens, batch_first=True, padding_value=0)
    
    # Convert to tensors
    src_lengths = torch.tensor(src_lengths, dtype=torch.long)
    tgt_lengths = torch.tensor(tgt_lengths, dtype=torch.long)
    quality_scores = torch.tensor(quality_scores, dtype=torch.float)
    
    return {
        'src_tokens': src_tokens,
        'tgt_tokens': tgt_tokens,
        'src_lengths': src_lengths,
        'tgt_lengths': tgt_lengths,
        'quality_scores': quality_scores
    }


def create_dataloaders(train_data_path: str = None, val_data_path: str = None, 
                      test_data_path: str = None, batch_size: int = 32,
                      max_length: int = 100, vocab_min_freq: int = 2,
                      use_synthetic: bool = False, synthetic_samples: int = 1000):
    """Create data loaders for training, validation, and testing"""
    
    processor = DataProcessor()
    
    # Load data
    if use_synthetic:
        print("Using synthetic data...")
        src_sentences, tgt_sentences, quality_scores = processor.create_synthetic_data(synthetic_samples)
        # Split into train/val/test
        total_samples = len(src_sentences)
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        
        train_src = [processor.tokenize(s) for s in src_sentences[:train_size]]
        train_tgt = [processor.tokenize(s) for s in tgt_sentences[:train_size]]
        train_scores = quality_scores[:train_size]
        
        val_src = [processor.tokenize(s) for s in src_sentences[train_size:train_size+val_size]]
        val_tgt = [processor.tokenize(s) for s in tgt_sentences[train_size:train_size+val_size]]
        val_scores = quality_scores[train_size:train_size+val_size]
        
        test_src = [processor.tokenize(s) for s in src_sentences[train_size+val_size:]]
        test_tgt = [processor.tokenize(s) for s in tgt_sentences[train_size+val_size:]]
        test_scores = quality_scores[train_size+val_size:]
        
    else:
        # Load real data
        print("Loading real data...")
        if train_data_path and os.path.exists(train_data_path):
            train_src_raw, train_tgt_raw, train_scores = processor.load_wmt_format(train_data_path)
            train_src = [processor.tokenize(s) for s in train_src_raw]
            train_tgt = [processor.tokenize(s) for s in train_tgt_raw]
        else:
            print("Train data not found, using synthetic data...")
            return create_dataloaders(use_synthetic=True, synthetic_samples=synthetic_samples)
            
        if val_data_path and os.path.exists(val_data_path):
            val_src_raw, val_tgt_raw, val_scores = processor.load_wmt_format(val_data_path)
            val_src = [processor.tokenize(s) for s in val_src_raw]
            val_tgt = [processor.tokenize(s) for s in val_tgt_raw]
        else:
            # Split training data for validation
            val_size = int(0.15 * len(train_src))
            val_src = train_src[-val_size:]
            val_tgt = train_tgt[-val_size:]
            val_scores = train_scores[-val_size:]
            train_src = train_src[:-val_size]
            train_tgt = train_tgt[:-val_size]
            train_scores = train_scores[:-val_size]
            
        if test_data_path and os.path.exists(test_data_path):
            test_src_raw, test_tgt_raw, test_scores = processor.load_wmt_format(test_data_path)
            test_src = [processor.tokenize(s) for s in test_src_raw]
            test_tgt = [processor.tokenize(s) for s in test_tgt_raw]
        else:
            # Use validation data as test data
            test_src = val_src
            test_tgt = val_tgt
            test_scores = val_scores
    
    # Build vocabularies
    print("Building source vocabulary...")
    src_vocab = Vocabulary(min_freq=vocab_min_freq)
    processor.build_vocabulary([' '.join(s) for s in train_src], src_vocab)
    
    print("Building target vocabulary...")
    tgt_vocab = Vocabulary(min_freq=vocab_min_freq)
    processor.build_vocabulary([' '.join(s) for s in train_tgt], tgt_vocab)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Create datasets
    train_dataset = QEDataset(train_src, train_tgt, train_scores, src_vocab, tgt_vocab, max_length)
    val_dataset = QEDataset(val_src, val_tgt, val_scores, src_vocab, tgt_vocab, max_length)
    test_dataset = QEDataset(test_src, test_tgt, test_scores, src_vocab, tgt_vocab, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab
    }


if __name__ == "__main__":
    # Test data loading
    data_loaders = create_dataloaders(use_synthetic=True, synthetic_samples=100)
    
    # Test one batch
    for batch in data_loaders['train_loader']:
        print("Batch shapes:")
        print(f"  src_tokens: {batch['src_tokens'].shape}")
        print(f"  tgt_tokens: {batch['tgt_tokens'].shape}")
        print(f"  src_lengths: {batch['src_lengths'].shape}")
        print(f"  tgt_lengths: {batch['tgt_lengths'].shape}")
        print(f"  quality_scores: {batch['quality_scores'].shape}")
        break