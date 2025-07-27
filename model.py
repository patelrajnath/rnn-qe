import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionRNNQE(nn.Module):
    """
    RNN-based Translation Quality Estimation model with attention mechanism.
    Based on the paper "Translation Quality Estimation using Recurrent Neural Network"
    """
    
    def __init__(self, vocab_size_src, vocab_size_tgt, embed_dim=300, hidden_dim=256, 
                 num_layers=2, dropout=0.3, bidirectional=True):
        super(AttentionRNNQE, self).__init__()
        
        self.vocab_size_src = vocab_size_src
        self.vocab_size_tgt = vocab_size_tgt
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layers
        self.src_embedding = nn.Embedding(vocab_size_src, embed_dim, padding_idx=0)
        self.tgt_embedding = nn.Embedding(vocab_size_tgt, embed_dim, padding_idx=0)
        
        # RNN layers for source and target
        self.src_rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.tgt_rnn = nn.LSTM(embed_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        # Attention mechanism
        self.attention = nn.Linear(self.num_directions * hidden_dim * 2, 1)
        
        # Feature combination layers
        combined_dim = self.num_directions * hidden_dim * 4  # src + tgt + attention context
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.fc3 = nn.Linear(128, 1)  # Quality score output
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src_tokens, tgt_tokens, src_lengths, tgt_lengths):
        """
        Forward pass for the QE model
        
        Args:
            src_tokens: [batch_size, src_len]
            tgt_tokens: [batch_size, tgt_len]
            src_lengths: [batch_size]
            tgt_lengths: [batch_size]
            
        Returns:
            quality_scores: [batch_size, tgt_len]
        """
        batch_size = src_tokens.size(0)
        
        # Source encoding
        src_embed = self.src_embedding(src_tokens)  # [batch, src_len, embed_dim]
        src_packed = nn.utils.rnn.pack_padded_sequence(
            src_embed, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        src_out, (src_hidden, _) = self.src_rnn(src_packed)
        src_out, _ = nn.utils.rnn.pad_packed_sequence(src_out, batch_first=True)
        # src_out: [batch, src_len, hidden_dim * num_directions]
        
        # Target encoding
        tgt_embed = self.tgt_embedding(tgt_tokens)  # [batch, tgt_len, embed_dim]
        tgt_packed = nn.utils.rnn.pack_padded_sequence(
            tgt_embed, tgt_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        tgt_out, (tgt_hidden, _) = self.tgt_rnn(tgt_packed)
        tgt_out, _ = nn.utils.rnn.pad_packed_sequence(tgt_out, batch_first=True)
        # tgt_out: [batch, tgt_len, hidden_dim * num_directions]
        
        # Attention mechanism
        quality_scores = []
        
        for i in range(tgt_out.size(1)):  # Iterate over target tokens
            tgt_t = tgt_out[:, i, :].unsqueeze(1)  # [batch, 1, hidden_dim * num_directions]
            
            # Compute attention scores
            # Expand target token to match source sequence length
            tgt_expanded = tgt_t.expand(-1, src_out.size(1), -1)  # [batch, src_len, hidden_dim * num_directions]
            
            # Concatenate source and target representations
            combined = torch.cat([src_out, tgt_expanded], dim=2)  # [batch, src_len, hidden_dim * num_directions * 2]
            
            # Compute attention weights
            attention_scores = self.attention(combined).squeeze(2)  # [batch, src_len]
            attention_weights = F.softmax(attention_scores, dim=1)  # [batch, src_len]
            
            # Compute context vector
            context = torch.bmm(attention_weights.unsqueeze(1), src_out).squeeze(1)  # [batch, hidden_dim * num_directions]
            
            # Combine features for quality prediction
            combined_features = torch.cat([
                context,  # Source context
                tgt_t.squeeze(1),  # Target representation
                context * tgt_t.squeeze(1),  # Element-wise product
                torch.abs(context - tgt_t.squeeze(1))  # Absolute difference
            ], dim=1)  # [batch, hidden_dim * num_directions * 4]
            
            # Predict quality score
            hidden = F.relu(self.fc1(self.dropout(combined_features)))
            hidden = F.relu(self.fc2(self.dropout(hidden)))
            score = torch.sigmoid(self.fc3(hidden))  # [batch, 1]
            
            quality_scores.append(score)
        
        # Stack scores for all target tokens
        quality_scores = torch.stack(quality_scores, dim=1).squeeze(2)  # [batch, tgt_len]
        
        return quality_scores


class BaselineRNNQE(nn.Module):
    """
    Simpler baseline RNN model for comparison
    """
    
    def __init__(self, vocab_size_src, vocab_size_tgt, embed_dim=200, hidden_dim=128):
        super(BaselineRNNQE, self).__init__()
        
        self.src_embedding = nn.Embedding(vocab_size_src, embed_dim, padding_idx=0)
        self.tgt_embedding = nn.Embedding(vocab_size_tgt, embed_dim, padding_idx=0)
        
        self.src_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.tgt_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(hidden_dim * 4, 1)  # Concatenate last hidden states
        
    def forward(self, src_tokens, tgt_tokens, src_lengths, tgt_lengths):
        batch_size = src_tokens.size(0)
        
        # Source encoding
        src_embed = self.src_embedding(src_tokens)
        src_packed = nn.utils.rnn.pack_padded_sequence(
            src_embed, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, src_hidden = self.src_rnn(src_packed)
        src_hidden = src_hidden.transpose(0, 1).contiguous().view(batch_size, -1)  # [batch, hidden_dim * 2]
        
        # Target encoding
        tgt_embed = self.tgt_embedding(tgt_tokens)
        tgt_packed = nn.utils.rnn.pack_padded_sequence(
            tgt_embed, tgt_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, tgt_hidden = self.tgt_rnn(tgt_packed)
        tgt_hidden = tgt_hidden.transpose(0, 1).contiguous().view(batch_size, -1)  # [batch, hidden_dim * 2]
        
        # Combine and predict
        combined = torch.cat([src_hidden, tgt_hidden], dim=1)
        score = torch.sigmoid(self.fc(combined))
        
        return score


if __name__ == "__main__":
    # Quick test
    model = AttentionRNNQE(vocab_size_src=1000, vocab_size_tgt=1000)
    
    # Dummy data
    src = torch.randint(1, 100, (32, 20))
    tgt = torch.randint(1, 100, (32, 15))
    src_len = torch.randint(10, 20, (32,))
    tgt_len = torch.randint(5, 15, (32,))
    
    scores = model(src, tgt, src_len, tgt_len)
    print(f"Output shape: {scores.shape}")  # Should be [32, 15]