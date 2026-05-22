########################################################################
########################################################################
##                                                                    ##
##                      ORIGINAL _ DO NOT PUBLISH                     ##
##                                                                    ##
########################################################################
########################################################################
import pandas as pd
import torch as tr
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld


batch_size = 32
output_size = 2
hidden_size = 64        # to experiment with

run_recurrent = True    # else run Token-wise MLP
use_RNN = True          # otherwise GRU
atten_size = 5          # atten > 0 means using restricted self atten

reload_model = False
num_epochs = 10
learning_rate = 0.001
test_interval = 50

# Loading sataset, use toy = True for obtaining a smaller dataset

train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size)

# Special matrix multipication layer (like torch.Linear but can operate on arbitrary sized
# tensors and considers its last two indices as the matrix.)

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias = True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels,out_channels)), requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1,1,out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):        
        x = torch.matmul(x,self.matrix) 
        if self.use_bias:
            x = x+ self.bias 
        return x
        
# Implements RNN Unit
class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size

        # RNN cell:
        # input is [current word embedding, previous hidden state]
        # output is the next hidden state
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)

        # Output MLP:
        # after the full review is parsed, hidden state -> 2 sentiment logits
        self.hidden2output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def name(self):
        return "RNN"

    def forward(self, x, hidden_state):
        # x shape: [batch_size, input_size]
        # hidden_state shape: [batch_size, hidden_size]
        combined = torch.cat((x, hidden_state), dim=1)
        hidden = torch.tanh(self.in2hidden(combined))
        output = self.hidden2output(hidden)
        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size, device=next(self.parameters()).device)

# Implements GRU Unit

class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()

        self.hidden_size = hidden_size

        # GRU gates.
        # Each gate gets the current word vector x_t and previous hidden state h_{t-1}
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # Candidate hidden-state update.
        # Here we use x_t together with the reset-filtered hidden state.
        self.candidate_layer = nn.Linear(input_size + hidden_size, hidden_size)

        # Output MLP: final hidden state -> 2 sentiment logits
        self.hidden2output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def name(self):
        return "GRU"

    def forward(self, x, hidden_state):
        # x shape: [batch_size, input_size]
        # hidden_state shape: [batch_size, hidden_size]

        # Combine input word and previous hidden state
        combined = torch.cat((x, hidden_state), dim=1)

        # Reset gate and update gate
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))

        # Candidate hidden state
        reset_hidden = r * hidden_state
        candidate_combined = torch.cat((x, reset_hidden), dim=1)
        h_tilde = torch.tanh(self.candidate_layer(candidate_combined))

        # Final hidden state
        # z close to 1 -> keep old memory
        # z close to 0 -> use new candidate memory
        hidden = z * hidden_state + (1 - z) * h_tilde

        # Sentiment prediction from current hidden state
        output = self.hidden2output(hidden)

        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size, device=next(self.parameters()).device)


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()

        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, hidden_size)
        self.layer3 = MatMul(hidden_size, output_size)

    def name(self):
        return 'MLP'

    def forward(self, x):
        # x shape: [batch_size, num_words, input_size]

        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.ReLU(x)
        x = self.dropout(x)

        x = self.layer3(x)

        # output shape: [batch_size, num_words, 2]
        # These are raw logits, not probabilities.
        return x


class ExLRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExLRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.atten_size = atten_size
        self.window_size = 2 * self.atten_size + 1
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))

        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=2)

        # First token-wise MLP projection: word embedding -> hidden representation
        self.layer1 = MatMul(input_size, hidden_size)

        # Learnable single-head attention matrices
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)

        # Learnable relative positional encoding for offsets:
        # [-5, -4, ..., 0, ..., +4, +5]
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 1, self.window_size, hidden_size)
        )
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        # Continue like the MLP from Task 2, after contextualization
        self.layer2 = MatMul(hidden_size, hidden_size)
        self.layer3 = MatMul(hidden_size, output_size)

    def name(self):
        return "MLP_atten"

    def forward(self, x):
        # Token-wise embedding projection
        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        # We build local neighborhoods of size 2*atten_size + 1 around each word.
        padded = pad(x, (0, 0, self.atten_size, self.atten_size, 0, 0))
        x_nei = []
        for k in range(-self.atten_size, self.atten_size + 1):
            x_nei.append(torch.roll(padded, k, dims=1))
        x_nei = torch.stack(x_nei, dim=2)
        # Remove the padded positions from the center-word axis
        # final x_nei shape: [batch_size, num_words, window_size, hidden_size]
        x_nei = x_nei[:, self.atten_size:-self.atten_size, :, :]
        # Add relative positional encoding to the neighbor vectors.
        # This tells the model whether a neighbor is left/right/near/far.
        x_nei = x_nei + self.pos_embedding
        # Single-head restricted self-attention
        query = self.W_q(x)          # [batch_size, num_words, hidden_size]
        keys = self.W_k(x_nei)       # [batch_size, num_words, window_size, hidden_size]
        vals = self.W_v(x_nei)       # [batch_size, num_words, window_size, hidden_size]
        # Attention scores: dot product between each word query
        # and the keys of its local neighbors.
        atten_scores = (query.unsqueeze(2) * keys).sum(dim=-1)
        atten_scores = atten_scores / self.sqrt_hidden_size
        # Attention weights over the local window
        atten_weights = self.softmax(atten_scores)
        # Weighted average of local value vectors
        context = (atten_weights.unsqueeze(-1) * vals).sum(dim=2)
        # Residual connection: keep original word info + local contextual info
        x = x + context
        # Token-wise classifier, same idea as Task 2
        x = self.layer2(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.layer3(x)
        # These are per-word contextualized logits.
        return x, atten_weights


# prints portion of the review (20-30 first words), with the sub-scores each work obtained
# prints also the final scores, the softmaxed prediction values and the true label values

def print_review(rev_text, sbs1, sbs2, lbl1, lbl2):
    """
    rev_text: list of tokens in the review
    sbs1: per-word positive logits, shape [num_words]
    sbs2: per-word negative logits, shape [num_words]
    lbl1, lbl2: true one-hot label values:
                 positive = [1, 0], negative = [0, 1]
    """

    n_words = len(rev_text)
    n_print = min(n_words, 30)

    # Keep only real words, not padding positions
    pos_logits = np.array(sbs1[:n_words])
    neg_logits = np.array(sbs2[:n_words])

    logits = np.stack([pos_logits, neg_logits], axis=1)

    # Stable softmax for readable probabilities
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Final review-level score by averaging logits
    final_logits = logits.mean(axis=0)
    final_shifted = final_logits - final_logits.max()
    final_probs = np.exp(final_shifted) / np.exp(final_shifted).sum()

    true_label = 'positive' if lbl1 > lbl2 else 'negative'
    pred_label = 'positive' if final_probs[0] > final_probs[1] else 'negative'

    print("\nReview preview:")
    print(" ".join(rev_text[:n_print]))

    print(f"\nTrue label: {true_label}")
    print(f"Predicted label: {pred_label}")
    print(f"Final avg logits: positive={final_logits[0]:.4f}, negative={final_logits[1]:.4f}")
    print(f"Final probabilities: positive={final_probs[0]:.4f}, negative={final_probs[1]:.4f}")

    table = pd.DataFrame({
        "word": rev_text[:n_print],
        "pos_logit": pos_logits[:n_print],
        "neg_logit": neg_logits[:n_print],
        "pos_prob": probs[:n_print, 0],
        "neg_prob": probs[:n_print, 1],
    })

    print("\nPer-word sub prediction scores:")
    print(table.to_string(index=False))
    print("-" * 80)

# select model to use

if run_recurrent:
    if use_RNN:
        model = ExRNN(input_size, output_size, hidden_size)
    else:
        model = ExGRU(input_size, output_size, hidden_size)
else:
    if atten_size > 0:
        model = ExLRestSelfAtten(input_size, output_size, hidden_size)
    else:
        model = ExMLP(input_size, output_size, hidden_size)

# Move model parameters to the same device as the data: CPU or GPU
model = model.to(ld.device)

print("Using model: " + model.name())

if reload_model:
    print("Reloading model")
    model.load_state_dict(torch.load(model.name() + ".pth"))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = 1.0
test_loss = 1.0

train_acc = 0.0
test_acc = 0.0

test_iterator = iter(test_dataset)

# training steps in which a test step is executed every test_interval

for epoch in range(num_epochs):

    itr = 0 # iteration counter within each epoch

    for labels, reviews, reviews_text in train_dataset:   # getting training batches

        itr = itr + 1

        if (itr + 1) % test_interval == 0:
            test_iter = True
            
            # Safely grab the next test batch
            try:
                labels, reviews, reviews_text = next(test_iterator) 
            except StopIteration:
                # If we run out of test data, reset the iterator!
                test_iterator = iter(test_dataset)
                labels, reviews, reviews_text = next(test_iterator)
                
        else:
            test_iter = False


        # Recurrent nets (RNN/GRU)
        target_labels = torch.argmax(labels, dim=1) if labels.dim() == 2 else labels
        if run_recurrent:
            hidden_state = model.init_hidden(int(labels.shape[0]))

            for i in range(num_words):
                output, hidden_state = model(reviews[:,i,:], hidden_state)  # HIDE

        else:  

        # Token-wise networks (MLP / MLP + Atten.) 
        
            sub_score = []
            if atten_size > 0:  
                # MLP + atten
                sub_score, atten_weights = model(reviews)
            else:               
                # MLP
                sub_score = model(reviews)

            # Creates a mask of shape [batch_size, num_words] (1 for real words, 0 for padding)
            mask = (reviews.abs().sum(dim=-1) > 0).float()

            # sub_score shape: [batch_size, num_words, 2]
            # Expand mask to match sub_score shape: [batch_size, num_words, 1]
            expanded_mask = mask.unsqueeze(-1)

            # Zero out padded sub-scores
            masked_sub_scores = sub_score * expanded_mask

            # Sum the scores and divide by the actual number of words (sum of the mask)
            output = masked_sub_scores.sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)
            target_labels = torch.argmax(labels, dim=1) if labels.dim() == 2 else labels
            
        # cross-entropy loss
        loss = criterion(output, target_labels)

        predictions = torch.argmax(output, dim=1)
        batch_acc = (predictions == target_labels).float().mean().item()

        # optimize in training iterations

        if not test_iter:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # averaged losses
        if test_iter:
            test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
            test_acc = 0.8 * batch_acc + 0.2 * test_acc
        else:
            train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss
            train_acc = 0.9 * batch_acc + 0.1 * train_acc

        if test_iter:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{itr + 1}/{len(train_dataset)}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, "
                f"Test Acc: {test_acc:.4f}"
            )

            if not run_recurrent:
                nump_subs = sub_score.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                print_review(reviews_text[0], nump_subs[0,:,0], nump_subs[0,:,1], labels[0,0], labels[0,1])

            # saving the model
            torch.save(model, model.name() + ".pth")