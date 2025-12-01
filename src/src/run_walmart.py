import pickle
import numpy as np
import torch
from hypergraph_funcs import Hypergraph
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from losses import InfoNCE
from utils import accuracy, clip_gradient, Evaluation, AdditionalLayer
from model_3_0 import IGNN,final_predictor
from normalization import aug_normalized_adjacency
from sklearn.model_selection import train_test_split
#with open('../data/walmart-trips/hyperedges-walmart-trips.txt', 'rb') as handle:
    #hypergraph = pickle.load(handle)
# Initialize an empty list to store lists
hypergraph1 = []
file_path='../data/walmart-trips/hyperedges-walmart-trips.txt'#'../data/walmart-trips/hyperedges-walmart-trips.txt'
# Open and read the file
with open(file_path, 'r') as file:
    for line in file:
        # Split the line by commas, convert each element to an integer, and append it to the list
        hypergraph1.append([int(num) for num in line.strip().split(',')])

#with open('../data/cocitation/citeseer/features.pickle', 'rb') as handle:
#    features = pickle.load(handle).todense()
#with open('../data/walmart-trips/node-labels-walmart-trips.txt', 'rb') as handle:
    #labels = pickle.load(handle)
labels1=[]
with open('../data/walmart-trips/node-labels-walmart-trips.txt', 'r') as file:
    for line in file:
        # Strip newline characters and append the line as a string to the list
        labels1.append(int(line.strip()))
labels=[x-1 for x in labels1]
hypergraph = [[x - 1 for x in sublist] for sublist in hypergraph1]
#print(max(len(sublist) for sublist in hypergraph))
#import pdb;pdb.set_trace()
#with open('../data/cocitation/citeseer/splits/1.pickle', 'rb') as handle:
    #split = pickle.load(handle)
overall=list(range(0,len(labels)))#88860
train, test = train_test_split(overall, test_size=0.7, random_state=1)

#import pdb;pdb.set_trace()
#train=split['train']
#test=split['test']
classes = set(labels)
onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
labels_class=np.array(list(map(onehot.get, labels)), dtype=np.int32)
my_hypergraph=Hypergraph(num_v=len(labels),e_list=hypergraph,device='cuda:0')
with torch.no_grad():
    features=nn.Embedding(len(labels),128).weight.detach().cpu().numpy()#88860
#edge_feat=[]
edge_feat=torch.load('../data/output_submit/walmart_edge.pt')
#adj=torch.from_numpy(row_normalize(my_hypergraph.L_HGNN.to_dense().cpu())).to_sparse().to('cuda:5')
class node_in_edge_loss(nn.Module):
    def __init__(self, subgraph_list, embedding_dim=128, hidden_dim=128):
        super(node_in_edge_loss, self).__init__()
        self.subgraph_list = subgraph_list  # List of edges, where each edge is a list of nodes
        self.classifier = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 2)  # Output logits for binary classification
        )

    def forward(self, node_embeddings, edge_embeddings, batch_size=1024):
        """
        Args:
            node_embeddings: Tensor of shape (num_nodes, embedding_dim)
            edge_embeddings: Tensor of shape (num_edges, embedding_dim)
            batch_size: Number of (node, edge) pairs to sample
        Returns:
            loss: Cross-entropy loss for the batch
        """
        
        num_edges = len(self.subgraph_list)
        num_nodes = node_embeddings.shape[0]

        # Sample a batch of edges
        edge_indices = torch.randint(0, num_edges, (batch_size,))
        sampled_edges = [self.subgraph_list[i] for i in edge_indices]  # List of lists of nodes

        # Sample nodes (some inside and some outside edges)
        node_indices = []
        labels = []
        for edge_nodes in sampled_edges:
            if torch.rand(1).item() < 0.5:  # 50% chance of picking a node from the edge
                node = torch.tensor(edge_nodes[torch.randint(0, len(edge_nodes), (1,)).item()])
                label = 1  # Node belongs to the edge
            else:  
                # Sample a node that is not in the edge
                while True:
                    node = torch.randint(0, num_nodes, (1,)).item()
                    if node not in edge_nodes:
                        break
                node = torch.tensor(node)
                label = 0  # Node does not belong to the edge
            
            node_indices.append(node)
            labels.append(label)

        node_indices = torch.stack(node_indices)  # (batch_size,)
        labels = torch.tensor(labels, dtype=torch.long, device=node_embeddings.device)  # (batch_size,)

        # Get embeddings
        node_emb = node_embeddings[node_indices]  # (batch_size, embedding_dim)
        edge_emb = edge_embeddings[edge_indices]  # (batch_size, embedding_dim)

        # Concatenate node and edge embeddings
        combined_emb = torch.cat([node_emb, edge_emb], dim=1)  # (batch_size, 2*embedding_dim)

        # Compute logits
        logits = self.classifier(combined_emb)  # (batch_size, 2)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss



class hyperedge_loss(nn.Module):
    def __init__(self, train_label_matrix):
        super(hyperedge_loss, self).__init__()
        self.train_label_matrix = train_label_matrix
        self.train_label_mask = train_label_matrix.sum(dim=1) > 0

        # Compute label_vector
        self.label_vector = self.compute_label_vector()

    def compute_label_vector(self):
        """
        Compute label vector where each row takes the index of the largest element in that row.
        If a row contains all zeros, assign label 0.
        """
        num_rows, num_labels = self.train_label_matrix.shape

        # Add small random noise to break ties arbitrarily
        noise = torch.rand_like(self.train_label_matrix, dtype=torch.float) * 1e-6
        noisy_matrix = self.train_label_matrix.float() + noise  # Convert to float to avoid rounding issues

        # Get indices of max elements in each row with randomness
        max_indices = noisy_matrix.argmax(dim=1) + 1  # Add 1 to make the label 1-indexed

        # Assign label 0 where the row is all zeros
        label_vector = torch.where(self.train_label_mask, max_indices, torch.tensor(0, device=self.train_label_matrix.device))

        return label_vector
    
    def sample_positives(self):
        """
        Sample one positive sample for each edge, where the positive has the same label.

        Returns:
            torch.Tensor: A tensor containing indices of sampled positive pairs for each edge.
        """
        batch_size = self.train_label_matrix.size(0)
        positive_indices = []

        for i in range(batch_size):
            # Get indices where the label matches the current entity's label (excluding the entity itself)
            same_label_indices = (self.label_vector == self.label_vector[i]) & (self.label_vector > 0)
            same_label_indices = same_label_indices.nonzero().squeeze(1)
            same_label_indices = same_label_indices[same_label_indices != i]  # Exclude the current entity itself

            # Randomly select one positive index from the available ones
            if same_label_indices.size(0) > 0:
                positive_index = same_label_indices[torch.randint(0, same_label_indices.size(0), (1,))]
                positive_indices.append(positive_index.item())
            else:
                positive_indices.append(i)  # If no positive found, fallback to itself (can be adjusted if needed)

        return torch.tensor(positive_indices, device=self.train_label_matrix.device)

    def one_one_cos_sim(self, embeddings_1, embeddings_2):
        """
        Compute cosine similarity between two sets of embeddings that have the same shape.

        Args:
            embeddings_1 (torch.Tensor): Tensor of shape (num_samples, embedding_dim)
            embeddings_2 (torch.Tensor): Tensor of shape (num_samples, embedding_dim)

        Returns:
            torch.Tensor: Tensor of shape (num_samples,) containing cosine similarity values.
        """

        # Normalize the embeddings
        norm_1 = F.normalize(embeddings_1, p=2, dim=-1)
        norm_2 = F.normalize(embeddings_2, p=2, dim=-1)

        # Compute cosine similarity
        sim = (norm_1 * norm_2).sum(dim=-1)

        return sim

        

    def forward(self, edge_embeddings, temperature=0.1, num_samples=256):
        """
        Compute the InfoNCE loss for hyperedges using their embeddings with sampled positives and negatives.

        Args:
            edge_embeddings (torch.Tensor): Tensor of shape (num_edges, embedding_dim)
            temperature (float): Temperature parameter for softmax.
            num_samples (int): Number of negative samples to use.

        Returns:
            torch.Tensor: Scalar InfoNCE loss value.
        """
        batch_size = edge_embeddings.size(0)
        
        # Sample positive pairs
        positive_indices = self.sample_positives()

        # Sample negative pairs randomly from the entire batch
        negative_indices = torch.randint(0, batch_size, (num_samples,))

        # Compute cosine similarity for the sampled pairs
        positive_embeddings = edge_embeddings[positive_indices]
        negative_embeddings = edge_embeddings[negative_indices]
        positive_sim = self.one_one_cos_sim(edge_embeddings, positive_embeddings)
        negative_sim = F.cosine_similarity(edge_embeddings.unsqueeze(1), negative_embeddings.unsqueeze(0), dim=-1)

        # Compute the logits (similarity scaled by temperature)
        positive_logit = positive_sim / temperature
        negative_logit = negative_sim / temperature
        
        # Compute loss
        positive = positive_logit
        negative = torch.logsumexp(negative_logit, dim=1) - torch.log(torch.tensor(num_samples, device=self.train_label_matrix.device))
        # import pdb;pdb.set_trace()
        loss = - (positive - negative).mean()

        return loss



def compute_train_label_matrix(subgraphs, labels, train_nodes):
    """
    Compute a matrix where each row corresponds to a subgraph and each column corresponds to a label.
    The entry (i, j) represents the number of train nodes in subgraph i with label j.

    Args:
        subgraphs (list of list of int): List of subgraphs, each represented as a list of node indices.
        labels (torch.Tensor): Tensor of shape (num_nodes,) containing node labels.
        train_nodes (list of int): List of node indices that are in the training set.

    Returns:
        torch.Tensor: Matrix of shape (num_subgraphs, num_labels) with label counts per subgraph.
    """
    num_subgraphs = len(subgraphs)
    num_labels = labels.max().item() + 1  # Assuming labels are 0-indexed

    # Convert train_nodes to a set for fast lookup
    train_set = set(train_nodes)

    # Initialize matrix
    M = torch.zeros((num_subgraphs, num_labels), dtype=torch.int32)

    # Compute the matrix
    for i, subgraph_nodes in enumerate(subgraphs):
        train_nodes_in_subgraph = [node for node in subgraph_nodes if node in train_set]  # Filter train nodes
        train_labels = labels[train_nodes_in_subgraph]  # Get their labels
        for label in train_labels:
            M[i, label] += 1  # Count occurrences of each label

    return M
block=torch.cat([torch.cat([torch.zeros(len(labels),len(labels)).to_sparse().to('cuda:0'),my_hypergraph.e2n], dim=1),torch.cat([my_hypergraph.n2e,torch.zeros(len(my_hypergraph.e[0]),len(my_hypergraph.e[0])).to_sparse().to('cuda:0')], dim=1)], dim=0) 
adj1=aug_normalized_adjacency(block.cpu().to_dense())
adj=torch.sparse_coo_tensor(adj1.nonzero(), adj1.data, adj1.shape).to('cuda:2')
#import pdb;pdb.set_trace()
my_hypergraph=my_hypergraph.to('cpu')
torch.cuda.empty_cache()
features_torch=torch.cat([torch.from_numpy(features).to('cuda:2'),edge_feat.to('cuda:2')],dim=0)
labels_torch=torch.from_numpy(np.array(labels)).to('cuda:2')
model=IGNN(nfeat=features_torch.shape[1],
           nhid=128,
           nclass=len(classes),
           num_node=(len(labels)+len(my_hypergraph.e[0])),
           dropout=0.2, #0.1
           kappa=0.9).to('cuda:2')
subgraph_list = my_hypergraph.e[0]
train_label_matrix = compute_train_label_matrix(subgraph_list, labels_torch, train)
edge_loss = node_in_edge_loss(subgraph_list).to('cuda:2')
#model_pred=final_predictor(nfeat=64,head=4,nclass=len(classes)).to('cuda:1')
optimizer=optim.Adam((list(model.parameters()) + list(edge_loss.parameters())),
                     lr=0.001,weight_decay=5e-4) #0.0009   5e-4

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=130, gamma=0.1)
split_index=len(labels)
epochs=200
best_ep=0
alpha = 0.4
features_torch=features_torch.T
testy=0.0
train_losses=[]
train_class_losses=[]
train_accs = []
test_accs = []
#import pdb;pdb.set_trace()
for ep in range(epochs):
    model.train()
    optimizer.zero_grad()
    #import pdb;pdb.set_trace()
    output,predemb = model(features_torch, adj)
    edge_embeddings = output[split_index:]
    node_embeddings=output[:split_index]
    predemb=predemb[:split_index]
    #import pdb;pdb.set_trace()
    #output=model_pred((output[:split_index]),(output[split_index:]))
    node_logits = F.log_softmax(predemb, dim=1)
    loss_train = F.nll_loss(node_logits[train], labels_torch[train])
    train_class_losses.append(loss_train.item())
    edge_loss_train = edge_loss(node_embeddings, edge_embeddings)
    # import pdb;pdb.set_trace()
    loss_train += alpha*edge_loss_train

    acc_train = accuracy(node_logits[train], labels_torch[train])


    train_losses.append(edge_loss_train.item())
    loss_train.backward()
    # clip_gradient(model, clip_norm=0.5)
    optimizer.step()
    lr_scheduler.step()
    model.eval()
    _,output = model(features_torch, adj)
    output=output[:split_index]
    #output=model_pred(output[:split_index],output[split_index:])
    output = F.log_softmax(output, dim=1)
    acc_test = accuracy(output[test], labels_torch[test])
    train_accs.append(acc_train.item())
    test_accs.append(acc_test.item())

    print('Epoch: {:03d}'.format(ep+1),
          'Loss: {:.4f}'.format(loss_train.item()), 
          'Edge Loss: {:.4f}'.format(edge_loss_train.item()),
          'Train Acc: {:.4f}'.format(acc_train.item()),
            'Test Acc: {:.4f}'.format(acc_test.item())
    )
    
    #print('Train Loss')
    #print(loss_train.item())
    if testy<acc_test:
        testy=acc_test
        best_ep=ep
        print('Test acc')
        print(testy)
print('Best Test Accuracy')
print(testy)
print('best Epoch')
print(best_ep)


import matplotlib.pyplot as plt
plt.plot(train_losses, label='Edge')
plt.plot(train_class_losses, label='Class')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Epochs')
plt.savefig('plots/walmart/training_loss_seed_9.png')
plt.close()


# plot the accuracy figure
plt.plot(train_accs, label='Train')
plt.plot(test_accs, label='Test')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.savefig('plots/walmart/acc.png')
plt.close()