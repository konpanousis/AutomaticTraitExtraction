# specify device
from torch import cuda
import torch.nn as nn
import transformers
from transformers import DistilBertTokenizer, DistilBertModel
import warnings
import torch
import torch
from transformers import DistilBertTokenizer, DistilBertModel
    
# Load the BERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  
bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
warnings.filterwarnings("ignore")

def loadBERT(location, modelname):
    
    """
    Load the pretrained BERT model for text description data.
    """
    
    warnings.filterwarnings("ignore")
    device = 'cuda' if cuda.is_available() else 'cpu'
    #bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

    
    class BERT(nn.Module):
        def __init__(self, bert):

            super(BERT, self).__init__()

            # Distil Bert model
            self.bert = bert
            ## Additional layers
            # Dropout layer
            self.dropout = nn.Dropout(0.1)
            # Relu activation function
            self.relu =  nn.ReLU()
            # Dense layer 1
            self.fc1 = nn.Linear(768, 512)
            # Dense layer 2 (Output layer)
            self.fc2 = nn.Linear(512, 2)
            # Softmax activation function
            self.softmax = nn.LogSoftmax(dim=1)

        #define the forward pass
        def forward(self, **kwargs):

            #pass the inputs to the model BERT  
            cls_hs = self.bert(**kwargs)
            hidden_state = cls_hs.last_hidden_state
            pooler = hidden_state[:, 0]

            # dense layer 1        
            x = self.fc1(pooler)
            # ReLU activation
            x = self.relu(x)
            # Drop out
            x = self.dropout(x)
            # dense layer 2
            x = self.fc2(x)
            # apply softmax activation
            x = self.softmax(x)

            return x
        
    model = BERT(bert)
    # push the model to GPU
    model = model.to(device)

    # Load trained model (colab)
    try:
        model_save_name = modelname
        path = location + model_save_name
        model.load_state_dict(torch.load(path))
        print('Cuda Success')

    except:
        model_save_name = modelname
        path = location + model_save_name
        model.load_state_dict(torch.load(path, 
                                         map_location=torch.device('cpu')))
        print('CPU Success')

    model.eval()
    
    return model

def load_simBERT():
    
    """
    Load BERT for sentence similarity.
    """
    
    device = 'cuda' if cuda.is_available() else 'cpu'
    bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    
    class BERT(nn.Module):
        def __init__(self, bert):
            super(BERT, self).__init__()

            # Distil Bert model
            self.bert = bert

        # Define the forward pass
        def forward(self, **kwargs):

            # Pass the inputs to the model BERT  
            cls_hs = self.bert(**kwargs)
            hidden_state = cls_hs.last_hidden_state

            return hidden_state
        
    model = BERT(bert)
    # Push the model to GPU
    model = model.to(device)
    # Eval mode
    model.eval()
    
    return model

def SpanPredictor(span, model, pred_values=False, truncation=True, threshold=False):

    """
    Uses a trained bert classifier to see if a span
    belongs to a species description or otherwise.
    """
        
    with torch.no_grad():
        # Tokenize input
        inputs = tokenizer(span, return_tensors="pt", truncation=truncation)
        # Predict class
        outputs = model(**inputs)
        # Get prediction values
        exps = torch.exp(outputs)
        # Get class
        span_class = exps.argmax(1).item()

        # Print the prediction values
        if pred_values:
            return span_class, exps[0]
        else:    
            return span_class
        
        
def similarity_matrix(sentence_list, SIMmodel):
    
    """
    Calculates a similarity cosine matrix based on a list of
    sentences.
    """
    
    # Initialize dictionary to store tokenized sentences
    tokens = {'input_ids': [], 'attention_mask': []}

    for sentence in sentence_list:
        # encode each sentence and append to dictionary
        new_tokens = tokenizer.encode_plus(sentence, max_length=512,
                                           truncation=True, 
                                           padding='max_length',
                                           return_tensors='pt')
        # Drop the batch dimension
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    
    # Reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    
    # Get vectors
    hiddenstates = SIMmodel(**tokens)
    # Sum along first axis
    summed_hs = torch.sum(hiddenstates, 1)
    # Detach
    summed_hs_np = summed_hs.detach().numpy()
    # Get the matrix
    return summed_hs_np

def load_CUB_Bert(location, modelname, outputsize=2000):
    
    device = 'cuda' if cuda.is_available() else 'cpu'

    class BERT(nn.Module):
        def __init__(self, bert):

            super(BERT, self).__init__()

            # Distil Bert model
            self.bert = bert
            ## Additional layers
            # Dropout layer
            self.dropout = nn.Dropout(0.3)
            # Relu 
            self.relu =  nn.ReLU()
            # Linear I 
            self.fc1 = nn.Linear(768, 512)
            # Linear II (Out)
            #self.fc2 = nn.Linear(512, 170)
            self.fc2 = nn.Linear(512, outputsize)
            # Softmax
            self.softmax = nn.LogSoftmax(dim=1)


        # Forward pass
        def forward(self, **kwargs):

            # Pass data trough bert and extract 
            cls_hs = self.bert(**kwargs)
            # Extract hidden state
            hidden_state = cls_hs.last_hidden_state
            # Only first is needed for classification
            pooler = hidden_state[:, 0]

            # Dense layer 1        
            x = self.fc1(pooler)
            # ReLU activation
            x = self.relu(x)
            # Drop out
            x = self.dropout(x)
            # Dense layer 2
            x = self.fc2(x)
            # Activation
            x = self.softmax(x)

            return x

    # Load the entire model
    model = BERT(bert)

    # Load trained model (colab)
    try:
        try:
            model_save_name = modelname
            path = location + model_save_name
            model.load_state_dict(torch.load(path))
            print('Google Success')

        except:
            model_save_name = modelname
            path = location + model_save_name
            model.load_state_dict(torch.load(path, 
                                             map_location=torch.device('cpu')))
            print('Local Success')
    except:
        print('No pretrained model found.')

    model.to(device)
    model.eval()
    model.zero_grad()
    return model

def load_PLANT_Bert(location, modelname):
    
    device = 'cuda' if cuda.is_available() else 'cpu'

    class BERT(nn.Module):
        def __init__(self, bert):

            super(BERT, self).__init__()

            # Distil Bert model
            self.bert = bert
            ## Additional layers
            # Dropout layer
            self.dropout = nn.Dropout(0.3)
            # Relu 
            self.relu =  nn.ReLU()
            # Linear I 
            self.fc1 = nn.Linear(768, 512)
            # Linear II (Out)
            #self.fc2 = nn.Linear(512, 170)
            self.fc2 = nn.Linear(512, 14557)
            # Softmax
            self.softmax = nn.LogSoftmax(dim=1)


        # Forward pass
        def forward(self, **kwargs):

            # Pass data trough bert and extract 
            cls_hs = self.bert(**kwargs)
            # Extract hidden state
            hidden_state = cls_hs.last_hidden_state
            # Only first is needed for classification
            pooler = hidden_state[:, 0]

            # Dense layer 1        
            x = self.fc1(pooler)
            # ReLU activation
            x = self.relu(x)
            # Drop out
            x = self.dropout(x)
            # Dense layer 2
            x = self.fc2(x)
            # Activation
            x = self.softmax(x)

            return x

    # Load the entire model
    model = BERT(bert)

    # Load trained model (colab)
    try:
        try:
            model_save_name = modelname
            path = location + model_save_name
            model.load_state_dict(torch.load(path))
            print('Google Success')

        except:
            model_save_name = modelname
            path = location + model_save_name
            model.load_state_dict(torch.load(path, 
                                             map_location=torch.device('cpu')))
            print('Local Success')
    except:
        print('No pretrained model found.')

    model.to(device)
    model.eval()
    model.zero_grad()
    return model