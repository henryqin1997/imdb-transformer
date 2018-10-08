import torch
from torch import nn
from torch.autograd import Variable

from pos import get_pos_onehot

class MultiHeadAttention(nn.Module):
    def __init__(self,input_size,hidden_size,num_heads):
        super(MultiHeadAttention,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.head_size = self.hidden_size / num_heads
        print "HEAD SIZE",self.head_size
        self.q_linear = nn.Linear(self.input_size,self.hidden_size)
        self.k_linear = nn.Linear(self.input_size,self.hidden_size)
        self.v_linear = nn.Linear(self.input_size,self.hidden_size)

        self.joint_linear = nn.Linear(self.hidden_size,self.hidden_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self,q,k,v):
        q_proj = self.q_linear(q).view(q.size(0), q.size(1), self.num_heads,self.head_size).transpose(1,2)
        k_proj = self.k_linear(k).view(k.size(0), k.size(1), self.num_heads,self.head_size).transpose(1,2)
        v_proj = self.v_linear(v).view(v.size(0), v.size(1), self.num_heads,self.head_size).transpose(1,2)


        unscaled_weights = torch.matmul(q_proj,k_proj.transpose(2,3))
        weights = self.softmax(unscaled_weights / torch.sqrt(torch.Tensor([self.head_size * 1.0]).to(unscaled_weights)))

        weighted_v = torch.matmul(weights,v_proj)

        weighted_v = weighted_v.transpose(1,2).contiguous()

        joint_proj = self.joint_linear(weighted_v.view(q.size(0),q.size(1),self.hidden_size))


        return joint_proj

class NoOp(nn.Module):
    def forward(self,x):
        return x

class Block(nn.Module):
    def __init__(self,input_size,hidden_size,num_heads,activation=nn.ReLU):
        super(Block,self).__init__()
        self.attention = MultiHeadAttention(input_size,hidden_size,num_heads)
        self.attention_norm = nn.LayerNorm(input_size)
        self.attention_dropout = nn.Dropout(0.1)
        self.ff = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            activation(),
            nn.Linear(hidden_size,input_size),
            nn.Dropout(0.1),
            )
        self.ff_norm = nn.LayerNorm(input_size)

    def forward(self,x):
        attended = self.attention_norm(self.attention_dropout(self.attention(x,x,x)) + x)
        return self.ff_norm(self.ff(attended) + x)



class Transformer(nn.Module):
    def __init__(self,input_size,hidden_size,ff_size,num_blocks,num_heads,activation=nn.ReLU):
        """
            Class defining the Transformer Network
        """
        super(Transformer,self).__init__()

        self.blocks = nn.Sequential(
                *[Block(input_size,hidden_size,num_heads,activation) 
                    for _ in xrange(num_blocks)]
            )

    def forward(self,x):
        return self.blocks(x)

class Net(nn.Module):
    def __init__(self,embeddings,max_length):
        super(Net,self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings,freeze=False)
        self.model_size = 128
        self.emb_ff = nn.Linear(300,self.model_size)
        self.pos = nn.Linear(max_length,self.model_size)
        self.max_length = max_length
        self.transformer = Transformer(self.model_size,self.model_size,self.model_size,1,1)
        self.output = nn.Linear(self.model_size,2)

    def forward(self,x):
        x_size = x.size()
        x = x.view(-1)
        x = self.emb_ff(self.embeddings(x))
        pos = self.pos(get_pos_onehot(self.max_length).to(x)).unsqueeze(0)
        x = x.view(*(x_size + (self.model_size,)))
        #x += pos
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.output(x)



if __name__ == "__main__":
    t = Transformer(10,20,30,3,5)

    input = Variable(torch.rand(40,20,10))

    print t(input)
