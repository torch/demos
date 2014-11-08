local matio = require 'matio'


function load_data(N)
--[[% This method loads the training, validation and test set.
% It also divides the training set into mini-batches.
% Inputs:
%   N: Mini-batch size.
% Outputs:
%   train_input: An array of size D X N X M, where
%                 D: number of input dimensions (in this case, 3).
%                 N: size of each mini-batch (in this case, 100).
%                 M: number of minibatches.
%   train_target: An array of size 1 X N X M.
%   valid_input: An array of size D X number of points in the validation set.
%   test: An array of size D X number of points in the test set.
%   vocab: Vocabulary containing index to word mapping.
]]--

dataset = 'data/data.mat';
data = matio.load(dataset);

vocab = data['data']['vocab'];

words = {}; 
vocab_ByIndex = {};
vocab_size = 0;
for i=1, #vocab do
   length = (#vocab[i])[2];
   word = ''
   for c=1, length do
     word = word .. string.char(vocab[i][1][c]);
   end
   words[word] = i;
   table.insert(vocab_ByIndex, word)
   vocab_size=i;
end

vocab = words;

testData = data['data']['testData'];
trainData = data['data']['trainData'];
validData = data['data']['validData'];

numdims = (#trainData)[1];
D = numdims - 1;
M = math.floor((#trainData)[2] / N);

train_input = torch.reshape(trainData[{ {1,D},{1,N*M} }],D,N,M);
train_target = torch.reshape(trainData[{ {D + 1},{1,N * M} }], 1, N, M);
valid_input = validData[{ {1,D},{} }];
valid_target = validData[{ {D + 1},{} }];
test_input = testData[{ {1,D},{} }];
test_target = testData[{ {D + 1}, {} }];


return train_input, train_target, valid_input, valid_target, test_input, test_target, vocab, vocab_size, vocab_ByIndex

end