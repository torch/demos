-- printTable_test.lua
-- unit test

require 'printTable'
require 'torch'

data = {}
data.one = 1
data.string = 'abc'
data.table = {}
data.table.two = 2
data.tensor1Da = torch.Tensor{1}
data.tensor1Db = torch.Tensor{1,2}
data.tensor2D  = torch.rand(3, 4)

data[23] = 'twenty three'
data[function(a,b) end] = 'function of two args'
data['a string as key'] = 'a string as value'
data[torch.Tensor{1,2}] = 'key is tensor with elements 1 and 2'

printTable('data', data)

print('ok printTable')
