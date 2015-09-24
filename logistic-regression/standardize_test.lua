-- standardize_test.lua
-- unit test

require 'standardize'
require 'torch'

-- data is from Wikipedia at "standard definition"
local v = torch.Tensor{2,4,4,4,5,5,7,9}

local vStandardized, mean, stdv = standardize(v)

-- test against values that Wikipedia reported
assert(mean == 5)
assert(stdv == 2)

-- test shape
assert(vStandardized:nDimension() == 1)
assert(vStandardized:size(1) == v:size(1))

-- check each element
local expected = torch.Tensor{-1.5, -0.5, -0.5, -0.5, 0, 0, 1, 2}
for i = 1, expected:size(1) do
   assert(vStandardized[i] == expected[i])
end

-- check that we can transform a new value correctly
local x = torch.Tensor{0}
local xStandardized = standardize(x, mean, stdv)
assert(xStandardized[1] == ((0 - mean) / stdv))

print('ok standardize')

