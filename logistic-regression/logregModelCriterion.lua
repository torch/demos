-- logregModelCriterion.lua

require 'nn'

-- return a model and criterion appropriate for non-regularized logistic regression
function logregModelCriterion(nFeatures, nClasses)
   local model = nn.Sequential()
   model:add(nn.Linear(nFeatures, nClasses))  
   model:add(nn.LogSoftMax())

   model:reset()  -- set internal parameter to random values

   local criterion = nn.ClassNLLCriterion()

   return model, criterion
end
