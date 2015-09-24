-- logregFit.lua

require 'logregFitLbfgs'
require 'logregFitSgd'

-- fit a logistic regression model to training data using a specified
-- algorithm and sample selection strategy.
-- ARGS
-- algorithm        : string in {'SGD', 'L-BFGS'}
-- algoParams       : table appropriate to the algorithm
-- sampling         : object describing how to sample the inputs and targets
-- stoppingCritera  : table describing when to stop the iterations
--                    For now, only one field
--                    .max_epochs : number of epochs
-- inputs           : 2D Tensor, each row is a training sample
-- targets          : 1D Tensor, each element is a training target 
--                    target in {1, 2, ..., nClasses}
-- nClasses         : number of classes, an integer >= 2
-- RETURNS
-- fitted_model     : model with optimized parameters
-- fitting_info     : table that depends on the algorithm
function logregFit(algorithm, 
                   algoParams, 
                   sampling,
                   stoppingCriteria, 
                   inputs, 
                   targets, 
                   nClasses)
   -- type check the args
   assert(type(algorithm) == 'string')
   assert(type(algoParams) == 'table')
   assert(type(sampling) == 'string')
   assert(type(stoppingCriteria) == 'table')
   assert(inputs:nDimension() == 2)
   assert(targets:nDimension() == 1)
   assert(type(nClasses) == 'number')
   
   -- check that inputs and targets have the same number of samples
   assert(inputs:size(1) == targets:size(1))

   if algorithm == 'SGD' then
      -- Use stochastic gradient descent on each sample in order
      return logregFitSgd(algoParams, 
                          sampling,
                          stoppingCriteria, 
                          inputs, 
                          targets, 
                          nClasses)
   elseif algorithm == 'L-BFGS' then
      -- Use L-BFGS on the full set of training data
      return logregFitLbfgs(algoParams, 
                            sampling,
                            stoppingCriteria, 
                            inputs, 
                            targets, 
                            nClasses)
   else
      error('unknown algorithm: ' .. tostring(algorithm))
   end
end
