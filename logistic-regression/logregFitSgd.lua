-- logregFitSgd.lua
-- fit a logistic regression model using stochastic gradient descent
-- ARGS
-- sgdParams        : table passed to optim.sgd
-- sampling         : string
--                    for now, must be 'sequential-1'
-- stoppingCritera  : table describing when to stop the iterations
--                    For now, only one field
--                    .max_epochs: number of epochs
-- inputs           : 2D Tensor, each row is a sample
-- targets          : 1D Tensor, each element is a target 
--                    target in {1, 2, ..., nClasses}
-- nClasses         : number of classes, an integer >= 2
-- RETURNS
-- fitted_model     : model with optimized parameters
-- fittingInfo      : table with these fields
--                    .reason_stopped : string describing why the iteration 
--                                      stopped
function logregFitSgd(sgdParams, sampling, stoppingCriteria, inputs, targets, nClasses)
   assert(sampling == 'sequential-1')

   local nSamples = inputs:size(1)
   local nFeatures = inputs:size(2)
   print(nSamples) print(nFeatures)
   local model, criterion = logregModelCriterion(nFeatures, nClasses)

   local x, dl_dx = model:getParameters()  -- view parameters inside the model
   local next_sample_index = 0

   -- return the loss and gradient at x_new for one sample
   local function objectiveFunction(x_new)  
      if x ~= x_new then
         x:copy(x_new)
      end

      -- select next training samples
      -- (a better approach is to select a random sample)
      next_sample_index = next_sample_index + 1
      if next_sample_index > nSamples then
         next_sample_index = 1
      end   

      local input = inputs[next_sample_index]
      local target = targets[next_sample_index]

      dl_dx:zero() -- reset the gradient in case we are used in a batch method

      local loss_x = criterion:forward(model:forward(input), target)
      model:backward(input, criterion:backward(model.output, target))
      
      return loss_x, dl_dx
   end

   -- solve the optimization problem using the objective function
   local epoch_number = 0
   local cumulative_loss = 0  
   local reason_stopped 
   repeat
      epoch_number = epoch_number + 1
      local cumulative_loss_in_epoch = 0
      for sample_number = 1, nSamples do
         local _, fs = optim.sgd(objectiveFunction, x, sgdParams)
         cumulative_loss_in_epoch = cumulative_loss_in_epoch + fs[1]
      end
      -- report average loss for the epoch
      local average_loss = cumulative_loss_in_epoch / nSamples
      print(string.format('logregFitSgd epoch %d of %d average loss %f', 
                          epoch_number, 
                          stoppingCriteria.max_epochs, 
                          average_loss))
   until epoch_number == stoppingCriteria.max_epochs

   -- return optimized model and and some details on the fitting procedure
   local fittingInfo = {reason_stopped = 'max_epochs'}
   return model, fittingInfo
end
