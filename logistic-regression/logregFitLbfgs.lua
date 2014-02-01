-- logregFitLbfgs.lua
-- fit a logistic regression model using L-BFGS
-- ARGS
-- lbfgsParams      : table passed to optim.lbfgs
-- samples          : string describing how sample the inputs and targets
--                    for now, must be 'entire-batch'
-- stoppingCritera  : table describing when to stop the iterations
--                    For now, the stopping criteria are specified entirely
--                    in the lbfgsParams, so this table must have no fields
-- inputs           : 2D Tensor, each row is a sample
-- targets          : 1D Tensor, each element is a target 
--                    target in {1, 2, ..., nClasses}
-- nClasses         : number of classes, an integer >= 2
-- RETURNS
-- fitted_model     : model with optimized parameters
-- fitting_info     : table with these fields
--                    .reason_stopped : string describing why the iteration 
--                                      stopped
function logregFitLbfgs(lbfgsParams, samples, stoppingCriteria, inputs, targets, nClasses)
   assert(samples == 'entire-batch')
   assert(stoppingCriteria ~= nil)
   assert(targets:nDimension() == 1)
   
   -- make sure no fields are provided in the stoppingCriteria
   for key, value in pairs(stoppingCriteria) do
      assert(false, 'stoppingCriteria not a table with no fields')
   end
    
   local nSamples = inputs:size(1)
   local nFeatures = inputs:size(2)
   local model, criterion = logregModelCriterion(nFeatures, nClasses)

   local x, dl_dx = model:getParameters()  -- view parameters inside the model

   -- return the loss and gradient at x_new for all samples (the entire batch)
   local function objectiveFunction(x_new)  
      if x ~= x_new then
         x:copy(x_new)
      end

      dl_dx:zero() -- reset the gradient 
      local loss_on_all_samples = 0
      for next_sample_index = 1, nSamples do
         local input = inputs[next_sample_index]
         local target = targets[next_sample_index]

         local loss_on_one_sample = criterion:forward(model:forward(input), target)
         model:backward(input, criterion:backward(model.output, target))

         loss_on_all_samples = loss_on_all_samples + loss_on_one_sample 
      end

      -- normalize for batch size 
      -- now loss and gradients for L-BFGS and SGD are comparable
      local average_loss = loss_on_all_samples / nSamples
      dl_dx = dl_dx:div(nSamples)

      return average_loss, dl_dx
   end

   -- solve the optimization problem using the objective function
   -- NOTE: stoppingCriteria are not used, as optim.lbfgs does
   -- the required iterations.
   local _, fs = optim.lbfgs(objectiveFunction, x, lbfgsParams)
   print('history of L-BFGS evaluations:')
   print(fs)

   -- return optimized model and and some details on the fitting procedure
   local fitting_info = {reason_stopped = 'lbfgsParams', losses = fs}
   return model, fitting_info
end

