-- standardize.lua
-- center data around zero by subtracting the mean and dividing by the standard deviation
function standardize(vector, mean, standard_deviation)
   local nObservations = vector:size(1)

   if mean == nil then
      mean = torch.sum(vector) / nObservations
   end

   if standard_deviation == nil then
      local differences = vector - mean
      local squared_differences = torch.cmul(differences, differences)
      local variance = torch.sum(squared_differences) / nObservations
      standard_deviation = math.sqrt(variance)
   end

   local standardized = torch.div(vector - mean, standard_deviation)
   return standardized, mean, standard_deviation
end
