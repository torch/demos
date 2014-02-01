-- testModel.lua
-- ARGS;
-- fittedModel : object responding to method forward(input) --> probabiliites
-- data        : table such that the test data is in data.test
--               data.test.input  : 2D Tensor of test examples
--               data.test.target : 1D Tensor of expected targets values
-- RETURNS
-- errorRate   : number, fraction of test examples for which the predicted target
--               value was equal to the expected target value
function testModel(fitted_model, data, algo, optimParams, actuallyPrint)
   if actuallyPrint == nil then
      actuallyPrint = true
   end

   if actuallyPrint then
      print('')
      print('============================================================')
      print('Testing model ' .. algo)
      print('')
   end

   -- Now that the model is trained, one can test it by evaluating it
   -- on new samples.

   -- The model constructed and trained above computes the probabilities
   -- of each class given the input values.

   -- We want to compare our model's results with those from the text.
   -- The input variables have narrow ranges, so we just compare all possible
   -- input variables in the training data.

   -- Determine actual frequency of the each female-age pair in the 
   -- training data

   -- return index of largest value
   local function maxIndex(a,b,c)
      if a >=b and a >= c then return 1 
      elseif b >= a and b >= c then return 2
      else return 3 end
   end

   -- return predicted brand and probabilities of each brand
   -- for the model in the text

   -- The R code in the text computes the probabilities of choosing
   -- brands 2 and 3 relative to the probability of choosing brand 1:
   --   Prob(brand=2)/prob(brand=1) = exp(-11.77 + 0.52*female + 0.37*age)
   --   Prob(brand=3)/prob(brand=1) = exp(-22.72 + 0.47*female + 0.69*age)
   local function predictText(age, female)
      --   1: calculate the "logit's"
      --      The coefficients come from the text.
      --      If you download the R script and run it, you may see slightly
      --      different results.
      local logit1 = 0
      local logit2 = -11.774655 + 0.523814 * female + 0.368206 * age
      local logit3 = -22.721396 + 0.465941 * female + 0.685908 * age

      --   2: calculate the unnormalized probabilities
      local uprob1 = math.exp(logit1)
      local uprob2 = math.exp(logit2)
      local uprob3 = math.exp(logit3)

      --   3: normalize the probabilities
      local z = uprob1 + uprob2 + uprob3
      local prob1 = (1/z) * uprob1
      local prob2 = (1/z) * uprob2
      local prob3 = (1/z) * uprob3

      return maxIndex(prob1, prob2, prob3), prob1, prob2, prob3
   end

   -- return predicted brand and the probabilities of each brand
   -- for our model
   local function predictOur(age, female)

      -- set input to a 1D vector with the transformed age and brand values
      -- from the test data. Transform the test data values using the mean and
      -- standard deviation from the training data.
      local input = torch.Tensor(2)
      input[data.cAge] = standardize(torch.Tensor{age}, data.train.ageMean, data.train.ageStandardDeviation)
      input[data.cIsFemale] = standardize(torch.Tensor{female}, data.train.isFemaleMean, data.train.isFemaleStandardDeviation)

      local logProbs = fitted_model:forward(input)  
      local probs = torch.exp(logProbs)
      local prob1, prob2, prob3 = probs[1], probs[2], probs[3]
      return maxIndex(prob1, prob2, prob3), prob1, prob2, prob3
   end

   local function makeKey(age, brand, isFemale)
      -- return a string containing the values

      -- Note that returning a table will not work, because each
      -- table is unique.

      -- Because Lua interns the strings, a string with a given sequence
      -- of characters is stored only once.
      return string.format('%2d%1d%1f', age, brand, isFemale)
   end

   -- determine distributions of brands condition on age and female
   -- for the raw data
   local counts = {}   -- keys are strings containg the age, brand, isFemale
   for i = 1, data.raw.nSamples do
      local age = data.raw.age[i]
      local brand = data.raw.brand[i]
      local isFemale = data.raw.isFemale[i]
      local key = makeKey(age, brand, isFemale)
      counts[key] = (counts[key] or 0) + 1
   end


   -- return probability of each brand conditioned on age and female
   local function actualProbabilities(age, female)
      local function countOf(age, brand, female)
         return counts[makeKey(age, brand, female)] or 0
      end
      local count1 = countOf(age, 1, female)
      local count2 = countOf(age, 2, female)
      local count3 = countOf(age, 3, female)
      local sumCounts = count1 + count2 + count3
      if sumCounts == 0 then
         return 0, 0, 0
      else
         return count1/sumCounts, count2/sumCounts, count3/sumCounts
      end
   end

   if actuallyPrint then
      print(' ')
      print('summary of data')
      summarizeData(data)

      print(' ')
      print('optimization parameters')
      for k,v in pairs(optimParams) do
         if type(v) ~= 'userdata' then
            print(string.format('%20s %s', k, tostring(v)))
         end
      end

      -- print the headers 
      print(' ')
      lineFormat = '%-6s %-3s| %-17s | %-17s | %-17s | %-1s %-1s %-1s'
      print(string.format(lineFormat,
                          '', '', 'actual probs', 'text probs', 'our probs', 'best', '', ''))
                          choices = 'brnd1 brnd2 brnd3'
      print(string.format(lineFormat,
                         'female', 'age', 
                         choices, choices, choices, 
                         'a', 't', 'o'))
   end

   -- print each row in the table

   local function formatFemale(female)
      return string.format('%1d', female)
   end

   local function formatAge(age)
      return string.format('%2d', age)
   end

   local function formatProbs(p1, p2, p3)
      return string.format('%5.3f %5.3f %5.3f', p1, p2, p3)
   end

   local function indexString(p1, p2, p3)
      -- return index of highest probability or '-' if nearly all zeroes
      if p1 < 0.001 and p2 < 0.001 and p3 < 0.001 then
         return '-'
      else 
         return string.format('%1d', maxIndex(p1, p2, p3))
      end
   end

   -- print table rows and accumulate accuracy
   local nErrors = 0  -- count number of errors vs. text
   local nTests = 0
   for female = data.raw.isFemaleMin, data.raw.isFemaleMax do
      for age = data.raw.ageMin, data.raw.ageMax do
         -- calculate the actual probabilities in the training data
         local actual1, actual2, actual3 = actualProbabilities(age, female)
         -- calculate the prediction and probabilities using the model in the text
         local textBrand, textProb1, textProb2, textProb3 = 
            predictText(age, female)
         -- calculate the probabilities using the model we just trained
         local ourBrand, ourProb1, ourProb2, ourProb3 = 
            predictOur(age, female)
         if actuallyPrint then
            print( string.format(lineFormat,
                                 formatFemale(female), 
                                 formatAge(age),
                                 formatProbs(actual1, actual2, actual3),
                                 formatProbs(textProb1, textProb2, textProb3),
                                 formatProbs(ourProb1, ourProb2, ourProb3),
                                 indexString(actual1,actual2,actual3),
                                 indexString(textProb1,textProb2,textProb3),
                                 indexString(ourProb1,ourProb2,ourProb3)))
         end
         local textBest = indexString(textProb1, textProb2, textProb3)
         local ourBest = indexString(ourProb1, ourProb2, ourProb3)
         if textBest ~= ourBest then
            nErrors = nErrors + 1
         end
         nTests = nTests + 1
      end
   end

   if actuallyPrint then
      print('nError vs. text', nErrors, 'of', nTests)
   end

   local error_rate = nErrors / nTests
   return error_rate
end

