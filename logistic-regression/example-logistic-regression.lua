-- example-logistic-regression.lua
-- logistic regression and multinomial logistic regression

----------------------------------------------------------------------
-- 1. Create the training data

-- The data come from a tutorial on using R from UCLA, which can be found at
-- http://www.ats.ucla.edu/stat/r/dae/mlogit.htm

-- The model is one of brand preference, where there are 3 brands and 2
-- explanatory variables. The variables are coded this way:
--  brand: 1, 2 or 3
--  female: 1 if the person is a female, 0 if a male
--  age: a positive integer

-- The data are stored in a csv file 'example-logistic-regression.csv'
-- and read through the class Csv

require 'csv'
require 'nn'
require 'optim'

-- Build a table with one row for each observation
-- Each row will contain {brand, female, age}
rows = {}

-- The data are in a comma separated values (CSV) file. The first record
-- contains field names and subsequent records contain data. The fields and
-- their formats are:
-- - num: observation number; an integer surrounded by double quote chars
-- - brand: brand number; 1, 2, or 3 surrounded by double quote chars
-- - female: indicator for is-female: 1 if female, 0 otherwise; no quote chars
-- - age: age of the person; no quote characters

-- Reading CSV files can be tricky. This code uses the Csv class for this.

-- establish a scope to keep the global name space clean
do 
   local csv = Csv('example-logistic-regression.csv', 'r') -- r means read mode
   -- read and check the header
   local header = csv:read()  -- header is an array of strings
   assert(header[1] == 'num', 'num not found where expected')
   assert(header[2] == 'brand', 'brand not found where expected')
   assert(header[3] == 'female', 'female not found where expected')
   assert(header[4] == 'age', 'age not found where expected')
   -- read and save each data line
   while true do
      local dataLine = csv:read()
      -- dataLine is null if we have reached the end of file
      if not dataLine then break end
      -- the values are strings, so convert them to numbers by adding 0
      rows[#rows+1] = {dataLine[2] + 0,
		       dataLine[3] + 0,
		       dataLine[4] + 0}
   end
   csv:close()
end

-- print the first few rows
print("brand female age")
for i=1,10 do
   print(string.format('%5d %6d %3d', rows[i][1], rows[i][2], rows[i][3]))
end

-- Convert the rows table into a 2D Torch Tensor. The tensor form has the
-- advantage that it stores its elements continguously (which may lead to
-- better performance) and a tensor allows one to select columns and rows
-- easily.

data = torch.Tensor{rows}  -- CLEMENT: this doesn't work, hence the loop below
data = torch.Tensor(#rows,3)
for i=1,#rows do
   for j=1,3 do
      data[i][j] = rows[i][j]
   end
end

brands  = data[{ {}, {1}}]  -- the entire first column
females = data[{ {}, {2}}]  -- the entire second column
ages    = data[{ {}, {3}}]  -- the entire third column

-- To implement the model, we need to know how many categories there are.

numberOfBrands = 0
do
   seen = {}
   for i = 1,brands:size(1) do
      -- brands[i] yields a 1D tensor
      local nextBrand = brands[i][1]  -- extract the integer value
      if not seen[nextBrand] then 
	 seen[nextBrand] = true
      end
   end
   numberOfBrands = #seen
end

-- summarize the data
function summarizeData()
   function p(name,value) 
      print(string.format('%20s %f', name, value) )
   end
   p('number of brands', numberOfBrands)
   p('min brand', torch.min(brands))
   p('max brand', torch.max(brands))
   
   p('min female', torch.min(females))
   p('max female', torch.max(females))
   
   p('min age', torch.min(ages))
   p('max age', torch.max(ages))
end

summarizeData()

-- check that the number of brands is exactly equal to the max brand value
if torch.max(brands) ~= numberOfBrands then
   error('number of brands is off')
end


----------------------------------------------------------------------
-- 2. Define the model (predictor)

-- The model is a multinomial logistic regression. 

-- It will consist of two layers that operate sequentially:
--  - 1: a linear model
--  - 2: a soft max layer

-- The linear model supposes that the un-normalized probability of choosing
-- a specific brand is proportional to the product of unknown weights and 
-- the observed variables plus a bias:
--   Prob(brand = b) = bias + weight1 * female * weight2 * age
-- There are two inputs (female and age) and three outputs (one for each
-- value that brand can take on)
linLayer = nn.Linear(2,3)

-- The soft max layer takes the 3 outputs from the linear layer and
-- transforms them to lie in the range (0,1) and to sum to 1. Thus, unlike
-- some text books in which the probabilities are un-normalized, the output
-- of the soft max layer will be normalized probabilities. 

-- The log soft max layer takes the log of these 3 outputs. This is done
-- because we want to feed the log values into the ClassNLLCriterion
-- described below.

softMaxLayer = nn.LogSoftMax()  -- the input and output are a single tensor

-- We need to put the layers into a sequential container.

model = nn.Sequential()
model:add(linLayer)
model:add(softMaxLayer)

----------------------------------------------------------------------
-- 3. Define a loss function, to be minimized.

-- In that example, we minimize the Mean Square Error (MSE) between
-- the predictions of our linear model and the groundtruth available
-- in the dataset.

-- Torch provides many common criterions to train neural networks.

-- The ClassNLLCriterion expects to be fed the log probabilities in a
-- tensor. Hence, the use of the LogSoftMax layer in the model instead
-- of SoftMax.

criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------
-- 4. Train the model

-- To minimize the loss defined above, using the linear model defined
-- in 'model', we follow a stochastic gradient descent procedure (SGD).

-- SGD is a good optimization algorithm when the amount of training data
-- is large, and estimating the gradient of the loss function over the 
-- entire training set is too costly.

-- Given an arbitrarily complex model, we can retrieve its trainable
-- parameters, and the gradients of our loss function wrt these 
-- parameters by doing so:

x, dl_dx = model:getParameters()

-- The above statement does not create a copy of the parameters in the 
-- model! Instead it create in x and dl_dx a view of the model's weights
-- and derivative wrt the weights. The view is implemented so that when
-- the weights and their derivatives changes, so do the x and dl_dx. The
-- implementation is efficient in that the underlying storage is shared.

-- A note on terminology: In the machine learning literature, the parameters
-- that one seeks to learn are often called weights and denoted with a W.
-- However, in the optimization literature, the parameter one seeks to 
-- optimize is often called x. Hence the use of x and dl_dx above.

-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. x is the vector of trainable weights,
-- which, in this example, are all the weights of the linear matrix of
-- our mode, plus one bias.

feval = function(x_new)
   -- set x to x_new, if differnt
   -- (in this simple example, x_new will typically always point to x,
   -- so the copy is really useless)
   if x ~= x_new then
      x:copy(x_new)
   end

   -- select a new training sample
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#data)[1] then _nidx_ = 1 end

   local sample = data[_nidx_]
   local target = sample[{ 1 }]      -- this funny looking syntax allows
   local inputs = sample[{ {2,3} }]    -- slicing of arrays.

   -- reset gradients (gradients are always accumulated, to accomodate 
   -- batch methods)
   dl_dx:zero()

   -- evaluate the loss function and its derivative wrt x, for that sample
   local loss_x = criterion:forward(model:forward(inputs), target)
   model:backward(inputs, criterion:backward(model.output, target))
   
   -- report on the parameters, which should be converging
   if false then
      local p = model:getParameters()
      print(string.format(
	    "parameters %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f", 
	    p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9])
	   )
   end

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

-- Given the function above, we can now easily train the model using SGD.
-- For that, we need to define four key parameters:
--   + a learning rate: the size of the step taken at each stochastic 
--     estimate of the gradient
--   + a weight decay, to regularize the solution (L2 regularization)
--   + a momentum term, to average steps over time
--   + a learning rate decay, to let the algorithm converge more precisely

sgd_params = {
   learningRate = 1e-1,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

-- We're now good to go... all we have left to do is run over the dataset
-- for a certain number of iterations, and perform a stochastic update 
-- at each iteration. The number of iterations is found empirically here,
-- but should typically be determinined using cross-correlation.

-- CLEMENT: Can we explain cross-correlation in this context?

epochs = 1e4  -- number of times to cycle over our training data

for i = 1,epochs do

   -- this variable is used to estimate the average loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   for i = 1,(#data)[1] do

      -- optim contains several optimization algorithms. 
      -- All of these algorithms assume the same parameters:
      --   + a closure that computes the loss, and its gradient wrt to x, 
      --     given a point x
      --   + a point x
      --   + some parameters, which are algorithm-specific

      _,fs = optim.sgd(feval,x,sgd_params)

      -- CLEMENT: Can we also implement BFGS (which converges in the R
      -- model in 7 iterations)?

      -- Functions in optim all return two things:
      --   + the new x, found by the optimization method (here SGD)
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.

      current_loss = current_loss + fs[1]
   end

   -- report average error on epoch
   current_loss = current_loss / (#data)[1]
   print('epoch = ' .. i .. 
	 ' of ' .. epochs .. 
	 ' current loss = ' .. current_loss)

end

----------------------------------------------------------------------
-- 5. Test the trained model.

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
function maxIndex(a,b,c)
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
function predictText(age, female)
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
function predictOur(age, female)
   local input = torch.Tensor(2)
   input[1] = female  -- must be in same order as when the model was trained!
   input[2] = age
   local logProbs = model:forward(input)  
   --print('predictOur', age, female, input)
   local probs = torch.exp(logProbs)
   --print('logProbs', logProbs)
   --print('probs', probs[1], probs[2], probs[3] )
   local prob1, prob2, prob3 = probs[1], probs[2], probs[3]
   return maxIndex(prob1, prob2, prob3), prob1, prob2, prob3
end
      
counts = {}

function makeKey(age, brand, female)
   -- return a string containing the values

   -- Note that returning a table will not work, because each
   -- table is unique.

   -- Because Lua interns the strings, a string with a given sequence
   -- of characters is stored only once.
   return string.format('%2d%1d%1f', age, brand, female)
end

for _,row in pairs(rows) do
   local brand = row[1]
   local female = row[2]
   local age = row[3]
   local key = makeKey (age, brand, female)
   counts[key] = (counts[key] or 0) + 1
end


-- return probability of each brand conditioned on age and female
function actualProbabilities(age, female)
   function countOf(age, brand, female)
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


print(' ')
print('summary of data')
summarizeData()

print(' ')
print('training variables')
for k,v in pairs(sgd_params) do
   print(string.format('%20s %f', k, v))
end
print(string.format('%20s %f', 'epochs', epochs))

print(' ')
print('current loss', current_loss)

-- print the headers 
print(' ')
lineFormat = '%-6s %-3s| %-17s | %-17s | %-17s | %-1s %-1s %-1s'
print(
   string.format(lineFormat,
		 '', '', 
		 'actual probs', 'text probs', 'our probs', 
		 'best', '', ''))
choices = 'brnd1 brnd2 brnd3'
print(string.format(lineFormat,
		    'female', 'age', 
		    choices, choices, choices, 
		    'a', 't', 'o'))

-- print each row in the table

function formatFemale(female)
   return string.format('%1d', female)
end

function formatAge(age)
   return string.format('%2d', age)
end

function formatProbs(p1, p2, p3)
   return string.format('%5.3f %5.3f %5.3f', p1, p2, p3)
end

function indexString(p1, p2, p3)
   -- return index of highest probability or '-' if nearly all zeroes
   if p1 < 0.001 and p2 < 0.001 and p3 < 0.001 then
      return '-'
   else 
      return string.format('%1d', maxIndex(p1, p2, p3))
   end
end

-- print table rows and accumulate accuracy
for female = 0,1 do
   for age = torch.min(ages),torch.max(ages) do
      -- calculate the actual probabilities in the training data
      local actual1, actual2, actual3 = actualProbabilities(age, female)
      -- calculate the prediction and probabilities using the model in the text
      local textBrand, textProb1, textProb2, textProb3 = 
	 predictText(age, female)
      -- calculate the probabilities using the model we just trained
      --print("main", age, female)
      local ourBrand, ourProb1, ourProb2, ourProb3 = 
	 predictOur(age, female)
      print(
	 string.format(lineFormat,
		       formatFemale(female), 
		       formatAge(age),
		       formatProbs(actual1, actual2, actual3),
		       formatProbs(textProb1, textProb2, textProb3),
		       formatProbs(ourProb1, ourProb2, ourProb3),
		       indexString(actual1,actual2,actual3),
		       indexString(textProb1,textProb2,textProb3),
		       indexString(ourProb1,ourProb2,ourProb3))
	   )
   end
end

----------------------------------------------------------------------
-- 6. Assess accuracy on the training data.

-- We can compare our model with the UCLA model.

-- The table generated just above shows the brand that each model would
-- have predicted if it were to predict the brand that it estimated had
-- the highest probability. It also shows the distribution of actual
-- brand choices. All these results are conditioned on a specific value
-- for the female and age variables. The predictions are in the last
-- 3 columns of the table, where 'a' means the value predicted from the 
-- actual data, 't' means the value predicted from the text, and 'o' means
-- the value predicted from our model.

-- One can see that when our estimate differs from that implied by the 
-- actual data, our predicted probabilities are very different from the
-- actual probabilities. Also on (female,age) pairings where our model
-- differs from the actual, the text model is often correct.

-- Both the text model and our model use the same cost function and the
-- same parameters. The text model finds the parameters by using BFGS 
-- whereas our model uses stochastic gradient descent.

print(' ')
print('Accuracy on training data')

countOfDifferences = {}
function makeKey(female, age, actualBrand, textPrediction, ourPrediction)
   return string.format('%6d %3d %6d %3d %4d',
			female, age, 
			actualBrand, textPrediction, ourPrediction)
end

for _,row in pairs(rows) do
   local actualBrand = row[1]
   local female = row[2]
   local age = row[3]
   local textPrediction = predictText(age, female)
   local ourPrediction = predictOur(age, female)
   if actualBrand == textPrediction then
      numberCorrect = (numberCorrect or 0) + 1
      countTextCorrect = (countTextCorrect or 0) + 1
   end
   if actualBrand == ourPrediction then
      numberCorrect = (numberCorrect or 0) + 1
      countOurCorrect = (countOurCorrect or 0) + 1
   end
   if (numberCorrect ~= 2) and (textPrediction ~= ourPrediction) then
      local key = makeKey(female,age,actualBrand,textPrediction,ourPrediction)
      countOfDifferences[key] = (countOfDifferences[key] or 0) + 1
   end
   countObservations = (countObservations or 0) + 1
end

print()
print("Differences between text's predictions and our predictions")
print("When only one prediction is correct")
print('female age actual text our occurs')
accuracyLine = '%-26s %6d'
table.sort(countOfDifferences)
for k,v in pairs(countOfDifferences) do
   print(string.format('%-26s %6d', k, v))
end

print()
function printCorrectFraction(name, numCorrect)
   print(
      string.format(
	 'Fraction of %s predictions that were correct on training set = %f',
	 name, numCorrect / countObservations))
end

printCorrectFraction('text', countTextCorrect)
printCorrectFraction('our', countOurCorrect)

print(' ')
print('Frequency of each brand in training data')

function countBrand(brandNumber)
   local count = 0
   for _,row in pairs(rows) do
      local actualBrand = row[1]
      if actualBrand == brandNumber then
	 count = count + 1
      end
   end
   return count
end

function printFrequency(brandNumber)
   print(string.format('Brand %d occurs %f',
		       brandNumber, countBrand(brandNumber) / #rows))
end

printFrequency(1)
printFrequency(2)
printFrequency(3)




-- From reviewing the output for 10,000 epochs, one can see that
-- when usually when our model differs from the most likely training
-- sample (the "a" column), the model in the text is accurate.

-- Your results will most likely differ.
-- CLEMENT: WHERE IS THE RANDOMNESS IS THIS MODEL? 

-- Note that our predictions are less accurate than simply predicting
-- brand 2 no matter what age and female are!

--[[ sample output 10,000 epochs
epoch = 10000 of 10000 current loss = 0.98691723284723
 
summary of data
    number of brands 3.000000
           min brand 1.000000
           max brand 3.000000
          min female 0.000000
          max female 1.000000
             min age 24.000000
             max age 38.000000
 
training variables
         evalCounter 7350000.000000
         weightDecay 0.000000
        learningRate 0.100000
            momentum 0.000000
   learningRateDecay 0.000100
              epochs 10000.000000
 
current loss    0.98691723284723
 
          | actual probs      | text probs        | our probs         | best    
female age| brnd1 brnd2 brnd3 | brnd1 brnd2 brnd3 | brnd1 brnd2 brnd3 | a t o
0      24 | 1.000 0.000 0.000 | 0.948 0.050 0.002 | 0.516 0.291 0.193 | 1 1 1
0      25 | 0.000 0.000 0.000 | 0.926 0.071 0.004 | 0.471 0.300 0.229 | - 1 1
0      26 | 1.000 0.000 0.000 | 0.894 0.099 0.007 | 0.424 0.307 0.269 | 1 1 1
0      27 | 1.000 0.000 0.000 | 0.851 0.136 0.013 | 0.378 0.310 0.312 | 1 1 1
0      28 | 0.667 0.000 0.333 | 0.793 0.183 0.024 | 0.333 0.310 0.358 | 1 1 3
0      29 | 0.875 0.125 0.000 | 0.718 0.240 0.042 | 0.289 0.305 0.405 | 1 1 3
0      30 | 0.500 0.333 0.167 | 0.625 0.302 0.073 | 0.249 0.298 0.454 | 1 1 3
0      31 | 0.588 0.294 0.118 | 0.518 0.361 0.121 | 0.211 0.287 0.502 | 1 1 3
0      32 | 0.407 0.432 0.161 | 0.405 0.408 0.187 | 0.177 0.273 0.549 | 2 2 3
0      33 | 0.211 0.474 0.316 | 0.296 0.432 0.272 | 0.148 0.258 0.595 | 2 2 3
0      34 | 0.167 0.542 0.292 | 0.203 0.427 0.370 | 0.122 0.240 0.638 | 2 2 3
0      35 | 0.000 0.182 0.818 | 0.131 0.397 0.472 | 0.099 0.223 0.678 | 3 3 3
0      36 | 0.133 0.333 0.533 | 0.080 0.350 0.571 | 0.080 0.204 0.715 | 3 3 3
0      37 | 0.200 0.200 0.600 | 0.046 0.294 0.660 | 0.065 0.186 0.749 | 3 3 3
0      38 | 0.000 0.278 0.722 | 0.026 0.239 0.735 | 0.052 0.169 0.780 | 3 3 3
1      24 | 0.000 0.000 0.000 | 0.915 0.082 0.003 | 0.394 0.381 0.225 | - 1 1
1      25 | 0.000 0.000 0.000 | 0.881 0.114 0.005 | 0.352 0.387 0.262 | - 1 2
1      26 | 0.000 0.000 0.000 | 0.834 0.156 0.010 | 0.311 0.388 0.301 | - 1 2
1      27 | 0.800 0.000 0.200 | 0.773 0.209 0.018 | 0.272 0.385 0.343 | 1 1 2
1      28 | 0.667 0.222 0.111 | 0.696 0.271 0.033 | 0.236 0.377 0.387 | 1 1 3
1      29 | 0.636 0.364 0.000 | 0.603 0.340 0.057 | 0.202 0.367 0.432 | 1 1 3
1      30 | 0.588 0.235 0.176 | 0.500 0.407 0.093 | 0.171 0.352 0.477 | 1 1 3
1      31 | 0.391 0.391 0.217 | 0.392 0.462 0.145 | 0.144 0.335 0.521 | 1 2 3
1      32 | 0.288 0.544 0.167 | 0.291 0.495 0.214 | 0.119 0.316 0.564 | 2 2 3
1      33 | 0.083 0.639 0.278 | 0.203 0.500 0.297 | 0.098 0.296 0.606 | 2 2 3
1      34 | 0.150 0.400 0.450 | 0.134 0.477 0.389 | 0.081 0.274 0.645 | 3 2 3
1      35 | 0.042 0.250 0.708 | 0.084 0.432 0.484 | 0.065 0.252 0.682 | 3 3 3
1      36 | 0.109 0.345 0.545 | 0.050 0.374 0.576 | 0.053 0.231 0.717 | 3 3 3
1      37 | 0.000 0.294 0.706 | 0.029 0.311 0.660 | 0.042 0.210 0.748 | 3 3 3
1      38 | 0.071 0.214 0.714 | 0.016 0.252 0.732 | 0.034 0.190 0.777 | 3 3 3
 
Accuracy on training data

Differences between text's predictions and our predictions
When only one prediction is correct
female age actual text our occurs
     0  31      1   1    3     10
     1  27      1   1    2      4
     1  29      1   1    3      7
     1  28      3   1    3      1
     1  31      3   2    3      5
     0  30      3   1    3      1
     0  32      3   2    3     19
     0  34      1   2    3      4
     0  34      2   2    3     13
     0  31      3   1    3      2
     1  27      3   1    2      1
     0  29      1   1    3      7
     0  34      3   2    3      7
     0  32      2   2    3     51
     0  30      2   1    3      2
     1  33      2   2    3     23
     1  30      3   1    3      3
     1  34      3   2    3     18
     1  32      2   2    3    117
     1  31      2   2    3      9
     1  34      1   2    3      6
     1  31      1   2    3      9
     0  28      1   1    3      4
     1  32      1   2    3     62
     1  33      1   2    3      3
     0  33      1   2    3      4
     1  28      1   1    3      6
     0  33      3   2    3      6
     1  34      2   2    3     16
     1  33      3   2    3     10
     1  29      2   1    3      4
     0  28      3   1    3      2
     1  28      2   1    3      2
     1  32      3   2    3     36
     0  33      2   2    3      9
     1  30      2   1    3      4
     0  32      1   2    3     48
     0  31      2   1    3      5
     0  30      1   1    3      3
     1  30      1   1    3     10
     0  29      2   1    3      1

Fraction of text predictions that were correct on training set = 0.552381
Fraction of our predictions that were correct on training set = 0.369126
 
Frequency of each brand in training data
Brand 1 occurs 0.281633
Brand 2 occurs 0.417687
Brand 3 occurs 0.300680
   --]]

-- With 100,000 epochs, our predictions are as accurate as always
-- predicting brand 2. We predict the same value as the text for each 
-- combination of female and age.

-- Studying the results on the training data, one can see that our
-- model does wors on it than the text's model, because of the differences
-- in estimating individaul (female,age) points and the distribution of
-- data in the training set.

--[[ sample output 100,000 epochs
epoch = 100000 of 100000 current loss = 0.96319934821442
 
summary of data
    number of brands 3.000000
           min brand 1.000000
           max brand 3.000000
          min female 0.000000
          max female 1.000000
             min age 24.000000
             max age 38.000000
 
training variables
         evalCounter 73500000.000000
         weightDecay 0.000000
        learningRate 0.100000
            momentum 0.000000
   learningRateDecay 0.000100
              epochs 100000.000000
 
current loss    0.96319934821442
 
          | actual probs      | text probs        | our probs         | best    
female age| brnd1 brnd2 brnd3 | brnd1 brnd2 brnd3 | brnd1 brnd2 brnd3 | a t o
0      24 | 1.000 0.000 0.000 | 0.948 0.050 0.002 | 0.884 0.107 0.009 | 1 1 1
0      25 | 0.000 0.000 0.000 | 0.926 0.071 0.004 | 0.850 0.135 0.015 | - 1 1
0      26 | 1.000 0.000 0.000 | 0.894 0.099 0.007 | 0.808 0.168 0.024 | 1 1 1
0      27 | 1.000 0.000 0.000 | 0.851 0.136 0.013 | 0.756 0.206 0.037 | 1 1 1
0      28 | 0.667 0.000 0.333 | 0.793 0.183 0.024 | 0.694 0.248 0.058 | 1 1 1
0      29 | 0.875 0.125 0.000 | 0.718 0.240 0.042 | 0.622 0.291 0.088 | 1 1 1
0      30 | 0.500 0.333 0.167 | 0.625 0.302 0.073 | 0.540 0.331 0.129 | 1 1 1
0      31 | 0.588 0.294 0.118 | 0.518 0.361 0.121 | 0.453 0.364 0.182 | 1 1 1
0      32 | 0.407 0.432 0.161 | 0.405 0.408 0.187 | 0.366 0.385 0.249 | 2 2 2
0      33 | 0.211 0.474 0.316 | 0.296 0.432 0.272 | 0.284 0.391 0.325 | 2 2 2
0      34 | 0.167 0.542 0.292 | 0.203 0.427 0.370 | 0.211 0.381 0.409 | 2 2 3
0      35 | 0.000 0.182 0.818 | 0.131 0.397 0.472 | 0.151 0.356 0.493 | 3 3 3
0      36 | 0.133 0.333 0.533 | 0.080 0.350 0.571 | 0.104 0.322 0.574 | 3 3 3
0      37 | 0.200 0.200 0.600 | 0.046 0.294 0.660 | 0.069 0.282 0.649 | 3 3 3
0      38 | 0.000 0.278 0.722 | 0.026 0.239 0.735 | 0.045 0.241 0.714 | 3 3 3
1      24 | 0.000 0.000 0.000 | 0.915 0.082 0.003 | 0.821 0.166 0.013 | - 1 1
1      25 | 0.000 0.000 0.000 | 0.881 0.114 0.005 | 0.775 0.205 0.020 | - 1 1
1      26 | 0.000 0.000 0.000 | 0.834 0.156 0.010 | 0.719 0.250 0.031 | - 1 1
1      27 | 0.800 0.000 0.200 | 0.773 0.209 0.018 | 0.654 0.297 0.048 | 1 1 1
1      28 | 0.667 0.222 0.111 | 0.696 0.271 0.033 | 0.581 0.346 0.073 | 1 1 1
1      29 | 0.636 0.364 0.000 | 0.603 0.340 0.057 | 0.502 0.392 0.106 | 1 1 1
1      30 | 0.588 0.235 0.176 | 0.500 0.407 0.093 | 0.420 0.430 0.150 | 1 1 2
1      31 | 0.391 0.391 0.217 | 0.392 0.462 0.145 | 0.340 0.456 0.205 | 1 2 2
1      32 | 0.288 0.544 0.167 | 0.291 0.495 0.214 | 0.265 0.465 0.269 | 2 2 2
1      33 | 0.083 0.639 0.278 | 0.203 0.500 0.297 | 0.199 0.458 0.342 | 2 2 2
1      34 | 0.150 0.400 0.450 | 0.134 0.477 0.389 | 0.145 0.436 0.419 | 3 2 2
1      35 | 0.042 0.250 0.708 | 0.084 0.432 0.484 | 0.102 0.401 0.497 | 3 3 3
1      36 | 0.109 0.345 0.545 | 0.050 0.374 0.576 | 0.069 0.358 0.573 | 3 3 3
1      37 | 0.000 0.294 0.706 | 0.029 0.311 0.660 | 0.046 0.312 0.642 | 3 3 3
1      38 | 0.071 0.214 0.714 | 0.016 0.252 0.732 | 0.030 0.265 0.705 | 3 3 3
 
Accuracy on training data

Differences between text's predictions and our predictions
When only one prediction is correct
female age actual text our occurs
     0  34      3   2    3      7
     0  34      2   2    3     13
     1  30      3   1    2      3
     1  30      2   1    2      4
     1  30      1   1    2     10
     0  34      1   2    3      4

Fraction of text predictions that were correct on training set = 0.552381
Fraction of our predictions that were correct on training set = 0.374150
 
Frequency of each brand in training data
Brand 1 occurs 0.281633
Brand 2 occurs 0.417687
Brand 3 occurs 0.300680
   --]]

-- with 1,000,000 epochs, we have slighlty improved our accuracy relative
-- to always predicting brand 2 but are still not close to the accuracy
-- of the text's model. The good news is that on each combination of
-- female and age in the test set, we return the same value as the text's
-- method.

-- TODO: Either delete this run or re-run it to get the additional output.
--[[
epoch = 1000000 of 1000000 current loss = 0.9571160317243
 
summary of data
    number of brands 3.000000
           min brand 1.000000
           max brand 3.000000
          min female 0.000000
          max female 1.000000
             min age 24.000000
             max age 38.000000
 
training variables
         evalCounter 735000000.000000
         weightDecay 0.000000
        learningRate 0.100000
            momentum 0.000000
   learningRateDecay 0.000100
              epochs 1000000.000000
 
current loss    0.9571160317243
 
          | actual probs      | text probs        | our probs         | best    
female age| brnd1 brnd2 brnd3 | brnd1 brnd2 brnd3 | brnd1 brnd2 brnd3 | a t o
0      24 | 1.000 0.000 0.000 | 0.948 0.050 0.002 | 0.934 0.063 0.003 | 1 1 1
0      25 | 0.000 0.000 0.000 | 0.926 0.071 0.004 | 0.909 0.087 0.005 | - 1 1
0      26 | 1.000 0.000 0.000 | 0.894 0.099 0.007 | 0.874 0.117 0.009 | 1 1 1
0      27 | 1.000 0.000 0.000 | 0.851 0.136 0.013 | 0.829 0.155 0.016 | 1 1 1
0      28 | 0.667 0.000 0.333 | 0.793 0.183 0.024 | 0.770 0.202 0.028 | 1 1 1
0      29 | 0.875 0.125 0.000 | 0.718 0.240 0.042 | 0.695 0.256 0.048 | 1 1 1
0      30 | 0.500 0.333 0.167 | 0.625 0.302 0.073 | 0.606 0.314 0.080 | 1 1 1
0      31 | 0.588 0.294 0.118 | 0.518 0.361 0.121 | 0.505 0.367 0.128 | 1 1 1
0      32 | 0.407 0.432 0.161 | 0.405 0.408 0.187 | 0.400 0.407 0.193 | 2 2 2
0      33 | 0.211 0.474 0.316 | 0.296 0.432 0.272 | 0.298 0.426 0.275 | 2 2 2
0      34 | 0.167 0.542 0.292 | 0.203 0.427 0.370 | 0.210 0.421 0.370 | 2 2 2
0      35 | 0.000 0.182 0.818 | 0.131 0.397 0.472 | 0.139 0.392 0.469 | 3 3 3
0      36 | 0.133 0.333 0.533 | 0.080 0.350 0.571 | 0.088 0.347 0.565 | 3 3 3
0      37 | 0.200 0.200 0.600 | 0.046 0.294 0.660 | 0.053 0.294 0.653 | 3 3 3
0      38 | 0.000 0.278 0.722 | 0.026 0.239 0.735 | 0.031 0.241 0.728 | 3 3 3
1      24 | 0.000 0.000 0.000 | 0.915 0.082 0.003 | 0.894 0.102 0.004 | - 1 1
1      25 | 0.000 0.000 0.000 | 0.881 0.114 0.005 | 0.856 0.137 0.007 | - 1 1
1      26 | 0.000 0.000 0.000 | 0.834 0.156 0.010 | 0.807 0.181 0.013 | - 1 1
1      27 | 0.800 0.000 0.200 | 0.773 0.209 0.018 | 0.744 0.234 0.022 | 1 1 1
1      28 | 0.667 0.222 0.111 | 0.696 0.271 0.033 | 0.668 0.294 0.038 | 1 1 1
1      29 | 0.636 0.364 0.000 | 0.603 0.340 0.057 | 0.579 0.358 0.063 | 1 1 1
1      30 | 0.588 0.235 0.176 | 0.500 0.407 0.093 | 0.482 0.418 0.100 | 1 1 1
1      31 | 0.391 0.391 0.217 | 0.392 0.462 0.145 | 0.383 0.466 0.152 | 1 2 2
1      32 | 0.288 0.544 0.167 | 0.291 0.495 0.214 | 0.289 0.493 0.219 | 2 2 2
1      33 | 0.083 0.639 0.278 | 0.203 0.500 0.297 | 0.207 0.495 0.299 | 2 2 2
1      34 | 0.150 0.400 0.450 | 0.134 0.477 0.389 | 0.140 0.472 0.388 | 3 2 2
1      35 | 0.042 0.250 0.708 | 0.084 0.432 0.484 | 0.091 0.429 0.480 | 3 3 3
1      36 | 0.109 0.345 0.545 | 0.050 0.374 0.576 | 0.057 0.374 0.570 | 3 3 3
1      37 | 0.000 0.294 0.706 | 0.029 0.311 0.660 | 0.034 0.314 0.652 | 3 3 3
1      38 | 0.071 0.214 0.714 | 0.016 0.252 0.732 | 0.020 0.256 0.724 | 3 3 3
 
Accuracy on training data
Fraction of text predictions that were correct = 0.552381
Fraction of our predictions that were correct = 0.452041
 
Frequency of each brand in training data
Brand 1 occurs 0.281633
Brand 2 occurs 0.417687
Brand 3 occurs 0.300680
--]]




