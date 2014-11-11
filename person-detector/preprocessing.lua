------------------------------------------------------------------------------
-- Preprocessing to apply to each dataset
------------------------------------------------------------------------------
-- Alfredo Canziani May 2013, E. Culurciello June 2014
------------------------------------------------------------------------------

print(sys.COLORS.red ..  '==> preprocessing data')

local channels = {'r','g','b'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print(sys.COLORS.red ..  '==> preprocessing data: global normalization:')
local mean = {}
local std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> verify statistics:')

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
   local trainMean = trainData.data[{ {},i }]:mean()
   local trainStd = trainData.data[{ {},i }]:std()

   local testMean = testData.data[{ {},i }]:mean()
   local testStd = testData.data[{ {},i }]:std()

   print('       training data, '..channel..'-channel, mean:               ' .. trainMean)
   print('       training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('       test data, '..channel..'-channel, mean:                   ' .. testMean)
   print('       test data, '..channel..'-channel, standard deviation:     ' .. testStd)
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> visualizing data:')

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if opt.visualize then
   -- Showing some training exaples
   local first128Samples = trainData.data[{ {1,128} }]
   image.display{image=first128Samples, nrow=16, legend='Some training examples'}
   -- Showing some testing exaples
   local first128Samples = testData.data[{ {1,128} }]
   image.display{image=first128Samples, nrow=16, legend='Some testing examples'}
end
