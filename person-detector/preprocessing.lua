------------------------------------------------------------------------------
-- Preprocessing to apply to each dataset
------------------------------------------------------------------------------
-- Alfredo Canziani May 2013
------------------------------------------------------------------------------

print '==> preprocessing data'

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes). Types can be easily converted in Torch, 
-- in general by doing: dst = src:type('torch.TypeTensor'), 
-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
-- for simplicity (float(),double(),cuda(),...):

trainData.data = trainData.data:float()
testData.data = testData.data:float()

-- We now preprocess the data. Preprocessing is crucial
-- when applying pretty much any kind of machine learning algorithm.

-- For natural images, we use several intuitive tricks:
--   + images are mapped into YUV space, to separate luminance information
--     from color information
--   + the luminance channel (Y) is locally normalized, using a contrastive
--     normalization operator: for each neighborhood, defined by a Gaussian
--     kernel, the mean is suppressed, and the standard deviation is normalized
--     to one.
--   + color channels are normalized globally, across the entire dataset;
--     as a result, each color component has 0-mean and 1-norm across the dataset.

-- Convert all images to YUV
print '==> preprocessing data: colorspace RGB -> YUV:'
for i = 1,trSize do
   trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
for i = 1,teSize do
   testData.data[i] = image.rgb2yuv(testData.data[i])
end

-- Name channels for convenience
local channels = {'y','u','v'}
--channels = {'r','g','b'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print '==> preprocessing data: global normalization:'
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

-- Local normalization
-- (note: the global normalization is useless, if this local normalization
-- is applied on all channels... the global normalization code is kept just
-- for the tutorial's purpose)
print '==> preprocessing data: local contrast normalization:'

-- Define the normalization neighborhood:
local neighborhood = image.gaussian1D(7)

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1e-3):float()

-- Normalize all channels locally:
for c=1,1 do-- in ipairs(channels) do
   print '       Normalising the training dataset'
   for i = 1,trSize do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
      xlua.progress(i,trSize)
   end
   print '       Normalising the testing dataset'
   for i = 1,teSize do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
      xlua.progress(i,teSize)
   end
end

----------------------------------------------------------------------
print '==> verify statistics:'

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
print '==> visualizing data:'

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if opt.visualize then
   -- Showing some training exaples
   local first128Samples_y = trainData.data[{ {1,128},1 }]
   local first128Samples_u = trainData.data[{ {1,128},2 }]
   local first128Samples_v = trainData.data[{ {1,128},3 }]
   image.display{image=first128Samples_y, nrow=16, legend='Some training examples: ' ..channels[1].. ' channel'}
   image.display{image=first128Samples_u, nrow=16, legend='Some training examples: ' ..channels[2].. ' channel'}
   image.display{image=first128Samples_v, nrow=16, legend='Some training examples: ' ..channels[3].. ' channel'}

   -- Showing some testing exaples
   local first128Samples_y = testData.data[{ {1,128},1 }]
   local first128Samples_u = testData.data[{ {1,128},2 }]
   local first128Samples_v = testData.data[{ {1,128},3 }]
   image.display{image=first128Samples_y, nrow=16, legend='Some testing examples: ' ..channels[1].. ' channel'}
   image.display{image=first128Samples_u, nrow=16, legend='Some testing examples: ' ..channels[2].. ' channel'}
   image.display{image=first128Samples_v, nrow=16, legend='Some testing examples: ' ..channels[3].. ' channel'}
end
