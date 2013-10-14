----------------------------------------------------------------------
-- This script demonstrates how to load the Face Detector 
-- training data, and pre-process it to facilitate learning.
--
-- It's a good idea to run this script with the interactive mode:
-- $ torch -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.
--
-- Clement Farabet, Eugenio Culurciello
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides a normalization operator

local opt = opt or {
   visualize = true,
   size = 'small'
}

----------------------------------------------------------------------
print '==> downloading dataset'

-- Here we download dataset files. 

-- Note: files were converted from their original Matlab format
-- to Torch's internal format using the mattorch package. The
-- mattorch package allows 1-to-1 conversion between Torch and Matlab
-- files.

local www = 'http://data.neuflow.org/data/'

local train_file = '../datasets/faces_cut_yuv_32x32'

-- file from: http://data.neuflow.org/data/faces_cut_yuv_32x32.tar.gz
if not sys.dirp(opt.dataset) then
   local path = sys.dirname(opt.dataset)
   local tar = sys.basename(opt.www)
   os.execute('mkdir -p ' .. path .. '; '..
              'cd ' .. path .. '; '..
              'wget ' .. opt.www .. '; '..
              'tar xvf ' .. tar)
end

if opt.patches ~= 'all' then
   opt.patches = math.floor(opt.patches/3)
end

----------------------------------------------------------------------
-- training/test size

local trsize,tesize
if opt.size == 'extra' then
   print '==> using extra training data'
   trsize = 73257 + 531131
   tesize = 26032
elseif opt.size == 'full' then
   print '==> using regular, full training data'
   trsize = 73257
   tesize = 26032
elseif opt.size == 'small' then
   print '==> using reduced training data, for fast experiments'
   trsize = 10000
   tesize = 2000
end

----------------------------------------------------------------------
print '==> loading dataset'

-- We load the dataset from disk, and re-arrange it to be compatible
-- with Torch's representation. Matlab uses a column-major representation,
-- Torch is row-major, so we just have to transpose the data.

-- Note: the data, in X, is 4-d: the 1st dim indexes the samples, the 2nd
-- dim indexes the color channels (RGB), and the last two dims index the
-- height and width of the samples.

local loaded = torch.load(train_file,'ascii')
local trainData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return trsize end
}

-- If extra data is used, we load the extra file, and then
-- concatenate the two training sets.

-- Torch's slicing syntax can be a little bit frightening. I've
-- provided a little tutorial on this, in this same directory:
-- A_slicing.lua

if opt.size == 'extra' then
   loaded = torch.load(extra_file,'ascii')
   local trdata = torch.Tensor(trsize,3,32,32)
   trdata[{ {1,(#trainData.data)[1]} }] = trainData.data
   trdata[{ {(#trainData.data)[1]+1,-1} }] = loaded.X:transpose(3,4)
   local trlabels = torch.Tensor(trsize)
   trlabels[{ {1,(#trainData.labels)[1]} }] = trainData.labels
   trlabels[{ {(#trainData.labels)[1]+1,-1} }] = loaded.y[1]
   trainData = {
      data = trdata,
      labels = trlabels,
      size = function() return trsize end
   }
end

-- Finally we load the test data.

local loaded = torch.load(test_file,'ascii')
local testData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return tesize end
}

----------------------------------------------------------------------
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
print '==> preprocessing data: colorspace RGB -> YUV'
for i = 1,trainData:size() do
   trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
for i = 1,testData:size() do
   testData.data[i] = image.rgb2yuv(testData.data[i])
end

-- Name channels for convenience
local channels = {'y','u','v'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print '==> preprocessing data: normalize each feature (channel) globally'
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

-- -- Local normalization
-- print '==> preprocessing data: normalize all three channels locally'

-- -- Define the normalization neighborhood:
-- local neighborhood = image.gaussian1D(11)

-- -- Define our local normalization operator (It is an actual nn module, 
-- -- which could be inserted into a trainable model):
-- local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- -- Normalize all channels locally:
-- for c in ipairs(channels) do
--    for i = 1,trainData:size() do
--       trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
--    end
--    for i = 1,testData:size() do
--       testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
--    end
-- end

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
   local trainMean = trainData.data[{ {},i }]:mean()
   local trainStd = trainData.data[{ {},i }]:std()

   local testMean = testData.data[{ {},i }]:mean()
   local testStd = testData.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end

----------------------------------------------------------------------
print '==> visualizing data'

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if opt.visualize then
   local first256Samples_y = trainData.data[{ {1,256},1 }]
   local first256Samples_u = trainData.data[{ {1,256},2 }]
   local first256Samples_v = trainData.data[{ {1,256},3 }]
   image.display{image=first256Samples_y, nrow=16, legend='Some training examples: Y channel'}
   image.display{image=first256Samples_u, nrow=16, legend='Some training examples: U channel'}
   image.display{image=first256Samples_v, nrow=16, legend='Some training examples: V channel'}
end

-- Exports
return {
   trainData = trainData,
   testData = testData,
   mean = mean,
   std = std
}

