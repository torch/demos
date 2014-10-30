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
-- Mon Oct 14 14:58:50 EDT 2013
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nnx'      -- provides a normalization operator

local opt = opt or {
   visualize = true,
   size = 'small',
   patches='all'
}

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> downloading dataset')

-- Here we download dataset files. 

-- Note: files were converted from their original Matlab format
-- to Torch's internal format using the mattorch package. The
-- mattorch package allows 1-to-1 conversion between Torch and Matlab
-- files.


-- local www = 'http://data.neuflow.org/data/'
local www = 'https://engineering.purdue.edu/elab/files/'
--local train_dir = '../../datasets/faces_cut_yuv_32x32/'
local train_dir = '.' -- se to current directory (move to where you like!)
-- local tar = 'faces_cut_yuv_32x32.tar.gz'
local tar = 'face-dataset.zip'


--load from 'https://engineering.purdue.edu/elab/files/face-dataset.zip'
if not paths.dirp('face-dataset') then
   --os.execute('mkdir -p ' .. train_dir)
   os.execute('cd ' .. train_dir)
   os.execute('wget ' .. www .. tar)
   os.execute('unzip ' .. tar)
end

if opt.patches ~= 'all' then
   opt.patches = math.floor(opt.patches/3)
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> loading dataset')

-- We load the dataset from disk
-- data bg: from 0 to 28033; face: from 28034 to 41266
local imagesAll = torch.Tensor(41267,3,32,32)
local labelsAll = torch.Tensor(41267)

-- classes: GLOBAL var!
classes = {'face','backg'}

-- load backgrounds:
for f=0,28033 do
  imagesAll[f+1] = image.load('face-dataset/bg/bg_'..f..'.png') 
  labelsAll[f+1] = 2 -- 2 = background
end
-- load faces:
for f=28034,41266 do
  imagesAll[f+1] = image.load('face-dataset/face/face_'..f..'.png') 
  labelsAll[f+1] = 1 -- 1 = face
end

-- shuffle dataset: get shuffled indices in this variable:
local labelsShuffle = torch.randperm((#labelsAll)[1])

local portionTrain = 0.8 -- 80% is train data, rest is test data
local trsize = torch.floor(labelsShuffle:size(1)*portionTrain)
local tesize = labelsShuffle:size(1) - trsize

-- create train set:
trainData = {
   data = torch.Tensor(trsize, 1, 32, 32),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}
--create test set:
testData = {
      data = torch.Tensor(tesize, 1, 32, 32),
      labels = torch.Tensor(tesize),
      size = function() return tesize end
   }

for i=1,trsize do
   trainData.data[i] = imagesAll[labelsShuffle[i]][1]:clone()
   trainData.labels[i] = labelsAll[labelsShuffle[i]]
end
for i=trsize+1,tesize+trsize do
   testData.data[i-trsize] = imagesAll[labelsShuffle[i]][1]:clone()
   testData.labels[i-trsize] = labelsAll[labelsShuffle[i]]
end

-- remove from memory temp image files:
-- imagesAll = nil
-- labelsAll = nil


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> preprocessing data')
-- faces and bg are already YUV here, no need to convert!

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes). Types can be easily converted in Torch, 
-- in general by doing: dst = src:type('torch.TypeTensor'), 
-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
-- for simplicity (float(),double(),cuda(),...):

-- trainData.data = trainData.data:float()
-- testData.data = testData.data:float()

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
-- print '==> preprocessing data: colorspace RGB -> YUV'
-- for i = 1,trainData:size() do
--    trainData.data[i] = image.rgb2yuv(trainData.data[i])
-- end
-- for i = 1,testData:size() do
--    testData.data[i] = image.rgb2yuv(testData.data[i])
-- end

-- Name channels for convenience
local channels = {'y'}--,'u','v'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print(sys.COLORS.red ..  '==> preprocessing data: normalize each feature (channel) globally')
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

-- Local contrast normalization is needed in the face dataset as the dataset is already in this form:
print(sys.COLORS.red ..  '==> preprocessing data: normalize all three channels locally')

-- Define the normalization neighborhood:
local neighborhood = image.gaussian1D(5) -- 5 for face detector training

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for c in ipairs(channels) do
   for i = 1,trainData:size() do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
   end
   for i = 1,testData:size() do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
   end
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> verify statistics')

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
print(sys.COLORS.red ..  '==> visualizing data')

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if opt.visualize then
   local first256Samples_y = trainData.data[{ {1,256},1 }]
   image.display{image=first256Samples_y, nrow=16, legend='Some training examples: Y channel'}
   local first256Samples_y = testData.data[{ {1,256},1 }]
   image.display{image=first256Samples_y, nrow=16, legend='Some testing examples: Y channel'}
end

-- Exports
return {
   trainData = trainData,
   testData = testData,
   mean = mean,
   std = std,
   classes = classes
}

