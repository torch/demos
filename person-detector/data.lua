----------------------------------------------------------------------
-- This script loads the INRIA person dataset
-- training data, and pre-process it to facilitate learning.
-- E. Culurciello
-- April 2013
----------------------------------------------------------------------

require 'image'   -- to visualize the dataset
require 'ffmpeg'

----------------------------------------------------------------------

function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> downloading dataset')

-- Here we download dataset files. 

-- Note: files were converted from their original Matlab format
-- to Torch's internal format using the mattorch package. The
-- mattorch package allows 1-to-1 conversion between Torch and Matlab
-- files.

local www = 'https://engineering.purdue.edu/elab/files/'
local train_dir = '.' -- set to current directory (move to where you like!)
local tar = 'INRIAPerson.zip'

--load from 'https://engineering.purdue.edu/elab/files/INRIAPerson.zip'
if not paths.dirp('INRIAPerson') then
   --os.execute('mkdir -p ' .. train_dir)
   os.execute('cd ' .. train_dir)
   os.execute('wget ' .. www .. tar)
   os.execute('unzip ' .. tar)
end

if opt.patches ~= 'all' then
   opt.patches = math.floor(opt.patches/3)
end

----------------------------------------------------------------------
-- load or generate new dataset:

if paths.filep('train.t7') 
   and paths.filep('test.t7') then

   print(sys.COLORS.red ..  '==> loading previously generated dataset:')
   trainData = torch.load('train.t7')
   testData = torch.load('test.t7')

   trSize = trainData.data:size(1)
   teSize = testData.data:size(1)

else

   print(sys.COLORS.red ..  '==> creating a new dataset from raw files:')

   -- video dataset to get background from:
   local dspath = 'INRIAPerson/bg.m4v'
   local source = ffmpeg.Video{path=dspath, width=284/2, height=160/2, encoding='png', 
   		fps=30, lenght=100, delete=false, load=false}

   local rawFrame = source:forward()
   -- input video params:
   ivch = rawFrame:size(1) -- channels
   ivhe = rawFrame:size(2) -- height
   ivwi = rawFrame:size(3) -- width

   local desImaX = 46 -- desired cropped dataset image size
   local desImaY = 46

   local cropTrX = 45 -- desired offset to crop images from train set
   local cropTrY = 48
   local cropTeX = 33 -- desired offset to crop images from test set
   local cropTeY = 35

   local labelPerson = 1 -- label for person and background:
   local labelBg = 2

   local trainDir = 'INRIAPerson/96X160H96/Train/pos/'
   local trainImaNumber = #ls(trainDir)
   trSize = (trainImaNumber-1)*2 -- twice because of bg data!
   local testDir = 'INRIAPerson/70X134H96/Test/pos/'
   local testImaNumber = #ls(testDir)
   teSize = (testImaNumber-1)*2 -- twice because of bg data!

   trainData = {
      data = torch.Tensor(trSize, ivch, desImaX, desImaY),
      labels = torch.Tensor(trSize),
      size = function() return trSize end
   }

   -- shuffle dataset: get shuffled indices in this variable:
   local trShuffle = torch.randperm(trSize) -- train shuffle
   
   -- load person train data: 2416 images
   for i = 1, trSize, 2 do
      img = image.loadPNG(trainDir..ls(trainDir)[(i-1)/2+1],ivch) -- we pick all of the images in train!
      trainData.data[trShuffle[i]] = image.crop(img, cropTrX-desImaX/2, cropTrY-desImaY/2, 
      cropTrX+desImaX/2, cropTrY+desImaY/2):clone()
      trainData.labels[trShuffle[i]] = labelPerson

      -- load background data:
      img = source:forward()
      local x = math.random(1, ivwi-desImaX+1)
      local y = math.random(15, ivhe-desImaY+1-30) -- added # to get samples more or less from horizon
      trainData.data[trShuffle[i+1]] = img[{ {},{y,y+desImaY-1},{x,x+desImaX-1} }]:clone()
      trainData.labels[trShuffle[i+1]] = labelBg
   end
   -- display some examples:
   if opt.visualize then
       image.display{image=trainData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Train Data'}
   end

   testData = {
      data = torch.Tensor(teSize, ivch,desImaX,desImaY),
      labels = torch.Tensor(teSize),
      size = function() return teSize end
   }

   -- load person test data: 1126 images
   for i = 1, teSize, 2 do
      img = image.loadPNG(testDir..ls(testDir)[(i-1)/2+1],ivch) -- we pick all of the images in test!
      testData.data[i] = image.crop(img, cropTeX-desImaX/2, cropTeY-desImaY/2, 
      cropTeX+desImaX/2, cropTeY+desImaY/2):clone()
      testData.labels[i] = labelPerson

      -- load background data:
      img = source:forward()
      local x = math.random(1,ivwi-desImaX+1)
      local y = math.random(15,ivhe-desImaY+1-30) -- added # to get samples more or less from horizon
      testData.data[i+1] = img[{ {},{y,y+desImaY-1},{x,x+desImaX-1} }]:clone()
      testData.labels[i+1] = labelBg
   end
   -- display some examples:
   if opt.visualize then
       image.display{image=testData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Test Data'}
   end
   --save created dataset:
   torch.save('train.t7',trainData)
   torch.save('test.t7',testData)
end

-- Displaying the dataset architecture ---------------------------------------
print(sys.COLORS.red ..  'Training Data:')
print(trainData)
print()

print(sys.COLORS.red ..  'Test Data:')
print(testData)
print()

-- Preprocessing -------------------------------------------------------------
dofile 'preprocessing.lua'

trainData.size = function() return trSize end
testData.size = function() return teSize end


-- classes: GLOBAL var!
classes = {'person','backg'}

-- Exports -------------------------------------------------------------------
return {
   trainData = trainData,
   testData = testData,
   mean = mean,
   std = std,
   classes = classes
}
