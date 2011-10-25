#!/usr/bin/env lua
--------------------------------------------------------------------------------
-- A simple autoencoder to learn sparse coding, using 
-- AutoEncoderTrainer
-- 
-- (related info : http://www.scholarpedia.org/article/Sparse_coding) 
--
-- Authors: Benoit Corda, C. Farabet
--------------------------------------------------------------------------------
require 'XLearn'

-- Option parsing
op = OptionParser('trainAutoencoder [options]')
op:add_option{'-p', '--path', action='store', dest='path', help='path to train set. should contain pngs.'}

options,args = op:parse_args()

if (options.path == nil) then
   op:print_help()
   op:fail('please provide path')
end

-------------------------------------------
-- params
-------------------------------------------
winX = 9
winY = 9
woutX = 8
woutY = 8
overcompleteX = 1
overcompleteY = 1
goodForTrainingSizeX = 3*(winX+woutX-1) + 2*(winX-1)
goodForTrainingSizeY = 3*(winY+woutY-1) + 2*(winY-1)
autoencoderFileName = 'autoencoderTrained.torch'
-- code rates
maxiterCode = 40
alphaCode = 0.4
codeLearningRate = 0.005
-- encoder rates
alphaEncoder = 1
encoderLearningRate = 0.01
encoderLearningDecay = 0.01
alphaEncoder = 1
-- decoder rates
alphaDecoder = 1
decoderLearningRate = 0.01
decoderLearningDecay = 0.01
alphaDecoder = 1
-------------------------------------------
-- end params
-------------------------------------------


-- build modules and trainer
encoder = nn.Sequential()
encoder:add(nn.LcEncoder(winX, winY, woutX, woutY, 
                         overcompleteX, overcompleteY))
encoder:add(nn.Tanh())

decoder = nn.LcDecoder(winX, winY, woutX, woutY, overcompleteX, overcompleteY)

-- Loss functions
codeCriterion = nn.AbsCriterion()
codeCriterion.sizeAverage = false
decoderCriterion = nn.MSECriterion()
decoderCriterion.sizeAverage = false
encoderCriterion = nn.MSECriterion()
encoderCriterion.sizeAverage = false
-- trainer
if toolBox.fileExists(autoencoderFileName) then
   trainer = nn.AutoEncoderTrainer()
   trainer:open(autoencoderFileName)
   print('load successful...')
else
   trainer = nn.AutoEncoderTrainer(encoder, decoder, 
                                codeCriterion, 
                                decoderCriterion, encoderCriterion,
                                fileName, maxiterCode)
   -- set params
   trainer.decoderLearningRate = decoderLearningRate
   trainer.encoderLearningRate = encoderLearningRate
   trainer.codeLearningRate = codeLearningRate
   trainer.alphaDecoder = alphaDecoder
   trainer.alphaEncoder = alphaEncoder
   trainer.alphaCode = alphaCode
   trainer.fileName = autoencoderFileName
   -- debug display
   local function display(trainer, example, id)
      if id % 50 == 0 then
         trainer.decoder:showFeatures{}
         trainer.encoder.modules[1]:showFeatures{} 
      end
   end
   trainer.hookExample = display
end


-- Loading Training Dataset
trainData = UnsupDataSet(options.path,'.jpg')
-- reset code target
targetx = overcompleteX * (goodForTrainingSizeX-winX+1)
targety = overcompleteY * (goodForTrainingSizeY-winY+1)
trainer.codeTarget = torch.Tensor(targetx,targety,1):zero()


-- debugger
--w=gfx.Window(200,200,'Input Data')
function debug(trainer, example, id)
   --image.display(Example:select(3,1), 2, 'Input Data', w)
   w = image.qtdisplay{tensor=example:select(3,1),
                       zoom=2,
                       legend = 'Input Data',
                       painter = w}
   if id % 100 == 0 then
      trainer.decoder:showFeatures{}
      trainer.encoder.modules[1]:showFeatures{} 
   end
end
trainer.hookExample = debug


--saving every ite
trainer.savetime = 1
--training time
trainer:train(trainData,goodForTrainingSizeX,goodForTrainingSizeY)


