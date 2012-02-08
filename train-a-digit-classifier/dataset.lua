require 'torch'
require 'paths'

mnist = {}

mnist.path_remote = 'http://data.neuflow.org/data/mnist-th7.tgz'
mnist.path_dataset = 'mnist-th7'
mnist.path_trainset = paths.concat(mnist.path_dataset, 'train.th7')
mnist.path_testset = paths.concat(mnist.path_dataset, 'test.th7')

function mnist.download()
   if not paths.filep(mnist.path_trainset) or not paths.filep(mnist.path_testset) then
      local remote = mnist.path_remote
      local tar = paths.basename(remote)
      os.execute('wget ' .. remote .. '; ' .. 'tar xvf ' .. tar .. '; rm ' .. tar)
   end
end

function mnist.loadTrainSet(maxLoad, geometry)
   return mnist.loadConvDataset(mnist.path_trainset, maxLoad, geometry)
end

function mnist.loadTestSet(maxLoad, geometry)
   return mnist.loadConvDataset(mnist.path_testset, maxLoad, geometry)
end

function mnist.loadFlatDataset(fileName, maxLoad)
   mnist.download()

   local f = torch.DiskFile(fileName, 'r')
   f:binary()

   local nExample = f:readInt()
   local dim = f:readInt()
   if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<mnist> loading only ' .. nExample .. ' examples')
   end
   print('<mnist> reading ' .. nExample .. ' examples with ' .. dim-1 .. '+1 dimensions...')
   local tensor = torch.Tensor(nExample, dim)
   tensor:storage():copy(f:readFloat(nExample*dim))
   print('<mnist> done')

   local dataset = {}
   dataset.tensor = tensor

   function dataset:normalize(mean_, std_)
      local data = tensor:narrow(2, 1, dim-1)
      local std = std_ or torch.std(data, 1, true)
      local mean = mean_ or torch.mean(data, 1)
      for i=1,dim-1 do
         tensor:select(2, i):add(-mean[1][i])
         if std[1][i] > 0 then
            tensor:select(2, i):mul(1/std[1][i])
         end
      end
      return mean, std
   end

   function dataset:normalizeGlobal(mean_, std_)
      local data = tensor:narrow(2, 1, dim-1)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   dataset.dim = dim-1

   function dataset:size()
      return nExample
   end

   local labelvector = torch.zeros(10)

   setmetatable(dataset, {__index = function(self, index)
                                       local input = tensor[index]:narrow(1, 1, dim-1)
                                       local class = tensor[index][dim]+1
                                       local label = labelvector:zero()
                                       label[class] = 1
                                       local example = {input, label}
                                       return example
                                    end})

   return dataset
end

function mnist.loadConvDataset(fileName, maxLoad, geometry)
   local dataset = mnist.loadFlatDataset(fileName, maxLoad)
   local cdataset = {}
   
   function cdataset:normalize(m,s)
      return dataset:normalize(m,s)
   end
   function cdataset:normalizeGlobal(m,s)
      return dataset:normalizeGlobal(m,s)
   end
   function cdataset:size()
      return dataset:size()
   end

   local iheight = geometry[2]
   local iwidth = geometry[1]
   local inputpatch = torch.zeros(1, iheight, iwidth)

   setmetatable(cdataset, {__index = function(self,index)
                                       local ex = dataset[index]
                                       local input = ex[1]
                                       local label = ex[2]
                                       local w = math.sqrt(input:nElement())
                                       local uinput = input:unfold(1,input:nElement(),input:nElement())
                                       local cinput = uinput:unfold(2,w,w)
                                       local h = cinput:size(2)
                                       local w = cinput:size(3)
                                       local x = math.floor((iwidth-w)/2)+1
                                       local y = math.floor((iheight-h)/2)+1
                                       inputpatch:narrow(3,x,w):narrow(2,y,h):copy(cinput)
                                       local example = {inputpatch, label}
                                       return example
                                    end})
   return cdataset
end
