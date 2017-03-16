require 'image'
require 'nn'
require 'torch'
require 'nn'
require 'optim'
require 'cudnn'
require 'image';
require 'tds'
require 'lmdb'
require 'ffi'
require 'stn'



require 'nngraph'
require 'stn'

lightingContrast, _ = torch.class('nn.lightingContrast', 'nn.Module')

function lightingContrast:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table')
   input, constantList = unpack(inputTable)
   self.output = self.output or input.new()
   self.output:resizeAs(input):copy(input)
   for i = 1,input:size(1) do
      local alpha = constantList:narrow(1,i,1)[1][1]
      --local beta = constantList:narrow(1,i,1)[1][2]
      --self.output:narrow(1, i, 1):mul(alpha):add(-128/255*alpha+128/255):add(beta)
      self.output:narrow(1, i, 1):pow(alpha)
   end
   return self.output
end

function lightingContrast:updateGradInput(inputTable, gradOutput)
   assert(torch.type(inputTable) == 'table')
   input, constantList = unpack(inputTable)
   --Img
   self.gradInput1 = self.gradInput1 or input.new()
   self.gradInput1:resizeAs(gradOutput)
   self.gradInput1:copy(gradOutput)
   for i = 1,input:size(1) do
      local alpha = constantList:narrow(1,i,1)[1][1]
      --local beta = constantList:narrow(1,i,1)[1][2]
      --self.gradInput1:mul(alpha)
      self.gradInput1:narrow(1,i,1):cmul(   input:narrow(1,i,1):pow(alpha-1):mul(alpha)   )
   end
   -- Constant
   -- self.gradInput2 = self.gradInput2 or input.new()
   -- self.gradInput2:resizeAs(input):copy(input):add(-128/255)
   -- self.gradInput2:resize(self.gradInput2:size(1), 
   --                        self.gradInput2:size(2)*self.gradInput2:size(3)*self.gradInput2:size(4))

   -- self.gradInput3 = self.gradInput3 or torch.ones(input:size(1), 1)
   self.gradInput2 = self.gradInput2 or input.new()
   self.gradInput2:resizeAs(gradOutput):copy(gradOutput)
   for i = 1,input:size(1) do
      local alpha = constantList:narrow(1,i,1)[1][1]
      self.gradInput2:narrow(1,i,1):cmul(   torch.pow(alpha, input:narrow(1,i,1)):mul(torch.log(alpha))   )
   end
   self.gradInput2:resize(self.gradInput2:size(1), 
                          self.gradInput2:size(2)*self.gradInput2:size(3)*self.gradInput2:size(4))

   self.gradInput = {}
   self.gradInput[1] = self.gradInput1
   self.gradInput[2] = torch.sum(self.gradInput2,2)--torch.cat(torch.sum(self.gradInput2, 2), self.gradInput3:cuda())
end



local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 32,        -- number of samples to produce
    noisetype = 'normal',  -- type of noise distribution (uniform / normal).
    net = '',              -- path to the generator network
    imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
    noisemode = 'random',  -- random / line / linefull1d / linefull
    name = 'generation1',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    display = 1,           -- Display image: 0 = false, 1 = true
    nz = 100,              
    dataset = 'folder',       -- imagenet / lsun / folder
    nThreads = 4,           -- #  of data loading threads to use
    loadSize = 256,
    fineSize = 224,
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)



if opt.display == 0 then opt.display = false end

assert(net ~= '', 'provide a generator model')

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
end

noise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
net = torch.load(opt.net)

-- for older models, there was nn.View on the top
-- which is unnecessary, and hinders convolutional generations.
if torch.type(net:get(1)) == 'nn.View' then
    net:remove(1)
end

print(net)

if opt.noisetype == 'uniform' then
    noise:uniform(-1, 1)
elseif opt.noisetype == 'normal' then
    noise:normal(0, 1)
end

noiseL = torch.FloatTensor(opt.nz):uniform(-1, 1)
noiseR = torch.FloatTensor(opt.nz):uniform(-1, 1)
if opt.noisemode == 'line' then
   -- do a linear interpolation in Z space between point A and point B
   -- each sample in the mini-batch is a point on the line
    line  = torch.linspace(0, 1, opt.batchSize)
    for i = 1, opt.batchSize do
        noise:select(1, i):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull1d' then
   -- do a linear interpolation in Z space between point A and point B
   -- however, generate the samples convolutionally, so a giant image is produced
    assert(opt.batchSize == 1, 'for linefull1d mode, give batchSize(1) and imsize > 1')
    noise = noise:narrow(3, 1, 1):clone()
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        noise:narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull' then
   -- just like linefull1d above, but try to do it in 2D
    assert(opt.batchSize == 1, 'for linefull mode, give batchSize(1) and imsize > 1')
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        noise:narrow(3, i, 1):narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
end

local DataLoader = paths.dofile('data/data.lua')
local data =  DataLoader.new(opt.nThreads, opt.dataset, opt)
one_time_counter = true
local sample_input = torch.randn(2,3,224,224)
for checkpointNUM = 1,100 do
    net = torch.load(string.format('./checkpoints/experiment1_%d_net_G.t7',checkpointNUM))
    foldername = 'generation_imgs'
    SAVE_NAME = string.format('%s/generation_checkpoint_%d.png', foldername,checkpointNUM)
    if torch.type(net:get(1)) == 'nn.View' then
        net:remove(1)
    end

    --print(net)

    if opt.gpu > 0 then
        net:cuda()
        if one_time_counter then
             myfakeSource = data:getBatch2()
             noise = myfakeSource:clone()
             one_time_counter = false
             noise = noise:cuda()
        end
        sample_input = sample_input:cuda()  
    else
       sample_input = sample_input:float()
       net:float()
    end

    -- a function to setup double-buffering across the network.
    -- this drastically reduces the memory needed to generate samples
    print('What is sample_input')
    print(sample_input:size())
    --optnet.optimizeMemory(net, sample_input:float())

    local images = net:forward(noise)
    print('Images size: ', images:size(1)..' x '..images:size(2) ..' x '..images:size(3)..' x '..images:size(4))
    images:add(1):mul(0.5)
    print('Min, Max, Mean, Stdv', images:min(), images:max(), images:mean(), images:std())
    image.save(SAVE_NAME, image.toDisplayTensor(images))
    print('Saved image to: ', SAVE_NAME .. '.png')

    if opt.display then
        disp = require 'display'
        disp.image(images)
        print('Displayed image')
    end
end
