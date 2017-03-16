require 'torch'
require 'nn'
require 'optim'
require 'cudnn'
require 'image';
require 'tds'
require 'lmdb'
require 'ffi'
require 'nngraph'
require 'stn'
require 'dpnn'

matio = require 'matio'
meanstd = {
   mean = { 0,-128,-128 },
   std = { 100,255,255 },
   scale = 1,
}
function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    local pfile = popen('ls -a "'..directory..'"')
    for filename in pfile:lines() do
        i = i + 1
        t[i] = filename
    end
    pfile:close()
    return t
end
function getRGBback(LAB_img_batch)
    for i = 1,LAB_img_batch:size(1) do
      thisImg = LAB_img_batch[i]
      thisImg = thisImg:squeeze():float()
      for channel=1,3 do
          thisImg[channel]:mul(meanstd.std[channel])
          thisImg[channel]:add(meanstd.mean[channel])                
      end               
      LAB_img_batch[i]:copy(image.lab2rgb(thisImg))
  end
  return LAB_img_batch
end

ds_verbose = false

opt = {
   dataset = 'lsun',       -- imagenet / lsun / folder
   batchSize = 8,
   loadSize = 256,
   fineSize = 224,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 500,             -- #  of iter at starting learning rate
   lr =  0.00005, -- 0.000001, --0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 13,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'ColorMove',
   noise = 'normal',       -- uniform / normal
   rot = false,       -- uniform / normal
   tra = true,
   sca = true,
   locnet = '',
   inputSize = 224,
   inputChannel = 3,
   no_cuda = false,
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data =  DataLoader.new(opt.nThreads, opt.dataset, opt)



print("Dataset1: " .. opt.dataset, " Size: ", data:size())
print("Dataset2: " .. opt.dataset, " Size: ", data:size2())
----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = 3
local nz = opt.nz    -- #  of dim for Z
local ndf = opt.ndf  -- #  of gen filters in first conv layer
local ngf = opt.ngf  -- #  of discrim filters in first conv layer
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = cudnn.SpatialConvolution
local SpatialFullConvolution = cudnn.SpatialFullConvolution


   -- nz = 100,               -- #  of dim for Z
   -- ngf = 64,               -- #  of gen filters in first conv layer
   -- ndf = 64,               -- #  of discrim filters in first conv layer
--local netG = nn.Sequential()
local networks = {}
-- These are the basic modules used when creating any macro-module
-- Can be modified to use for example cudnn
networks.modules = {}
networks.modules.convolutionModule = cudnn.SpatialConvolutionMM
networks.modules.poolingModule = cudnn.SpatialMaxPooling
networks.modules.nonLinearityModule = cudnn.ReLU

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
-- local st = networks.new_spatial_tranformer( opt.locnet,
--                                             opt.rot, opt.sca, opt.tra,
--                                             opt.inputSize, opt.inputChannel,
--                                             opt.no_cuda, opt)

   local input_img = nn.Identity()()
   -- input_scale = nn.Constant(0.5)
  local localization_network = nn.Sequential()
  print('Using pretrained ResNet-101 as the loc-net')
  local VGG16_loc = torch.load('../ICCV-netD-2-CUHKPQ/checkpoints/model_best.t7') -- torch.load('../ICCV-netD-2-finetuneWithCrop/checkpoints/model_best.t7')
  VGG16_loc:get(1).gradInput = torch.Tensor()
  VGG16_loc:remove(11)
  VGG16_loc:evaluate()
  local incep_fea = VGG16_loc(input_img)

   local m_sca = nn.HardTanh()(nn.Linear(2048,  2)(incep_fea))
   local const_k = (0.95 - 0.5) / 2;
   local const_b = 0.95 - const_k;
   m_sca = nn.AddConstant(  const_b  )(nn.MulConstant( const_k )(m_sca)) -- This is [0.3, 0.95]

   local m_tra = nn.Linear(2048,  2)(incep_fea)
   require 'dsBound'
   dsBound = nn.dsBound()
   local m_tra_bounded = dsBound({m_tra, m_sca})

   require 'labFilter'
   alpha = nn.HardTanh()(nn.Linear(2048, 2)(incep_fea))

   local alpha_const_k = (0.49 - 0.05) / 2;
   local alpha_const_b = 0.49 - alpha_const_k;
   alpha = nn.AddConstant(  alpha_const_b  )(nn.MulConstant( alpha_const_k )(alpha)) -- This is [0.3, 0.95]
   lab_moved = nn.labFilter()({input_img, alpha})   

   require 'contrastFilter'
   local contrast = nn.HardTanh()(nn.Linear(2048,1)(incep_fea))
   local contrast_const_k = (1.5 - 0.5) / 2;
   local contrast_const_b = 1.5 - contrast_const_k
   contrast = nn.AddConstant( contrast_const_b  )(nn.MulConstant(contrast_const_k)(contrast))

   local brightness = nn.HardTanh()(nn.Linear(2048,1)(incep_fea))
   local brightness_const_k = (0.5 + 0.5) / 2;
   local brightness_const_b = 0.5 - brightness_const_k
   brightness = nn.AddConstant( brightness_const_b  )(nn.MulConstant(brightness_const_k)(brightness))

   local m_fc7_1 = nn.JoinTable(1,1){m_sca, m_tra_bounded}

   m_transp1 = nn.Transpose({2,3},{3,4})(lab_moved) -- rot, sca or tra
   m_affineT = nn.AffineTransformMatrixGenerator(false, true, true)(m_fc7_1)
   m_affineG = nn.AffineGridGeneratorBHWD(224,224)(m_affineT)
   m_bilinear = nn.BilinearSamplerBHWD(){m_transp1, m_affineG}
   output_img = nn.Transpose({3,4},{2,3})(m_bilinear)




   -- MLK transform
   -- T_linear = nn.Linear(2048,4)
   -- T_linear.bias = torch.Tensor({1,0,1,0})
   -- T_linear.weight:zero()

   -- T = T_linear(incep_fea)

   -- T_correct = nn.Linear(2048,2)   
   -- T_correct.weight:zero()
   -- colorCorrection = T_correct(incep_fea)

   -- colorCorrection = nn.Replicate(224*224, 2)(colorCorrection)
   -- colorCorrection = nn.Transpose({2,3})(colorCorrection)
   -- colorCorrection = nn.Reshape(2,224,224,true)(colorCorrection)

   -- T_transform = nn.Reshape(2,2,true)(T)

   -- chrome = nn.Narrow(2,2,2)(output_img)
   -- luminance = nn.Narrow(2,1,1)(output_img)

   -- meanImg = nn.Reshape(2,224*224,true)(chrome)
   -- meanImg = nn.Mean(3)(meanImg)
   -- meanImg = nn.Replicate(224*224,2)(meanImg)
   -- meanImg = nn.Transpose({2,3})(meanImg)
   -- meanImg = nn.Reshape(2,224,224,true)(meanImg)
   -- minus_meanImg = nn.MulConstant(-1)(meanImg)

   -- subtracted = nn.CAddTable()({chrome, minus_meanImg})

   -- subtracted = nn.Reshape(2,224*224,true)(subtracted)
   -- subtracted = nn.Transpose({2,3})(subtracted)

   -- transformed = nn.MM()({subtracted, T_transform})
   -- transformed = nn.Transpose{2,3}(transformed)
   -- transformed = nn.Reshape(2,224*224,true)(transformed)

   -- transformed_output = nn.CAddTable()({transformed, meanImg, colorCorrection})
   -- final_output = nn.JoinTable(2)({luminance, transformed_output})



   STN = nn.gModule({input_img}, {output_img})





netG = nn.Sequential()
netG:add(STN)


 dummy_input_img =nn.Identity()()
 dummy_m_ds_sca = nn.Identity()()
 dummy_m_ds_tra = nn.Identity()()
 dummy_dsBound = nn.dsBound()
 dummy_m_ds_tra_bounded = dsBound({dummy_m_ds_tra, dummy_m_ds_sca})
 dummy_m_fc7_1 = nn.JoinTable(2)({dummy_m_ds_sca, dummy_m_ds_tra_bounded})
 dummy_m_transp1 = nn.Transpose({2,3},{3,4})(dummy_input_img) -- rot, sca or tra
 dummy_m_affineT = nn.AffineTransformMatrixGenerator(false, true, true)(dummy_m_fc7_1)
 dummy_m_affineG = nn.AffineGridGeneratorBHWD(224,224)(dummy_m_affineT)
 dummy_m_bilinear = nn.BilinearSamplerBHWD(){dummy_m_transp1, dummy_m_affineG}
 dummy_output_img = nn.Transpose({3,4},{2,3})(dummy_m_bilinear)
 dummy_stn  = nn.gModule({dummy_input_img, dummy_m_ds_sca, dummy_m_ds_tra}, {dummy_output_img})




VGG16_loc = nil
collectgarbage()

local function deepCopy(tbl)
   --print('Making a clean copy before saving it')
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

local ds_timer = torch.Timer()
local model_copy = netG:clearState():float():clone()
print('clone copy created')
local lrs1 = model_copy:getParameters()
lrs1:fill(1);

local VGG_loc_module_conv = model_copy:get(1):get(2)
assert(VGG_loc_module_conv ~= nil)
VGG_loc_module_conv:apply(function(m) 
  if m.weight then
     m.weight:fill(0)
  end
  if m.bias then
     m.bias:fill(0)
  end
end
)


lr_multiplierG = lrs1:clone():cuda()
lrs1 = nil
model_copy = nil
collectgarbage()
print('NetG created successfully')







-- netG:apply(weights_init)

local netD = nn.Sequential()
local VGG16 = torch.load('../ICCV-netD-2-CUHKPQ/checkpoints/model_best.t7') -- torch.load('../ICCV-netD-2-finetuneWithCrop/checkpoints/model_best.t7') --
VGG16:get(1).gradInput = torch.Tensor()
VGG16:evaluate()
local new_layer = VGG16:get(11):clone()
VGG16:remove(11)
netD:add(VGG16)

netD:add(new_layer)
netD:add(nn.PReLU())
netD:add(nn.Linear(2,1))
-- netD:add(nn.Narrow(2,1,1))
netD:add(nn.Mean(1))



local model_copy = netD:clearState():float():clone()
local lrs2 = model_copy:getParameters()
lrs2:fill(1);

local VGG_loc_module_conv = model_copy:get(1)
assert(VGG_loc_module_conv ~= nil)
VGG_loc_module_conv:apply(function(m) 
  if m.weight then
     m.weight:fill(0)
  end
  if m.bias then
     m.bias:fill(0)
  end
end
)

lr_multiplierD = lrs2:clone():cuda()
lrs2 = nil
model_copy = nil
collectgarbage()



VGG16 = nil
collectgarbage()
print('netD created successfully')



Lab2RGB_module = torch.load('../dcgan.Lab2RGB.3layer/checkpoints/Lab2RGB_50_net_G.t7')
cudnn.convert(Lab2RGB_module, nn)
Lab2RGB_module:evaluate()
input_img = nn.Identity()()
input255 = nn.MulConstant(255)(input_img)
R = nn.Narrow(2,1,1)(input255)
r_con_1 = nn.Constant(-103.939, 3)(R)
r_con_1 = nn.Replicate(224,3)(r_con_1)
r_con_1 = nn.Replicate(224,3)(r_con_1)
R_new = nn.CAddTable()({R, r_con_1})
G = nn.Narrow(2,2,1)(input255)
g_con_1 = nn.Constant(-116.779, 3)(G)
g_con_1 = nn.Replicate(224,3)(g_con_1)
g_con_1 = nn.Replicate(224,3)(g_con_1)
G_new = nn.CAddTable()({G, g_con_1})
B = nn.Narrow(2,3,1)(input255)
b_con_1 = nn.Constant(-123.68, 3)(B)
b_con_1 = nn.Replicate(224,3)(b_con_1)
b_con_1 = nn.Replicate(224,3)(b_con_1)
B_new = nn.CAddTable()({B, b_con_1})
normalized_rgb = nn.JoinTable(2)({R_new, G_new, B_new})
normalization_module = nn.gModule({input_img}, {normalized_rgb})
Lab2RGB_FULL = nn.Sequential()
Lab2RGB_FULL:add(Lab2RGB_module):add(normalization_module)
Lab2RGB_FULL:cuda()
perceptual_loss_net = torch.load('../fast-neural-style/models/vgg16.t7')
perceptual_loss_net:evaluate()
perceptual_loss_net:remove(40)
perceptual_loss_net:remove(39)


DS_FULL_Perceptual_NETWORK = nn.Sequential()
DS_FULL_Perceptual_NETWORK:add(Lab2RGB_FULL):add(perceptual_loss_net)
DS_FULL_Perceptual_NETWORK:evaluate()
DS_FULL_Perceptual_NETWORK:cuda()

local percep_crit = nn.MSECriterion()


local criterion = nn.BCECriterion()
require 'MSECriterionDS'
local criterionG = nn.MSECriterionDS()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local input2 = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local input_per1 = torch.Tensor(opt.batchSize, 4096)
local input_per2 = torch.Tensor(opt.batchSize, 4096)
local output_good = torch.Tensor(opt.batchSize, 1)
local output_bad = torch.Tensor(opt.batchSize, 1)

local dummy_zero_translation = torch.Tensor(opt.batchSize, 2):fill(0):cuda()
local dummy_identity_scale = torch.Tensor(opt.batchSize, 2):fill(1):cuda()

-- GAN loss
local df_dg_gan = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
-- regularization
local df_dg_reg = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
-- perceptual loss
local df_dg_per = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)


local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local fakeSource = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local errD
local errG = torch.Tensor(1)
local errD_real = torch.Tensor(1)
local errD_fake = torch.Tensor(1)
local grad_of_ones = torch.Tensor(1):fill(1)
local grad_of_mones = torch.Tensor(1):fill(-1)
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  fakeSource = fakeSource:cuda();
   input2 = input2:cuda()
   input_per1 = input_per1:cuda()
   input_per2 = input_per2:cuda()
   output_good = output_good:cuda()
   output_bad = output_bad:cuda()
   df_dg_gan = df_dg_gan:cuda()
   df_dg_reg = df_dg_reg:cuda()
   df_dg_per = df_dg_per:cuda()


   errD_real = errD_real:cuda()
   errD_fake = errD_fake:cuda()
   errG = errG:cuda()

   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      -- cudnn.convert(netG, cudnn)
      -- cudnn.convert(netD, cudnn)
   end
   netD:cuda();
   netG:cuda();
   criterion:cuda()
   criterionG:cuda()
   percep_crit:cuda()
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

-- noise_vis = noise:clone()
-- if opt.noise == 'uniform' then
--     noise_vis:uniform(-1, 1)
-- elseif opt.noise == 'normal' then
--     noise_vis:normal(0, 1)
-- end

one_time_counter = true
-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume() -- timer
   local real = data:getBatch()
   data_tm:stop()
   --input:copy(real)
   input:copy(dummy_stn:forward({real:float(), dummy_identity_scale, dummy_zero_translation}))
   local output = netD:forward(input)
   errD_real:copy(output)
   -- criterion:forward(output, label)
   -- local df_do = criterion:backward(output, label)
   --netD:backward(input, df_do)
   netD:backward(input, grad_of_ones)
   if ds_verbose then
     before_zeroing = gradParametersD:norm()
     gradParametersD:cmul(lr_multiplierD)
   
      io.write(string.format('gradient-updateD-REAL: (%f) %f\n', before_zeroing, gradParametersD:norm()))
   end

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end

   data_tm:resume() -- timer
   local myfakeSource = data:getBatch2()
   if one_time_counter then
      noise_vis = myfakeSource:clone()
      one_time_counter = false
   end

   data_tm:stop()
   fakeSource:copy(myfakeSource)
   --local fake = netG:forward(fakeSource)
   local fake = netG:forward(fakeSource)
   -- print(netG:get(1):get(9))   
   input:copy(fake)

   local output = netD:forward(input)

   errD_fake:copy(output)
   -- criterion:forward(output, label)
   -- local df_do = criterion:backward(output, label)
   -- netD:backward(input, df_do)
   -- print('NetD backward should have dim')
   -- print(df_do:size())
   netD:backward(input, grad_of_mones)
   if ds_verbose then
     before_zeroing = gradParametersD:norm()
     gradParametersD:cmul(lr_multiplierD:cuda())
     io.write(string.format('gradient-updateD-FAKE: (%f) %f\n', before_zeroing, gradParametersD:norm()))
   end
   --errD = errD_real + errD_fake
   errD = errD_real - errD_fake
   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()

   local myfakeSource = data:getBatch2()
   fakeSource:copy(myfakeSource)
   input2:copy(fakeSource)
   input2:copy(dummy_stn:forward({myfakeSource:float(),dummy_identity_scale, dummy_zero_translation}))
   target = netD:forward(input2)
   output_bad:copy(netD:get(4).output)


   local fake = netG:forward(fakeSource)
   input:copy(fake)
   output = netD:forward(input)
   output_good:copy(netD:get(4).output)
   errG:copy(output)

   -- (1) Gan Loss
   local df_dg = netD:updateGradInput(input, grad_of_ones)
   df_dg_gan:copy(df_dg)
   local df_dg_gan_norm = df_dg_gan:norm()
   if   (df_dg_gan_norm >= 3) then
         df_dg_gan:mul(3.0/df_dg_gan_norm)
   end      

    -- (2) Regularization Loss
   local dummy = criterionG:forward(output_good, output_bad)
   local df_dD = criterionG:backward(output_good, output_bad)
5
   for d = 4,1,-1 do
      if d > 1 then
        df_dD = netD:get(d):updateGradInput(netD:get(d-1).output, df_dD)
      else
        df_dD = netD:get(1):updateGradInput(input, df_dD)
      end
   end
   df_dg_reg:copy(df_dD)
   local df_dg_reg_norm = df_dg_reg:norm()
   if   (df_dg_reg_norm >= 3) then
         df_dg_reg:mul(3.0/df_dg_reg_norm)
   end  



   -- (3) Perceptual Loss
  fc7_1 = DS_FULL_Perceptual_NETWORK:forward(input)    -- cropped image
  input_per1:copy(fc7_1)
  fc7_2 = DS_FULL_Perceptual_NETWORK:forward(input2)    -- original bad image
  input_per2:copy(fc7_2)
  -------------------------------------------------- --pred, y
  local percep_loss = percep_crit:forward(input_per1, input_per2)
  local grad_out_percep = percep_crit:backward(input_per1, input_per2)
  grad_out_percep = DS_FULL_Perceptual_NETWORK:updateGradInput(input, grad_out_percep)
  df_dg_per:copy(grad_out_percep)
   local df_dg_per_norm = df_dg_per:norm()
   if   (df_dg_per_norm >= 3) then
         df_dg_per:mul(3.0/df_dg_per_norm)
   end  

   io.write(string.format('Gradient contribution G1-gan: %f, G2-reg: %f, G3-per: %f\n', df_dg_gan:norm(), df_dg_reg:norm(), df_dg_per:norm()))

   local myNorm_G = (df_dg_gan+df_dg_reg+df_dg_per):norm()
   netG:backward(fakeSource, df_dg_gan+df_dg_reg+df_dg_per)
   gradParametersG:cmul(lr_multiplierG)

   io.write(string.format('Gradient to attack STN fc params, [scale_x: %f, scale_y: %f, x: %f, y: %f] - total gradient from D at G: %f - \n', 
    netG:get(1):get(15).gradInput[1][1][1], netG:get(1):get(15).gradInput[1][1][2],
    netG:get(1):get(14).gradInput[1][1][1], netG:get(1):get(14).gradInput[1][1][2],
    myNorm_G))
   io.write(string.format('Gradient to attack STN fc params, [alphaA: %f, betaB: %f] -- ',  netG:get(1):get(4).gradInput[1][1],  netG:get(1):get(4).gradInput[1][2]))

   local myNorm = gradParametersG:norm()
   io.write(string.format('Total GradientUpdate norm = %f -- not clip at 100\n',myNorm))
   return errG, gradParametersG
end

-- train

gen_iteration = 0

generation_iter = 0
for epoch = 1, opt.niter do

    netG:training()
    -- netG:get(1):get(3):evaluate()
    netD:training()
    -- netD:get(1):evaluate()

   epoch_tm:reset()
   local counter = 0
   i = 0
   while i <= math.min(data:size(), opt.ntrain) do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      j = 0

      Diters = 0
      if generation_iter < 25 or generation_iter % 500 == 0 then
          Diters = 100
      else
          Diters = 5
      end
      Diters = 1
      while j < Diters and i <= math.min(data:size(), opt.ntrain) do
        optim.rmsprop(fDx, parametersD, optimStateD)
        -- dannyLinearD:clamp(-0.01,0.01)        
        parametersD:clamp(-2,2)

        j = j + 1
        i = i + opt.batchSize
        counter = counter + 1
      end

      Giters = 5
      k = 0
      while k < Giters and i <= math.min(data:size(), opt.ntrain) do
        -- (2) Update G network: maximize log(D(G(z)))
        --print(string.format(' k = %d, i = %d, Giters = %d, total = %d', k, i, Giters,math.min(data:size(), opt.ntrain)))
        generation_iter = generation_iter + 1
        optim.rmsprop(fGx, parametersG, optimStateG)
        i = i + opt.batchSize
        k = k + 1
      end
      -- display
      counter = counter + 1
      if generation_iter % 1 == 0 and opt.display then
          local myfakeSource = data:getBatch2()
          noise_vis:copy(myfakeSource)
          local fake = netG:forward(noise_vis:cuda())
          local real = data:getBatch()
          
          -- print(string.format('L: [%.3f, %.3f]',fake[1][1]:min(),fake[1][1]:max()))
          -- print(string.format('a: [%.3f, %.3f]',fake[1][2]:min(),fake[1][2]:max()))
          -- print(string.format('b: [%.3f, %.3f]',fake[1][3]:min(),fake[1][3]:max()))

          -- matio.save(string.format('./generated_imgs/%d.mat',generation_iter), fake[1]:float())
          noise_vis1 = getRGBback(noise_vis)
          fake1 = getRGBback(fake)       
          -- image.save(string.format('./generated_imgs/%d.jpg',generation_iter), fake1[1]:float())   
          real1 = getRGBback(real)          

          savePath1 = string.format('./generated_imgs/Iter_%d_input.jpg',generation_iter)
          savePath2 = string.format('./generated_imgs/Iter_%d_enhanced.jpg',generation_iter)
          if generation_iter % 10 == 0 then
            disp.image(noise_vis1, {normalize=false,win=opt.display_id,    title=string.format('low-quality: %s',opt.name), saveThisOne=true, saveName=savePath1})
            disp.image(fake1,      {normalize=false,win=opt.display_id * 3,title=string.format('Enhanced:%s',opt.name),    saveThisOne=true, saveName=savePath2})
            disp.image(real1,      {normalize=false,win=opt.display_id * 9,title=string.format('High-quality:%s',opt.name)})
          else
            disp.image(noise_vis1, {normalize=false,win=opt.display_id,    title=string.format('low-quality:%s',opt.name), saveThisOne=false, saveName=savePath1})
            disp.image(fake1,      {normalize=false,win=opt.display_id * 3,title=string.format('Enhanced:%s',opt.name),    saveThisOne=false, saveName=savePath2})
            disp.image(real1,      {normalize=false,win=opt.display_id * 9,title=string.format('High-quality:%s',opt.name)})
          end
      end

      -- logging
         ds_n = ((i-1) / opt.batchSize)
         ds_trainSize = math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize)

         io.write(string.format('[outputs]: alpha = %.3f\n  [outputs]: beta = %.3f\n [outputs] s1 = %.3f, s2 = %.3f, t1 = %.3f, t2 = %.3f\n', netG:get(1):get(6).output[1][1], netG:get(1):get(6).output[1][2],
                   netG:get(1):get(15).output[1][1],netG:get(1):get(15).output[1][2],netG:get(1):get(15).output[1][3],netG:get(1):get(15).output[1][4]))

         print(string.format('LR_Mul_D = %f', lr_multiplierD:sum()))

         fc_index = 2
         print(string.format('netD-FC: %f (%f), %f; gradInput: %f\n', 
          netD:get(fc_index).weight:norm(),netD:get(fc_index).weight[1][1], netD:get(fc_index).bias[1], netD:get(fc_index).gradInput:norm()))
         fc_index = 4
         print(string.format('netD-FC: %f (%f), %f; gradInput: %f\n', 
          netD:get(fc_index).weight:norm(),netD:get(fc_index).weight[1][1], netD:get(fc_index).bias[1], netD:get(fc_index).gradInput:norm()))

         print(('Epoch: [%d][%8d %8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. ' Loss_D %.4f, Loss_G %.4f, Loss_D_real %.4f, Loss_D_fake %.4f  Time %.3f  ETA: %7.3f'):format(
                 epoch, generation_iter, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errD and errD[1] or -1, errG and errG[1] or -1, errD_real and errD_real[1] or -1, errD_fake and errD_fake[1] or -1,
                 epoch_tm:time().real, epoch_tm:time().real/(ds_n/ds_trainSize))
              )

       if generation_iter % 200 == 0 then

   
          netG:training()
          netD:training()

           -- Checkpointing
           paths.mkdir('checkpoints')
           parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
           parametersG, gradParametersG = nil, nil
           torch.save('checkpoints/' .. opt.name .. '_' .. generation_iter .. '_net_G.t7', netG:clearState())
           torch.save('checkpoints/' .. opt.name .. '_' .. generation_iter .. '_net_D.t7', netD:clearState())
           parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
           parametersG, gradParametersG = netG:getParameters()
           print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
                    epoch, opt.niter, epoch_tm:time().real))         
             
          netG:training()
          netD:training() 
      end         

   end
   


   if epoch % 4 == 0 then
      optimStateD.learningRate = optimStateD.learningRate / 10.0
      print(string.format('Learning Rate is Changed to %f at the end of epoch %d', optimStateD.learningRate, epoch))
   end


end
