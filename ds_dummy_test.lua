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
require 'cunn'
matio = require 'matio'
meanstd = {
   mean = { 0,-128,-128 },
   std = { 100,255,255 },
   scale = 1,
}
function scandir(directory)
    i, t, popen = 0, {}, io.popen
    pfile = popen('ls -a "'..directory..'"')
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
   dataset = 'folder',       -- imagenet / lsun / folder
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

----------------------------------------------------------------------------
function weights_init(m)
   name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

nc = 3
nz = opt.nz    -- #  of dim for Z
ndf = opt.ndf  -- #  of gen filters in first conv layer
ngf = opt.ngf  -- #  of discrim filters in first conv layer
real_label = 1
fake_label = 0

SpatialBatchNormalization = nn.SpatialBatchNormalization
SpatialConvolution = cudnn.SpatialConvolution
SpatialFullConvolution = cudnn.SpatialFullConvolution


   -- nz = 100,               -- #  of dim for Z
   -- ngf = 64,               -- #  of gen filters in first conv layer
   -- ndf = 64,               -- #  of discrim filters in first conv layer
--netG = nn.Sequential()
networks = {}
-- These are the basic modules used when creating any macro-module
-- Can be modified to use for example cudnn
networks.modules = {}
networks.modules.convolutionModule = cudnn.SpatialConvolutionMM
networks.modules.poolingModule = cudnn.SpatialMaxPooling
networks.modules.nonLinearityModule = cudnn.ReLU

Convolution = cudnn.SpatialConvolution
Avg = cudnn.SpatialAveragePooling
ReLU = cudnn.ReLU
Max = nn.SpatialMaxPooling
SBatchNorm = nn.SpatialBatchNormalization
-- st = networks.new_spatial_tranformer( opt.locnet,
--                                             opt.rot, opt.sca, opt.tra,
--                                             opt.inputSize, opt.inputChannel,
--                                             opt.no_cuda, opt)

   input_img = nn.Identity()()
   -- input_scale = nn.Constant(0.5)
  localization_network = nn.Sequential()
  print('Using pretrained ResNet-101 as the loc-net')
  VGG16_loc = torch.load('../ICCV-netD-2-CUHKPQ/checkpoints/model_best.t7') -- torch.load('../ICCV-netD-2-finetuneWithCrop/checkpoints/model_best.t7')
  VGG16_loc:get(1).gradInput = torch.Tensor()
  VGG16_loc:remove(11)
  VGG16_loc:evaluate()
  incep_fea = VGG16_loc(input_img)

   m_sca = nn.HardTanh()(nn.Linear(2048,  2)(incep_fea))
   --m_sca = nn.Constant(1,1)(nn.Linear(1024,  1)(m_view1))

   const_k = (0.95 - 0.3) / 2;
   const_b = 0.95 - const_k;
   m_sca = nn.AddConstant(  const_b  )(nn.MulConstant( const_k )(m_sca)) -- This is [0.3, 0.95]

   m_tra = nn.Linear(2048,  2)(incep_fea)

   require 'dsBound'
   dsBound = nn.dsBound()
   m_tra_bounded = dsBound({m_tra, m_sca})

   require 'labFilter'
   alpha = nn.Linear(2048, 2)(incep_fea)

   alpha_const_k = (0.49 - 0.01) / 2;
   alpha_const_b = 0.49 - alpha_const_k;
   alpha = nn.AddConstant(  alpha_const_b  )(nn.MulConstant( alpha_const_k )(alpha)) -- This is [0.3, 0.95]
   lab_moved = nn.labFilter()({input_img, alpha})   


   m_fc7_1 = nn.JoinTable(1,1){m_sca, m_tra_bounded}

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

function deepCopy(tbl)
   --print('Making a clean copy before saving it')
   -- creates a copy of a network with new modules and the same tensors
   copy = {}
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

ds_timer = torch.Timer()
model_copy = netG:clearState():float():clone()
print('clone copy created')
lrs1 = model_copy:getParameters()
lrs1:fill(1);

VGG_loc_module_conv = model_copy:get(1):get(2)
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

netD = nn.Sequential()
VGG16 = torch.load('../ICCV-netD-2-CUHKPQ/checkpoints/model_best.t7') -- torch.load('../ICCV-netD-2-finetuneWithCrop/checkpoints/model_best.t7') --
VGG16:get(1).gradInput = torch.Tensor()
VGG16:evaluate()
new_layer = VGG16:get(11):clone()
VGG16:remove(11)
netD:add(VGG16)

netD:add(new_layer)
netD:add(nn.PReLU())
netD:add(nn.Linear(2,1))
-- netD:add(nn.Narrow(2,1,1))
netD:add(nn.Mean(1))



model_copy = netD:clearState():float():clone()
lrs2 = model_copy:getParameters()
lrs2:fill(1);

VGG_loc_module_conv = model_copy:get(1)
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

percep_crit = nn.MSECriterion()


criterion = nn.BCECriterion()
require 'MSECriterionDS'
criterionG = nn.MSECriterionDS()
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
input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
input2 = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
input_per1 = torch.Tensor(opt.batchSize, 4096)
input_per2 = torch.Tensor(opt.batchSize, 4096)
output_good = torch.Tensor(opt.batchSize, 1)
output_bad = torch.Tensor(opt.batchSize, 1)

dummy_zero_translation = torch.Tensor(opt.batchSize, 2):fill(0):cuda()
dummy_identity_scale = torch.Tensor(opt.batchSize, 2):fill(1):cuda()

-- GAN loss
df_dg_gan = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
-- regularization
df_dg_reg = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
-- perceptual loss
df_dg_per = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)


noise = torch.Tensor(opt.batchSize, nz, 1, 1)
fakeSource = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
errG = torch.Tensor(1)
errD_real = torch.Tensor(1)
errD_fake = torch.Tensor(1)
grad_of_ones = torch.Tensor(1):fill(1)
grad_of_mones = torch.Tensor(1):fill(-1)
epoch_tm = torch.Timer()
tm = torch.Timer()
data_tm = torch.Timer()
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


