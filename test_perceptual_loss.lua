require 'torch'
require 'optim'
require 'image'
require 'dpnn'
require 'cudnn'
require 'cunn'
require 'nngraph'

require 'fast_neural_style.DataLoader'
require 'fast_neural_style.PerceptualCriterion'

utils = require 'fast_neural_style.utils'
preprocess = require 'fast_neural_style.preprocess'
models = require 'fast_neural_style.models'

cmd = torch.CmdLine()


--[[
Train a feedforward style transfer model
--]]

-- Generic options
-- Generic options
cmd:option('-arch', 'c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3')
cmd:option('-use_instance_norm', 1)
cmd:option('-task', 'style', 'style|upsample')
cmd:option('-h5_file', 'data/ms-coco-256.h5')
cmd:option('-padding_type', 'reflect-start')
cmd:option('-tanh_constant', 150)
cmd:option('-preprocessing', 'vgg')
cmd:option('-resume_from_checkpoint', '')

-- Generic loss function options
cmd:option('-pixel_loss_type', 'L2', 'L2|L1|SmoothL1')
cmd:option('-pixel_loss_weight', 0.0)
cmd:option('-percep_loss_weight', 1.0)
cmd:option('-tv_strength', 1e-6)

-- Options for feature reconstruction loss
cmd:option('-content_weights', '1.0')
cmd:option('-content_layers', '16')
cmd:option('-loss_network', 'models/vgg16.t7')

-- Options for style reconstruction loss
cmd:option('-style_image', 'images/styles/candy.jpg')
cmd:option('-style_image_size', 256)
cmd:option('-style_weights', '1')
cmd:option('-style_layers', '4,9,16,23')
cmd:option('-style_target_type', 'gram', 'gram|mean')

-- Upsampling options
cmd:option('-upsample_factor', 4)

-- Optimization
cmd:option('-num_iterations', 40000)
cmd:option('-max_train', -1)
cmd:option('-batch_size', 4)
cmd:option('-learning_rate', 1e-3)
cmd:option('-lr_decay_every', -1)
cmd:option('-lr_decay_factor', 0.5)
cmd:option('-weight_decay', 0)

-- Checkpointing
cmd:option('-checkpoint_name', 'checkpoint')
cmd:option('-checkpoint_every', 1000)
cmd:option('-num_val_batches', 10)

-- Backend options
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
cmd:option('-backend', 'cuda', 'cuda|opencl')


opt = cmd:parse(arg)

  -- Parse layer strings and weights
opt.content_layers, opt.content_weights =
  utils.parse_layers(opt.content_layers, opt.content_weights)
opt.style_layers, opt.style_weights =
  utils.parse_layers(opt.style_layers, opt.style_weights)

-- Build the model

percep_loss_weight = 0.5
-- Set up the perceptual loss function

Lab2RGB_module = torch.load('../dcgan.Lab2RGB.3layer/checkpoints/Lab2RGB_30_net_G.t7')
cudnn.convert(Lab2RGB_module, nn)
Lab2RGB_module:cuda()
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

meanstd = {
mean = { 0,-128,-128},
std = { 100, 255, 255},
scale = 1,
}


input_buffer1 = torch.CudaTensor(1,3,224,224)
input_buffer2 = torch.CudaTensor(1,3,224,224)
I = image.load('1.png')
-- I = image.load('Iter_1_input.jpg')                                                             
I = image.scale(I, 224, 224, 'bicubic')

out = image.load('3.png')
-- out = image.load('Iter_1_enhanced.jpg')
out = image.scale(out, 224, 224, 'bicubic')

I = image.rgb2lab(I)
out = image.rgb2lab(out)
                                                                                                                                                                                                                          
                                                                
for i=1,3 do
I[i]:add(-meanstd.mean[i])
I[i]:div(meanstd.std[i])
end
                                                                    
for i=1,3 do
out[i]:add(-meanstd.mean[i])
out[i]:div(meanstd.std[i])
end       

out = nn.Unsqueeze(1):forward(out)                                                              
I = nn.Unsqueeze(1):forward(I)

Lab2RGB_FULL:cuda()
y = Lab2RGB_FULL:forward(I:cuda())
input_buffer1:copy(y)
image.save('new_y.jpg', Lab2RGB_FULL:get(1).output:squeeze():float())
out = Lab2RGB_FULL:forward(out:cuda())
input_buffer2:copy(out)
image.save('new_out.jpg', Lab2RGB_FULL:get(1).output:squeeze():float())
loss_net = torch.load('../fast-neural-style/models/vgg16.t7')
crit_args = {
      cnn = loss_net,
      content_layers = opt.content_layers,
      content_weights = opt.content_weights,
    }
percep_crit = nn.PerceptualCriterion(crit_args):cuda()
-- Compute perceptual loss and gradient
target = {content_target=input_buffer2}
percep_loss = percep_crit:forward(input_buffer1, target)
print(percep_loss)

