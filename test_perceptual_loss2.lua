require 'torch'
require 'optim'
require 'image'
require 'dpnn'
require 'cudnn'
require 'cunn'
require 'nngraph'
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
out = Lab2RGB_FULL:forward(out:cuda())
input_buffer2:copy(out)
loss_net = torch.load('../fast-neural-style/models/vgg16.t7')
loss_net:cuda()
loss_net:evaluate()
o1 = loss_net:forward(input_buffer1)
output1 = loss_net:get(38).output:clone()
o2 = loss_net:forward(input_buffer2)
output2 = loss_net:get(38).output:clone()
Loss = (output1 - output2):cmul((output1-output2))
Loss = Loss:sum() / (224*224*3)
print(Loss)



I = image.load('1.png')
-- I = image.load('Iter_1_input.jpg')                                                             
I = image.scale(I, 224, 224, 'bicubic')

out = image.load('2.png')
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
out = Lab2RGB_FULL:forward(out:cuda())
input_buffer2:copy(out)
loss_net = torch.load('../fast-neural-style/models/vgg16.t7')
loss_net:cuda()
loss_net:evaluate()
o1 = loss_net:forward(input_buffer1)
output1 = loss_net:get(38).output:clone()
o2 = loss_net:forward(input_buffer2)
output2 = loss_net:get(38).output:clone()
Loss = (output1 - output2):cmul((output1-output2))
Loss = Loss:sum() / (224*224*3)
print(Loss)
