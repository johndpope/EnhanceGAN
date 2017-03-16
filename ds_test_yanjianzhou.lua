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
require 'dsBound'
require 'labFilter'
require 'contrastFilterCurve'
require 'TopK'
require 'InstanceNormalization'
require 'contrastFilter'

XP_NAME = 'fullconv_KNN_2'

function randomCrop224(input)
    local size =224
    local w, h = input:size(3), input:size(2)
    if w == size and h == size then
       return input
    end
    local x1, y1 = torch.random(0, w - size), torch.random(0, h - size)
    -- print(string.format('[x1, y1] = [%.2f, %.2f] -- ',x1,y1))
    local out = image.crop(input, x1, y1, x1 + size, y1 + size)
    assert(out:size(2) == size and out:size(3) == size, 'wrong crop size')
    return out
end


local alphaA_grad_index = 14
local alphaA_output_index = 16

local curveA_grad_index = 19
local curveA_output_index = 21

local curveB_grad_index = 23
local curveB_output_index = 25

local curveP1_grad_index = 27
local curveP1_output_index = 29

local curveP2_grad_index = 31
local curveP2_output_index = 33


local scale_grad_index = 48
local tra_grad_index = 52
local crop_output_index = 53

local conv_touch_index = 3
local conv_crop_index = 37

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
          matio = require 'matio'
meanstd = {
   mean = { 0,-128,-128 },
   std = { 100,255,255 },
   scale = 1,
}

vgg_meanstd = {
   mean = { 103.939, 116.779, 123.68 },
   std = { 1,1,1 },
   scale = 255,
}

function getRGBback(LAB_img_batch)
    if LAB_img_batch:nDimension() == 3 then
        LAB_img_batch = nn.Unsqueeze(1):forward(LAB_img_batch:double())
    end
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
function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

-- Step 1 
-- Generate the cropped image from netG

-- Step 2
-- Do a netD and get the score

VGG = torch.load('../ECCV16-Net/VGG_16_ds.t7'); --
VGG:remove(1)
VGG:remove(1)
VGG:cuda()
VGG:evaluate()


AVA = torch.load('../ICCV-eval-2-AVA-step2/checkpoints/model_20.t7')
AVA:evaluate()
AVA:cuda()
AVA_score_list1 = torch.Tensor(1134, 1):zero()
AVA_score_list2 = torch.Tensor(1134, 1):zero()

AVA_score_list3 = torch.Tensor(1134, 1):zero()
AVA_score_list4 = torch.Tensor(1134, 1):zero()



img_list = scandir('../All_Images_YanJianzhou')
-- if file_exists('testing_img.t7') then
--    print('Load saved .t7')
--    all_test_image = torch.load('testing_img.t7')
--    test_image_sizes = all_test_image.test_image_sizes
--    all_test_image = all_test_image.all_test_image
-- else
  all_test_image = torch.Tensor(1134,3,224,224)
  test_image_sizes = torch.Tensor(1134,3)
  for i = 1, 1134 do
     print(string.format('Loading [%d/%d]\n', i, 1134))
     I = image.load(string.format('../All_Images_YanJianzhou/%s', img_list[i+2]), 3, 'float')
     test_image_sizes:narrow(1,i,1):copy(torch.Tensor({I:size(1), I:size(2), I:size(3)}))
     I = image.scale(I, 224,224)
     I = image.rgb2lab(I)
     for i=1,3 do
      I[i]:add(-meanstd.mean[i])
      I[i]:div(meanstd.std[i])
     end     
     all_test_image:narrow(1,i,1):copy(I)
  end
--   torch.save('testing_img.t7', {all_test_image=all_test_image, test_image_sizes=test_image_sizes})
-- end





original_mean = -1
original_std = -1
AVA_score_list1:zero()
AVA_score_list3:zero()
I = torch.CudaTensor(1,3,224,224)
I_VGG = torch.CudaTensor(25,3,224,224)
ds_timer = torch.Timer()
for g = 1,26, 25 do
  ds_timer:reset()
  ds_timer:resume()
  AVA_mean = {}
  AVA_std = {}
  AVA_mean2 = {}
  AVA_std2 = {}

  AVA_score_list2:zero()
  AVA_score_list4:zero()
  netG_iter = string.format('%s_%d_net_G', XP_NAME, g*200)
  netG = torch.load(string.format('./checkpoints/%s.t7',netG_iter))
  -- print('Shithead, set num input dims 2 in nn.View(49)')
  -- netG:get(1):get(22):setNumInputDims(2)
  netG:evaluate()
  netG:cuda()
  os.execute("mkdir " .. string.format('1YANJIANZHOUresult_%s', netG_iter))
  new_results = string.format('1YANJIANZHOUresult_%s', netG_iter)
  for i = 1, 1134 do

     original_size = test_image_sizes:narrow(1, i, 1)
     big_width = original_size[1][3]
     big_height = original_size[1][2]
     original_img = image.load(string.format('../All_Images_YanJianzhou/%s', img_list[i+2]), 3, 'float')
     original_img_save = original_img:clone():float()

     --print(string.format('Testing [%d/%d]', i, 100))
     I:copy(all_test_image:narrow(1,i,1))

     -------------------------------------------------------------
     ----------------Getting cropping Parameters------------------
     -------------------------------------------------------------
     I_cropped = netG:forward(I)

     alpha_beta = netG:get(1):get(alphaA_output_index).output
     alpha = alpha_beta[1][1]
     beta = alpha_beta[1][2]

     a = netG:get(1):get(curveA_output_index).output[1][1]
     b = netG:get(1):get(curveB_output_index).output[1][1]
     p = netG:get(1):get(curveP1_output_index).output[1][1]
     q = netG:get(1):get(curveP2_output_index).output[1][1]

     -- contrast = netG:get(1):get(contrast_output_index).output[1][1]
     -- brightness = netG:get(1):get(brightness_output_index).output[1][1]

     crop_conv_output = netG:get(1):get(conv_crop_index).output
     crop_conv_output2 = netG:get(1):get(conv_crop_index+1).output
     lab_conv_output = netG:get(1):get(conv_touch_index).output
     lab_conv_output2 = netG:get(1):get(conv_touch_index+1).output
     save_name = string.format('./%s/crop_%d.mat',new_results, i)
     matio.save(save_name, {t1=crop_conv_output:float(), t2=crop_conv_output2:float(), t3=lab_conv_output:float(), t4=lab_conv_output2:float()})


     LAB = image.rgb2lab(original_img)
     for lab=1,3 do
      LAB[lab]:add(-meanstd.mean[lab])
      LAB[lab]:div(meanstd.std[lab])
     end     

     L = LAB:narrow(1,1,1)
     A = LAB:narrow(1,2,1)
     B = LAB:narrow(1,3,1)

      local k1 = a / ((a - b)^(1/p))
      local k2 = a / ((a - b)^(1/q))
      L:apply(function(m) 
          if m < b then 
              return 0
          elseif m >= b and m < a then
              return k1* (         (m -      b)^(1/p)    ) +  0
          elseif m >= a and m < 1-a then
              return m
          elseif m >= 1-a and m < 1-b then
              return k2* (         (m - (1-a))^(1/q)    ) + 1-a
          else
              return 1
          end
        end
        )       


      k = 1.0 / (1 - 2*alpha)
      b = -k*alpha
      A:apply(function(m) 
        if m < alpha then
            return 0
        elseif m > 1-alpha then
            return 1
        else 
            return k*m + b
        end
      end
      )
      k = 1.0 / (1 - 2*beta)
      b = -k*beta
      B:apply(function(m) 
        if m < beta then
            return 0
        elseif m > 1-beta then
            return 1
        else 
            return k*m + b
        end
      end
      )
     LAB:narrow(1,1,1):copy(L)
     LAB:narrow(1,2,1):copy(A)
     LAB:narrow(1,3,1):copy(B)
     filtered_img = getRGBback(LAB):squeeze():float()
     save_name = string.format('./%s/filteredONLY_%d.jpg',new_results, i)
     image.save(save_name, filtered_img)         
      
     crop_params = netG:get(1):get(crop_output_index).output
     -----------------------------------------------------------------------------------------------------
     ----------------Getting parameters of crop, apply it onto the original image------------------
     -----------------------------------------------------------------------------------------------------

     s1 = crop_params[1][1]
     s2 = crop_params[1][2]
     t1 = crop_params[1][3]
     t2 = crop_params[1][4]
     save_name = string.format('./%s/crop_coordinates_%d.mat',new_results, i)
     matio.save(save_name, {s1=s1, s2=s2, t1=t1, t2=t2})     
     -- row, col = I:size(3), I:size(4)
     row, col = big_height, big_width
     sub_rows = torch.round(row * s1)
     sub_cols = torch.round(col * s2)
     I_cropped_output = torch.rand(1, 3, sub_rows, sub_cols)
     croppedONLY = torch.rand(1,3,sub_rows, sub_cols)
     I_mask = original_img:clone():fill(0)
     row_start = torch.floor((row - sub_rows)/2)+1   +   torch.round(t1 * sub_rows/2);
     col_start = torch.floor((col - sub_cols)/2)+1   +   torch.round(t2 * sub_cols/2);
     if row_start <= 0 then 
        row_start = 1
     end
     if col_start <= 0 then 
        col_start = 1
     end

     if row_start + sub_rows - 1 > row then
         row_start = row - sub_rows + 1
     end

     if col_start + sub_cols - 1 > col then
        col_start = col - sub_cols + 1
     end


     -- print(filtered_img:size())
     -- print(row_start, sub_rows, col_start, sub_cols)
     croppedONLY:copy(original_img:narrow(2,row_start,sub_rows):narrow(3, col_start, sub_cols))
     save_name = string.format('./%s/cropONLY_%d.jpg',new_results, i)
     image.save(save_name, croppedONLY:squeeze())     

     I_cropped_output:copy(filtered_img:narrow(2,row_start,sub_rows):narrow(3, col_start, sub_cols))
     I_mask:narrow(2,row_start,sub_rows):narrow(3, col_start, sub_cols):fill(1):float()
     I_cropped_output = I_cropped_output:squeeze():float()
     save_name = string.format('./%s/crop_%d.jpg',new_results, i)
     image.save(save_name, I_cropped_output)     
    
     -----------------------------------------------------------------------------------------------------
     ----------------I_cropped_output is the RGB cropped, filtered image in its Original Size------------
     -----------------------------------------------------------------------------------------------------
      I_cropped_vgg = image.scale(I_cropped_output, 256,256)
       for i=1,3 do
        I_cropped_vgg[i]:mul(vgg_meanstd.scale)
        I_cropped_vgg[i]:add(-vgg_meanstd.mean[i])
        I_cropped_vgg[i]:div(vgg_meanstd.std[i])
       end   


     I_cropped_lab = image.scale(I_cropped_output, 224,224)
     I_cropped_lab = image.rgb2lab(I_cropped_lab)
     for i=1,3 do
      I_cropped_lab[i]:add(-meanstd.mean[i])
      I_cropped_lab[i]:div(meanstd.std[i])
     end     

                 if g == 1 then
                     -- Getting performance on the original image
                     AVA_score1 = AVA:forward(I:cuda()) -- testing normalized lab image
                     AVA_score_list1:narrow(1,i,1):copy(AVA_score1[2])

                     I_rgb = image.scale(original_img_save:clone(), 256,256)
                       for i=1,3 do
                        I_rgb[i]:mul(vgg_meanstd.scale)
                        I_rgb[i]:add(-vgg_meanstd.mean[i])
                        I_rgb[i]:div(vgg_meanstd.std[i])
                       end   
                       for i = 1,25 do
                          I_VGG:narrow(1,i,1):copy(randomCrop224(I_rgb))
                       end
                     AVA_score_vgg = VGG:forward(I_VGG) -- size 25x2
                     temp = AVA_score_vgg:clone():float()
                     -- print(temp)
                     local numVotes, myScore
                     if temp:narrow(2,2,1):gt(0.5):sum() >= 13 then
                        --print('good img')
                        numVotes = temp:narrow(2,2,1):gt(0.5):sum()
                        myScore =  temp:narrow(2,2,1):cmul(temp:narrow(2,2,1):gt(0.5):float()):sum() / numVotes
                     elseif temp:narrow(2,2,1):gt(0.5):sum() == 0 then
                        --print('bad img')
                        numVotes = 25
                        myScore =  temp:narrow(2,2,1):cmul(temp:narrow(2,2,1):lt(0.5):float()):sum() / numVotes
                     else
                        --print('bad img')
                        numVotes = temp:narrow(2,2,1):lt(0.5):sum()
                        -- print(string.format('numVotes is %f', numVotes))
                        myScore =  temp:narrow(2,2,1):cmul(temp:narrow(2,2,1):lt(0.5):float()):sum() / numVotes
                        -- print(string.format('My Score is %f', myScore))
                     end
                     -- print(string.format('My Score is %f', myScore))
                     AVA_score_list3:narrow(1,i,1):fill(myScore)--:copy(torch.Tensor(1,1):fill(myScore))
                 end

     I:copy(I_cropped_lab) -- test normalized cropped lab  
     AVA_score2 = AVA:forward(I)   
     AVA_score_list2:narrow(1,i,1):copy(AVA_score2[2])
     -- Testing using VGG
     for i = 1,25 do
        I_VGG:narrow(1,i,1):copy(randomCrop224(I_cropped_vgg))
     end
     AVA_score2_vgg = VGG:forward(I_VGG)
                     temp = AVA_score2_vgg:clone():float()
                     local numVotes, myScore

                     if temp:narrow(2,2,1):gt(0.5):sum() >= 13 then
                        --print('good img')
                        numVotes = temp:narrow(2,2,1):gt(0.5):sum()
                        myScore =  temp:narrow(2,2,1):cmul(temp:narrow(2,2,1):gt(0.5):float()):sum() / numVotes
                     elseif temp:narrow(2,2,1):gt(0.5):sum() == 0 then
                        --print('bad img')
                        numVotes = 25
                        myScore =  temp:narrow(2,2,1):cmul(temp:narrow(2,2,1):lt(0.5):float()):sum() / numVotes
                     else
                        --print('bad img')
                        numVotes = temp:narrow(2,2,1):lt(0.5):sum()
                        -- print(string.format('numVotes is %f', numVotes))
                        myScore =  temp:narrow(2,2,1):cmul(temp:narrow(2,2,1):lt(0.5):float()):sum() / numVotes
                        -- print(string.format('My Score is %f', myScore))
                     end
     AVA_score_list4:narrow(1,i,1):fill(myScore)--:copy(torch.Tensor(   {myScore}   ):resize(1,1))

     --final_output = filtered_img:clone():float()
     I_write = torch.cat(original_img_save:float(), filtered_img:cmul(I_mask:float()),2)


     local function getSign(t)
       if t > 0 then
          t = string.format('+%.2f', t)
       else 
          t = string.format('%.2f', t)
       end
       return t
     end

     save_name = string.format('./%s/crop_%d_(%.3f, %.3f)_(%.3f, %.3f)_[%.2f, %.2f, %s, %s]_[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f].jpg',new_results, i, 
     AVA_score_list1[i][1], AVA_score_list2[i][1], 
     AVA_score_list3[i][1], AVA_score_list4[i][1],
     s1,s2,getSign(t1),getSign(t2), alpha, beta, a,b,p,q)
     image.save(save_name, I_write);
     -- image.save(string.format('./%s/crop_%d_[%.2f, %.2f, %.2f, %.2f].jpg',new_results, i, s1,s2,t1,t2), I_cropped_output);
  end   
  mean_score = AVA_score_list2:mean()
  std_score = AVA_score_list2:std()
  AVA_mean[g] = mean_score
  AVA_std[g]  = std_score

  mean_score2 = AVA_score_list4:mean()
  std_score2 = AVA_score_list4:std()
  AVA_mean2[g] = mean_score2
  AVA_mean2[g] = std_score2
  if g == 1 then
    original_mean = AVA_score_list1:mean()
    original_std = AVA_score_list1:std()
    original_mean2 = AVA_score_list3:mean()
    original_std2 = AVA_score_list3:std()    
  end
  print(string.format('NetG: [iter %d], score mean = %.4f, %4f (%.4f, %.4f), std = %.4f, %.4f (%.4f, %.4f), t = %f', 
    g, mean_score,mean_score2,
       original_mean,original_mean2,
       std_score,std_score2,
       original_std,original_std2,
       ds_timer:time().real))
  matio = require 'matio'
  save_name = string.format('./%s/AVA_score_list1.mat',new_results)
  matio.save(save_name, {t1=AVA_score_list1:float(), t2=AVA_score_list2:float(), t3=AVA_score_list3:float(), t4=AVA_score_list4:float()})  
end






