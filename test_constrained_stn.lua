require 'stn'
require 'image'
require 'nngraph'
require 'nn'
require 'dsBound'
 input_img =nn.Identity()()
 m_ds_sca = nn.Identity()()
 m_ds_tra = nn.Identity()()
 dsBound = nn.dsBound()
 m_ds_tra_bounded = dsBound({m_ds_tra, m_ds_sca})
 m_fc7_1 = nn.JoinTable(2)({m_ds_sca, m_ds_tra_bounded})
 m_transp1 = nn.Transpose({2,3},{3,4})(input_img) -- rot, sca or tra
 m_affineT = nn.AffineTransformMatrixGenerator(false, true, true)(m_fc7_1)
 m_affineG = nn.AffineGridGeneratorBHWD(224,224)(m_affineT)
 m_bilinear = nn.BilinearSamplerBHWD(){m_transp1, m_affineG}
 output_img = nn.Transpose({3,4},{2,3})(m_bilinear)
 stn = nn.gModule({input_img, m_ds_sca, m_ds_tra}, {output_img})



	 I = image.load('original.jpg')
	 I = image.scale(I, 224,224, 'bicubic')
	 I = nn.Unsqueeze(1):forward(I)
	 c1 = 0.8   -- how many rows
     c2 = 0.8 -- how many cols
     t1 = -0.6
     t2 = 0.4
	 max_x = (1-c1)/c1
   max_y = (1-c2)/c2
     print(string.format('input:\ns1 = %.2f\ns2 = %.2f\nt1 = %.2f\nt2 = %.2f', c1, c2, t1, t2))
	 print(string.format('Max of translation is %.2f-%.2f', max_x, max_y))

	 input = {I , nn.Unsqueeze(1):forward(torch.Tensor({c1,c2})), 
	                           nn.Unsqueeze(1):forward(torch.Tensor({t1, t2}))}
	 output = stn:forward(input)
	 b1 = stn:backward(input, output:fill(-1))



	 --   top  left
	 --   0.5,-0.8: BL
	 --  -0.5, 0.8: TR
	 image.save('original.jpg', I:squeeze())
	 image.save('output.jpg', output:squeeze())
     print(stn:get(6).output)
     print('gradient of translation')
     print(stn:get(5).gradInput[1]) 

     print('gradient of scale')
     print(stn:get(5).gradInput[2])
 -- stn:backward(I,I)





-- transformParams = torch.Tensor({0.8,0.5}):resize(1,2)
-- rotationOutput = completeTransformation:narrow(2,1,2):narrow(3,1,2):clone()
-- completeTransformation = torch.zeros(batchSize,3,3):typeAs(transformParams)
--     completeTransformation:select(3,1):select(2,1):add(1)
--     completeTransformation:select(3,2):select(2,2):add(1)
--     completeTransformation:select(3,3):select(2,3):add(1)
-- transformationBuffer = torch.Tensor(batchSize,3,3):typeAs(transformParams)
-- paramIndex = 1



-- scaleFactors1 = transformParams:select(2,paramIndex)
-- scaleFactors2 = transformParams:select(2,paramIndex+1)
--  paramIndex = paramIndex + 2

-- transformationBuffer:zero()
-- transformationBuffer:select(3,1):select(2,1):copy(scaleFactors1)
-- transformationBuffer:select(3,2):select(2,2):copy(scaleFactors2)
-- transformationBuffer:select(3,3):select(2,3):add(1)

-- completeTransformation = torch.bmm(completeTransformation, transformationBuffer)
-- scaleOutput = completeTransformation:narrow(2,1,2):narrow(3,1,2):clone()
-- output=completeTransformation:narrow(2,1,2)

-- input = transformParams
-- _gradParams = gradOutput
-- _tranformParams = input
-- transformParams = _tranformParams
-- gradParams = _gradParams:clone()
-- batchSize = transformParams:size(1)
-- paramIndex = transformParams:size(2)
-- gradInput:resizeAs(transformParams)
-- -- translation
--       gradInputScaleparams = gradInput:narrow(2,paramIndex-1,2)
--       sParams1 = transformParams:select(2,paramIndex-1)
--       sParams2 = transformParams:select(2,paramIndex)
--       paramIndex = paramIndex-1

--       selectedOutput = rotationOutput
--       selectedGradParams = gradParams:narrow(2,1,2):narrow(3,1,2)
--       gradInputScaleparams:copy(torch.cmul(selectedOutput, selectedGradParams):sum(2))

--       gradParams:select(3,1):select(2,1):cmul(sParams1)
--       gradParams:select(3,2):select(2,1):cmul(sParams1)
--       gradParams:select(3,1):select(2,2):cmul(sParams2)
--       gradParams:select(3,2):select(2,2):cmul(sParams2)
