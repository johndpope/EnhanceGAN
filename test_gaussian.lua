require 'nngraph'
nngraph.setDebug(true)


n_features = 28 * 28
n_z = 20
rnn_size = 200
n_canvas = 28 * 28
seq_length = 10

N = 12
A = 28
n_data = 1

function duplicate(x)
  local y = nn.Reshape(1)(x)
  local l = {}
  for i = 1, A do 
    l[#l + 1] = nn.Copy()(y)  
  end
  local z = nn.JoinTable(2)(l)
  return z
end

rnn_size = 1024
--read
h_dec_prev = nn.Identity()()
x = nn.Identity()()





gx = duplicate(nn.Linear(rnn_size, 1)(h_dec_prev))
gy = duplicate(nn.Linear(rnn_size, 1)(h_dec_prev))
delta = duplicate(nn.Linear(rnn_size, 1)(h_dec_prev))
gamma = duplicate(nn.Linear(rnn_size, 1)(h_dec_prev))
sigma = duplicate(nn.Linear(rnn_size, 1)(h_dec_prev))
delta = nn.Exp()(delta)
gamma = nn.Exp()(gamma)
sigma = nn.Exp()(sigma)
sigma = nn.Power(-2)(sigma)
sigma = nn.MulConstant(-1/2)(sigma)
gx = nn.AddConstant(1)(gx)
gy = nn.AddConstant(1)(gy)
gx = nn.MulConstant((A + 1) / 2)(gx)
gy = nn.MulConstant((A + 1) / 2)(gy)
delta = nn.MulConstant((math.max(A,A)-1)/(N-1))(delta)
-- encoder = nn.gModule({x,h_dec_prev}, {gx,gy,delta,sigma,gamma})



function genr_filters(g)
  filters = {}
  for i = 1, N do
      mu_i = nn.CAddTable()({g, nn.MulConstant(i - N/2 - 1/2)(delta)})
      mu_i = nn.MulConstant(-1)(mu_i)
      d_i = nn.Power(2)(mu_i)
      exp_i = nn.CMulTable()({d_i, sigma})
      exp_i = nn.Exp()(exp_i)
      exp_i = nn.View(n_data, 1, A)(exp_i)
      filters[#filters + 1] = nn.CMulTable()({exp_i, gamma})
  end
  filterbank = nn.JoinTable(2)(filters)
  return filterbank
end

filterbank_x = genr_filters(gx)
filterbank_y = genr_filters(gy)
patch = nn.MM()({filterbank_x, x})
patch = nn.MM(false, true)({patch, filterbank_y})
--read end
encoder = nn.gModule({x, h_dec_prev}, {patch})
encoder.name = 'encoder'