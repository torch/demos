-- Clement's old gabor function


require 'image'
require 'lab'
require 'xlua'

-- For the similar Gabor code go to 
-- http://www.mathworks.com/matlabcentral/fileexchange/23253-gabor-filter/content/Gabor%20Filter/gabor_fn.m


-- Size should be odd number 
-- angle (in rad)
-- elipse_ratio = aspect ratio(0.5)

-- test:
--image.display{image=gabor(9,0.5,45,1,0.5), zoom =4}

function gabor(size, sigma, angle, period, ellipse_ratio)
      -- init matrix
      local data = torch.zeros(size,size)

      -- image -> pixel
      period = period * size
      sigma = sigma * size

      -- set params
      local halfsize = math.floor(size/2)
      local sigma_x = sigma
      local sigma_y = sigma/ellipse_ratio

      for y=-halfsize,halfsize do
         for x=-halfsize,halfsize do
				
            x_angle = x*math.cos(angle) + y*math.sin(angle)
            y_angle = -x*math.sin(angle) + y*math.cos(angle)
            data[x+halfsize+1][y+halfsize+1] 
               = math.exp(-0.5*(x_angle^2/sigma_x^2 + y_angle^2/sigma_y^2))
                 * math.cos(2*math.pi*x_angle/period)
         end
      end
      
      -- return new tensor
      return data
   end
