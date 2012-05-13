-- validations.lua
-- A collection of static methods used to validate arguments to functions

do
   local Validations = torch.class('Validations')

   -- nothing to do, because all the methods are static
   function Validations:__init()
   end

   function Validations.isBoolean(value, name)
      if value == nil then error(name .. ' is missing') end
      if type(value) ~= 'boolean' then error(name .. ' is not a boolean') end
   end

   function Validations.isIntegerGe0(value, name)
      if value == nil then error(name .. ' is missing') end
      if type(value) ~= 'number' then error(name..' is not a number') end
      if math.floor(value) ~= value then error(name..' is not an integer') end
      if value < 0 then error(name .. ' is not >= 0') end
   end

   function Validations.isIntegerGt0(value, name)
      if value == nil then error(name .. ' is missing') end
      if type(value) ~= 'number' then error(name..' is not a number') end
      if math.floor(value) ~= value then error(name..' is not an integer')end
      if value <= 0 then error(name .. ' is not > 0') end
   end


   function Validations.isNilOrBoolean(value, name)
      if value == nil then return end
      Validations.isBoolean(value, name)
   end

   function Validations.isNilOrFunction(value, name)
      if value == nil then return end
      if type(value) ~= 'function' then error(name..' is not a function') end
   end

   function Validations.isNotNil(value, name)
      print('Validations.isNotNil value name', value, name)
      if not (value == nil) then return end
      error(name .. ' is nil')
   end

   function Validations.isNilOrIntegerGe0(value, name)
      if value == nil then return end
      Validations.isIntegerGe0(value, name)
   end

   function Validations.isNilOrIntegerGt0(value, name)
      if value == nil then return end
      Validations.isIntegerGt0(value, name)
   end

   function Validations.isNilOrNumberGe0(value, name)
      if value == nil then return end
      Validations.isNumberGe0(value, name)
   end

   function Validations.isNilOrNumberGt0(value, name)
      if value == nil then return end
      Validations.isNumberGt0(value, name)
   end

   function Validations.isNilOrVectorGe0(value, name)
      if value == nil then return end
      isVectorGe0(value, name)
   end

   function Validations.isNumberGe0(value, name)
      if type(value) ~= 'number' then error(name .. ' is not a number') end
      if value < 0 then error(name .. ' is not >= 0') end
   end

   function Validations.isNumberGt0(value, name)
      if type(value) ~= 'number' then error(name .. ' is not a number') end
      if value <= 0 then error(name .. ' is not > 0') end
   end

   function Validations.isVectorGe0(value, name)
      if type(value) ~= 'userdata' then error(name .. ' is not a Tensor') end
      if value:nDimension() ~= 1 then 
	 error(name .. ' is not a 1D Tensor') 
      end
      for i=1,value:size(1) do
	 if value[i] < 0 then error(name .. ' is not element wise >= 0') end
      end
   end


end -- definition of class Validations