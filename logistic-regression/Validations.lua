-- validations.lua
-- A collection of static methods used to validate arguments to functions

do
   local Validations = torch.class('Validations')

   -- nothing to do, because all the methods are static
   function Validations:__init()
   end

   local function isNotNil(value, name)
      if type(value) == nil then error(name .. ' is nil and should not be') end
   end
        
   local function isType(value, name, expected, userExpected)
      if type(value) == expected then return end
      userExpected = userExpected or expected
      error(name .. ' is a ' .. type(value) .. ' not a ' .. userExpected)
   end

   function Validations.isBoolean(value, name)
      isNotNil(value, name)
      isType(value, name, 'boolean')
   end

   function Validations.isFunction(value, name)
      isNotNil(value, name)
      isType(value, name, 'function')
   end

   function Validations.isNumber(value, name)
      isNotNil(value, name)
      isType(value, name, 'number')
   end

   function Validations.isNumberOrTensor(value, name)
      isNotNil(value, name)
      if type(value) == 'number' or type(value) == 'userdata' then return end
      error(name .. ' is a ' .. type(value) .. ' not a number or Tensor')
   end

   function Validations.isTensor(value, name)
      isNotNil(value, name)
      isType(value, name, 'userdata', 'Tensor')
   end

   function Validations.is1DTensor(value, name)
      Validations.isTensor(value, name)
      assert(value:nDimension() == 1, 
             name .. ' has ' .. value:nDimension() .. ' dimensions, not 1')
   end

   function Validations.is2DTensor(value, name)
      Validations.isTensor(value, name)
      assert(value:nDimension() == 2, 
             name .. ' has ' .. value:nDimension() .. ' dimensions, not 2')
   end

   function Validations.isIntegerGe0(value, name)
      isNotNil(value, name)
      isType(value, name, 'number')
      if math.floor(value) ~= value then error(name..' is not an integer') end
      if value < 0 then error(name .. ' is not >= 0') end
   end

   function Validations.isIntegerGt0(value, name)
      isNotNil(value, name)
      isType(value, name, 'number')
      if math.floor(value) ~= value then error(name..' is not an integer')end
      if value <= 0 then error(name .. ' is not > 0') end
   end

   function Validations.isNilOrBoolean(value, name)
      if value == nil then return end
      Validations.isBoolean(value, name)
   end

   function Validations.isNilOrFunction(value, name)
      if value == nil then return end
      Validations.isFunction(value, name)
   end

   function Validations.isNotNil(value, name)
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
      isNotNil(value, name)
      isType(value, name, 'number')
      if value < 0 then error(name .. ' is not >= 0') end
   end

   function Validations.isNumberGt0(value, name)
      isNotNil(value, name)
      isType(value, name, 'number')
      if value <= 0 then error(name .. ' is not > 0') end
   end

   function Validations.isTable(value, name)
      isNotNil(value, name)
      isType(value, name, 'table')
   end

   -- each element >= 0
   function Validations.isVectorGe0(value, name)
      Validations.is1DTensor(value, name)
      for i=1,value:size(1) do
	 if value[i] < 0 then error(name .. ' is not element wise >= 0') end
      end
   end
     

end -- definition of class Validations