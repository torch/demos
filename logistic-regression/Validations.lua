-- validations.lua
-- A collection of static methods used to validate arguments to functions

-- return true if the validation condition is satified
-- otherwise raise an error

do
   local Validations = torch.class('Validations')

   -- nothing to do, because all the methods are static
   function Validations:__init()
   end

   local function isNotNil(value, name)
      if type(value) == nil then error(name .. ' is nil and should not be') end
      return true
   end
        
   local function isType(value, name, expected)
      if false then
         print('isType value', value)
         print('isType name', name)
         print('isType expected', expected)
      end
      if type(value) == expected then return true end
      error(name .. ' is a ' .. type(value) .. ' not a ' .. expected)
   end

   function Validations.isBoolean(value, name)
      isNotNil(value, name)
      return isType(value, name, 'boolean')
   end

   function Validations.isEqual(a, b, msgA, msgB)
      if a == b then return true end
      error(a .. '(' .. msgA .. ') is not equal to ' .. b .. '(' .. msgB .. ')')
   end

   function Validations.isFunction(value, name)
      isNotNil(value, name)
      return isType(value, name, 'function')
   end

   function Validations.isFunctionOrTable(value, name)
      isNotNil(value, name)
      if type(value) == 'function' or 
         type(value) == 'table' then 
         return 
      end
      error(name .. ' is a ' .. type(value) .. ' not a function or table')
   end

   function Validations.isNumberLe(value, amount, msgValue, msgAmount)
      Validations.isNumber(value)
      assert(value <= amount,
             value .. '(' ..
             msgValue .. ') is not <= ' .. 
             amount .. '(' .. 
             msgAmount .. ')')
      return true
   end

   function Validations.isNumber(value, name)
      if false then
         print('isNumber value', value)
         print('isNumber name', name)
      end
      isNotNil(value, name)
      return isType(value, name, 'number')
   end

   function Validations.isNumberOrTensor(value, name)
      isNotNil(value, name)
      if type(value) == 'number' or type(value) == 'userdata' then 
         return true 
      end
      error(name .. ' is a ' .. type(value) .. ' not a number or Tensor')
   end

   function Validations.isTable(value, name)
      isNotNil(value, name)
      return isType(value, name, 'table')
   end

   function Validations.isTensor(value, name)
      isNotNil(value, name)
      local tn = torch.typename(value)
      assert(tn, name .. ' is a ' .. type(value) .. ' not a torch.*Tensor')
      if string.match(tn, 'torch.*Tensor') then return true end
      error(name .. ' is a ' .. type(value) .. ' not a torch.*Tensor')
   end

   function Validations.is1DTensor(value, name)
      Validations.isTensor(value, name)
      assert(value:nDimension() == 1, 
             name .. ' has ' .. value:nDimension() .. ' dimensions, not 1')
      return true
   end

   function Validations.isTensor1D(value, name)
      return Validations.is1DTensor(value, name)
   end

   function Validations.is2DTensor(value, name)
      Validations.isTensor(value, name)
      assert(value:nDimension() == 2, 
             name .. ' has ' .. value:nDimension() .. ' dimensions, not 2')
      return true
   end

   function Validations.isIntegerGe0(value, name)
      isNotNil(value, name)
      isType(value, name, 'number')
      if math.floor(value) ~= value then error(name..' is not an integer') end
      if value < 0 then error(name .. ' is not >= 0') end
      return true
   end

   function Validations.isIntegerGt0(value, name)
      isNotNil(value, name)
      isType(value, name, 'number')
      if math.floor(value) ~= value then error(name..' is not an integer')end
      if value <= 0 then error(name .. ' is not > 0') end
      return true
   end

   function Validations.isNilOrBoolean(value, name)
      if value == nil then return end
      return Validations.isBoolean(value, name)
   end

   function Validations.isNilOrFunction(value, name)
      if value == nil then return end
      return Validations.isFunction(value, name)
   end

   function Validations.isNotNil(value, name)
      if not (value == nil) then return end
      error(name .. ' is nil')
      return true
   end

   function Validations.isNilOrIntegerGe0(value, name)
      if value == nil then return end
      return Validations.isIntegerGe0(value, name)
   end

   function Validations.isNilOrIntegerGt0(value, name)
      if value == nil then return end
      return Validations.isIntegerGt0(value, name)
   end

   function Validations.isNilOrNumberGe0(value, name)
      if value == nil then return end
      return Validations.isNumberGe0(value, name)
   end

   function Validations.isNilOrNumberGt0(value, name)
      if value == nil then return end
      return Validations.isNumberGt0(value, name)
   end

   function Validations.isNilOrTable(value, name)
      if value == nil then return end
      return Validations.isTable(value, name)
   end

   function Validations.isNilOrVectorGe0(value, name)
      if value == nil then return end
      return isVectorGe0(value, name)
   end

   function Validations.isNumberGe0(value, name)
      isNotNil(value, name)
      isType(value, name, 'number')
      if value < 0 then error(name .. ' is not >= 0') end
      return true
   end

   function Validations.isNumberGt0(value, name)
      isNotNil(value, name)
      isType(value, name, 'number')
      if value <= 0 then error(name .. ' is not > 0') end
      return true
   end

   function Validations.isTable(value, name)
      isNotNil(value, name)
      return isType(value, name, 'table')
   end

   -- each element >= 0
   function Validations.isVectorGe0(value, name)
      Validations.is1DTensor(value, name)
      for i=1,value:size(1) do
	 if value[i] < 0 then error(name .. ' is not element wise >= 0') end
      end
      return true
   end
     

end -- definition of class Validations
