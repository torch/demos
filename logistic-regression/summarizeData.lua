-- summarizeData.lua

function summarizeData(data)
   local function p(name,value) 
      print(string.format('%20s %d', name, value) )
   end
   
   -- min and max values
   local function minmax(maxField, minField, tensor)
      data.raw[maxField] = torch.max(tensor)
      data.raw[minField] = torch.min(tensor)
      p(maxField, data.raw[maxField])
      p(minField, data.raw[minField])
   end

   minmax('ageMax', 'ageMin', data.raw.age)
   minmax('brandMax', 'brandMin', data.raw.brand)
   minmax('isFemaleMax', 'isFemaleMin', data.raw.isFemale)

   local function numberOf(resultField, sourceField)
      local count = 1 + data.raw[sourceField .. 'Max'] - data.raw[sourceField .. 'Min']
      data.raw[resultField] = count
      p(resultField, count)
   end

   print()
   numberOf('nAge', 'age')
   numberOf('nBrand', 'brand')
   numberOf('nIsFemale', 'isFemale')
end

