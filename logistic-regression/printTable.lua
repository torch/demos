-- printTable.lua

-- compacly print all the fields in a table
-- ARGS
-- tableName  : string, the name of the table
-- tableValue : table, the table itself
-- RETURNS
-- nil        
function printTable(tableName, tableValue)
   assert(type(tableName) == 'string')
   assert(type(tableValue) == 'table')

   -- print if fieldValue is not a Tensor nor a table
   local function printOther(fieldName, fieldValue)
      print(string.format('%-30s = %s', fieldName, tostring(fieldValue)))
   end
   
   -- print if fieldValue is a userdata
   local function printUserdata(fieldName, tensorValue)
      local torchTypename = torch.typename(tensorValue)
      if torchTypename == nil then
         -- some kind of userdata not originated by torch
         printOther(fieldName, 'userdata')
      else  
         -- an instance of a torch class
         local sizes = tensorValue:size()
         if sizes == nil then
            -- not a torch.Tensor
            printOther(fieldName, torchTypename)
         else
            -- since responds to method :size(), assumes its a torch.Tensor
            local shape = torchTypename .. ' size'
            for i = 1, #sizes do
               if i == 1 then
                  shape = shape .. ' ' .. tostring(sizes[i])
               else
                  shape = shape .. ' x ' .. tostring(sizes[i])
               end
            end
            printOther(fieldName, shape)
         end
      end
   end

   -- sort the keys
   local keys = {}
   for key in pairs(tableValue) do
      table.insert(keys, {tostring(key), key})
   end
   
   local function compare(a, b)
      return a[1] < b[1]
   end

   table.sort(keys, compare) -- sort on the string version of each key

   -- print keys and values in their sorted order
   for _, key in ipairs(keys) do
      local keystring = key[1]
      local keyvalue = key[2]
      local value = tableValue[keyvalue]

      local fieldName = tableName .. '.' .. keystring

      local valueType = type(value)
      if valueType== 'table' then
         printTable(fieldName, value)
      elseif valueType == 'userdata' then
         printUserdata(fieldName, value)
      else
         printOther(fieldName, value)
      end
   end
end
