-- csv-test.lua
-- test Csv class

require 'vardump'

-- if instead use require, will not reload if edited during testing
dofile 'Csv.lua' 


tempfilename = "csv-test-delete-me.csv"

function testerror(a, b, msg)
   print("a") vardump(a)
   print("b") vardump(b)
   error(msg)
end

-- test two arrays
function testequalarray(a, b)
   --print("a") vardump(a)
   --print("b") vardump(b)
   if #a ~= #b then
      testerror(a, b,
		string.format("#a == %d ~= %d == #b", #a, #b))
   end
   for i = 1, #a do
      if a[i] ~= b[i] then
	 testerror(a, b, string.format("for i=%d, %q not equal %q", 
				       i, a[i], b[i]))
      end
   end
end

-- test two values
function testvalue(a, b)
   local res = a == b
   if res then return end
   testerror(a, b, string.format("%q not equal %q", a, b))
end



-- test writing file
function writeRecs(csv)
   csv:write({"a","b","c"})
   csv:write({01, 02, 03})
   csv:write({11, 12, 13})
end

csv = Csv(tempfilename, "w")
writeRecs(csv)
csv:close()


-- test reading same file line by line
function readRecs(csv)
   row = csv:read()
   testequalarray(row, {"a","b","c"})
   datarownum = 0
   while true do
      local row = csv:read()
      if not row then break end
      datarownum = datarownum + 1
      if datarownum == 1 then
	 testequalarray(row, {"1", "2", "3"})
      else
	 testequalarray(row, {"11", "12", "13"})
      end
   end
end

csv = Csv(tempfilename, "r")
readRecs(csv)
csv:close()

-- read same file all at once
csv = Csv(tempfilename, "r")
lines = csv:readall()
csv:close()
testequalarray(lines[1], {"a","b","c"})
testequalarray(lines[2], {"1", "2", "3"})
testequalarray(lines[3], {"11", "12", "13"})

-- test using a | instead of , as a separator
csv = Csv(tempfilename, "w", "|")
writeRecs(csv)
csv:close()

-- now read the records
csv = Csv(tempfilename, "r", "|")
readRecs(csv)
csv:close()

os.execute("rm " .. tempfilename)

print("all tests passed")
