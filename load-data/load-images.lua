----------------------------------------------------------------------
-- load-images.lua
-- 
-- This script shows how to load images from a directory, and sort
-- the files according to their name

-- This mostly demonstrates how to use Lua's table data structure,
-- and interact with the file system.

-- note: to run this script, simply do:
-- torch load-images.lua

-- By default, the script loads jpeg images. You can change that
-- by specifying another extension:
-- torch load-images.lua --ext png

require 'torch'
require 'xlua'
require 'image'

----------------------------------------------------------------------
-- 1. Parse command-line arguments

op = xlua.OptionParser('load-images.lua [options]')
op:option{'-d', '--dir', action='store', dest='dir', help='directory to load', req=true}
op:option{'-e', '--ext', action='store', dest='ext', help='only load files of this extension', default='jpg'}
opt = op:parse()
op:summarize()

----------------------------------------------------------------------
-- 2. Load all files in directory

-- We process all files in the given dir, and add their full path
-- to a Lua table.

-- Create empty table to store file names:
files = {}

-- Go over all files in directory. We use an iterator, paths.files().
for file in paths.files(opt.dir) do
   -- We only load files that match the extension
   if file:find(opt.ext .. '$') then
      -- and insert the ones we care about in our table
      table.insert(files, paths.concat(opt.dir,file))
   end
end

-- Check files
if #files == 0 then
   error('given directory doesnt contain any files of type: ' .. opt.ext)
end

----------------------------------------------------------------------
-- 3. Sort file names

-- We sort files alphabetically, it's quite simple with table.sort()

table.sort(files, function (a,b) return a < b end)

print('Found files:')
print(files)

----------------------------------------------------------------------
-- 4. Finally we load images

-- Go over the file list:
images = {}
for i,file in ipairs(files) do
   -- load each image
   table.insert(images, image.load(file))
end

print('Loaded images:')
print(images)

-- Display a of few them
for i = 1,math.min(#files,10) do
   image.display{image=images[i], legend=files[i]}
end
