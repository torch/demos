------------------------------------------------------------
-- common tools for segmentation: display and so on
--
-- Clement Farabet
--

segmtools = {}

--
-- this function prints segmentation components' names at their 
-- centroid
-- components are assumed to be a list of entries, each entry being
-- an array: {[1]=x, [2]=y, [3]=size, [4]=class}
-- size filtering is done: if size < minsize pixels, class is
-- not shown (should probably be changed for something smarter)
--
function segmtools.overlayclasses (args)
   local painter = args.win or error 'arg win missing: a valid window/painter descriptor'
   local classes = args.classes or error 'arg classes missing: a list of classes'
   local components = args.components or error 'arg components missing: a list of components to tag'
   local zoom = args.zoom or 1
   local fontsize = args.font or (10*zoom)
   local minsize = args.minsize or 1
   local offx = args.offx or 0 
   local offy = args.offy or 0

   font = qt.QFont{serif=false, italic=false, size=fontsize or 10}
   painter:setfont(font)
   painter:setcolor('black')

   if components.centroid_x then
      for i = 1,components:size() do
         local size = components.surface[i]
         if size > minsize then
            local class = classes[components.id2class[components.id[i]][1]]
            local x = components.centroid_x[i]*zoom + offx 
               - (#class)*fontsize/5
            local y = components.centroid_y[i]*zoom + offy 
               +fontsize/4
            painter:moveto(x,y)
            painter:show(class)
         end
      end
   else
      for _,component in ipairs(components) do
         local size = component[3]
         if size > minsize then
            local class = classes[component[4]]
            local x = component[1]*zoom + offx - (#class)*fontsize/5
            local y = component[2]*zoom + offy + fontsize/4 
            painter:moveto(x,y)
            painter:show(class)
         end
      end
   end
end
