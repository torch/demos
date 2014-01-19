-- global state (excluding ui state)
local state = {}

state.rawFrame       = torch.Tensor()
state.rawFrameP      = torch.Tensor()
state.procFrame      = torch.Tensor()
state.results        = {} -- stores the position of tracked objects
state.memory         = {} -- stores an image patch (prototype)
state.distributions  = torch.Tensor()
state.learn          = nil -- signal for adding new prototype

-- flag that the end of video or dataset has been reached
state.finished = false
state.finish = function()
                  state.logit('you have reached the end of the video')
                  if state.dsoutfile then
                     state.dsoutfile:close()
                  end
               end

-- options
state.classes   = options.classes
state.threshold = options.threshold
state.autolearn = options.autolearn
state.maxProb   = 0

if options.nogui then
   function state.begin()
      while not state.finished do
         profiler:start('full-loop','fps')
         print('Frame:',source.current)
         process()
         print('')
         profiler:lap('full-loop')
               print(profiler.list[1])
      end
      state.finish()
   end
   function state.logit(msg)
      print(msg)
   end
else
   local function loop()
      profiler:start('full-loop','fps')
      process()
      display.update()
      profiler:lap('full-loop')
      display.results()
   end
   function state.begin()
      display.begin(loop)
   end
   -- provide log
   state.log = {}
   function state.logit(str,color)
      -- color can be a class id (number) or a color name (string)
      print(str)
      if type(color) == 'number' then
         color = ui.colors[color]
      end
      table.insert(state.log,{str=str, color=color or 'black'})
   end
end

return state
