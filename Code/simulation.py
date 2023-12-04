import os
import sys
import random


import cv2
import bpy


DEVICE = 'cuda'

class run_sim:
   def __init__(self):
       self.curr_folder = os.path.dirname(__file__)
       self.scene = bpy.data.scenes['Scene']
       
       self.camera = bpy.data.objects['Camera']
       self.window = bpy.data.objects['Window']
       self.background = bpy.data.objects['Background']
       # Compositing nodes
       bpy.context.scene.use_nodes = True
       self.tree = bpy.context.scene.node_tree
       for obj in bpy.data.objects:
            obj.select_set(False)
            obj.animation_data_clear()

   def get_frame1(self):
       frame_number = 1
       # Current camera location
       curr_pos = self.camera.location
       print("current cam loc before correction",self.camera.location)
       #----------Frame 1-----------------------------------------------------------------
       # set camera to frame 1
       height = (0,0,0.2)
       new_pos = tuple(x+y for x, y in zip(curr_pos,height))
       self.camera.location = new_pos
       print("current cam loc after correction",self.camera.location)
       # Get frame 1
       bpy.data.scenes["Scene"].node_tree.nodes["File Output 2"].file_slots[0].path = f'frame{frame_number}'
       bpy.data.scenes["Scene"].node_tree.nodes["File Output"].file_slots[0].path = f'mask{frame_number}'
       # Render the image. All outputs should save at their respective locations.
       bpy.ops.render.render(write_still=False)
       self.camera.location = tuple(x-y for x, y in zip(new_pos,height))
       # read frame
#       self.frame1 = cv2.imread(self.curr_folder+f"/Outputs/frames/frame{frame_number:03d}")
       
   def get_frame2(self):
       frame_number = 2
       # Current camera location
       curr_pos = self.camera.location
       print("current cam loc frame2",self.camera.location)
       bpy.data.scenes["Scene"].node_tree.nodes["File Output 2"].file_slots[0].path = f'frame{frame_number}'
       bpy.data.scenes["Scene"].node_tree.nodes["File Output"].file_slots[0].path = f'mask{frame_number}'
       # Render the image. All outputs should save at their respective locations.
       bpy.ops.render.render(write_still=False)
       


def main():
   
   # Get the frames
   simulate = run_sim()
   simulate.get_frame2()
   simulate.get_frame1()

#   frame1 = simulate.frame1
#   frame2 = simulate.frame2
   # Current folder
   curr_folder = simulate.curr_folder 

if __name__=="__main__":
   # donot run main.py if imported as a module
   main()
