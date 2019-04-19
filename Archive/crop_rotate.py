# -*- coding: utf-8 -*-
from PIL import Image,ImageDraw
import numpy
import math


CenterPoint = (2080,1482)
ClickPoint = (1200,1200)
AreaHeight = 400
AreaWidth = 300


img = Image.open("Circle.png")
#---------------
draw =ImageDraw.Draw(img)
draw.ellipse((CenterPoint[0]-10,CenterPoint[1]-10,CenterPoint[0]+10,CenterPoint[1]+10), fill = (255, 0, 0))
draw.ellipse((ClickPoint[0]-10,ClickPoint[1]-10,ClickPoint[0]+10,ClickPoint[1]+10),fill = (0, 255, 0))
draw.line([CenterPoint,ClickPoint], fill = (255,0,0), width = 5)
#---------------
img.show()

Radians = math.atan2(ClickPoint[1]-CenterPoint[1], ClickPoint[0]-CenterPoint[0])
Degrees = math.degrees(Radians)+90
print("Origin Point",CenterPoint)
print("Click Point",ClickPoint)
print("Degrees",Degrees)
rotated_img =img.rotate(Degrees, expand=False, center=(CenterPoint))
#rotated_img.show()

RPoint = (CenterPoint[0],CenterPoint[1]-numpy.sqrt(numpy.square(ClickPoint[1]-CenterPoint[1])+numpy.square(ClickPoint[0]-CenterPoint[0])))

'''
print(RPoint)
draw2 =ImageDraw.Draw(rotated_img)
draw2.ellipse( (RPoint[0]-10,RPoint[1]-10,RPoint[0]+10,RPoint[1]+10), fill = (255, 0, 0))
rotated_img.show()
'''
Area = ( RPoint[0]-AreaWidth,RPoint[1]-AreaHeight ,
         RPoint[0]+AreaWidth,RPoint[1]+AreaHeight)
cropped_img = rotated_img.crop(Area)
cropped_img.show()