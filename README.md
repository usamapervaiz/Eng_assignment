# Eng_assignment


The approach is comprised of following steps: <br>
a) Determine pair-wise intersections of lines (e.g., which line intersecting which line?) <br>
b) Construct an adjaceny matrix to represent this finite undirected graph. <br>
c) Clustering the adjaceny matrix to identify the number of total clusters. <br>
<br>
<img src="https://user-images.githubusercontent.com/22897244/124961991-0142f600-e016-11eb-97ab-c6e28bc95bee.png" width="400">
<br>     
     

<b> Line-Line Intersection:</b> <br>

    
A linear Bezier curve can describe how far B(t) is from P0 to P1. Example given below: <br>
  
<img src="https://user-images.githubusercontent.com/22897244/124962761-eb820080-e016-11eb-99a9-3b63da949421.png" width="400">
<br>

We can define any two lines intersection in terms of Bezier parameters (detailed mathematics in jupyter notebook).


<b>  Multiple Lines Intersection:</b> <br>

We can use a sweep line algorithm that can find all crossings in a set of line segments (reasons and explanation detailed in notebook). <br>

<img src="https://user-images.githubusercontent.com/22897244/124963549-e1143680-e017-11eb-9d02-ed77cdca3a28.png" width="400">

Please refer to the notebook for the complete details of implementation
