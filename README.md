# Eng_assignment


The approach is comprised of following steps: <br>
a) Determine pair-wise intersections of lines (e.g., which line intersecting which line?) <br>
b) Construct an adjaceny matrix to represent this finite undirected graph. <br>
c) Clustering the adjaceny matrix to identify the number of total clusters. <br>
<br>
<img src="https://user-images.githubusercontent.com/22897244/124961991-0142f600-e016-11eb-97ab-c6e28bc95bee.png" width="400">
<br>     
     

Line-Line Intersection: <br>

<b> Bezier Curve:</b>
    
A linear Bezier curve can describe how far B(t) is from P0 to P1. Example given below: <br>
  
<img src="https://user-images.githubusercontent.com/22897244/124962761-eb820080-e016-11eb-99a9-3b63da949421.png" width="400">
<br>

We cab define L1 and L2 in terms of Bezier parameters:

<math>
 $ L_1  =  \begin{bmatrix}x_1  \\ y_1\end{bmatrix} +  t \begin{bmatrix}x_2-x_1 \\ y_2-y_1\end{bmatrix}$

<br>
    
$L_2 =   \begin{bmatrix}x_3     \\ y_3\end{bmatrix} + u \begin{bmatrix}x_4-x_3 \\ y_4-y_3\end{bmatrix}$
</math>

The lines are intersecting, if t and u are equal to:
<br>
<math>
$t = \frac{(x_1 - x_3)(y_3-y_4)-(y_1-y_3)(x_3-x_4)}{(x_1-x_2)(y_3-y_4)-(y_1-y_2)(x_3-x_4)}$
</math>
<br>
<math>
$u = \frac{(x_2 - x_1)(y_1-y_3)-(y_2-y_1)(x_1-x_3)}{(x_1-x_2)(y_3-y_4)-(y_1-y_2)(x_3-x_4)}$
</math>
<br>

The intersection will be within the L1 and L2 if 0.0&nbsp;≤&nbsp;"t"&nbsp;≤&nbsp;1.0 or 
0.0&nbsp;≤&nbsp;"u"&nbsp;≤&nbsp;1.0
