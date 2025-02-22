Simple neural network for predicting future episode's describings of any art.
Dataset tutorial:
X - Input (Season, Episode, chunk)
Y - Output (Ex,Ey,Px,Py)
Chunk = 10% of Episode(time)

Ex & Ey - coordinates of complex number in plane of Emotion.
(1.0 ; 1.0) like "Thankfully"
(1.0 ; 0.0) like "I dont care about you"
(1.0 ; -1.0) like "Im afraid"
(0.0 ; 1.0) like "Im happy"
(0.0 ; 0.0) like "chill, stone, sigma"
(0.0 ; -1.0) like "im sad"
(-1.0 ; 1.0) like "I love you"
(-1.0 ; 0.0) like "I dont know you"
(-1.0 ; -1.0) like " I hate you"

Px & Py - coordinates of complex number in plane of Plot.
(1.0 ; 1.0) like "Do you remember this plottwist?"
(1.0 ; 0.0) like "Do you remember this place?"
(1.0 ; -1.0) like "Do you remember this guy?"
(0.0 ; 1.0) like "Plottwist"
(0.0 ; 0.0) like "Place"
(0.0 ; -1.0) like "Guy"
(-1.0 ; 1.0) like "WOW! PLOTTWIST"
(-1.0 ; 0.0) like "WOW! NEW PLACE"
(-1.0 ; -1.0) like "WOW! NEW GUY"

every number (E-P x-y) in range [-1.0 ; 1.0]
