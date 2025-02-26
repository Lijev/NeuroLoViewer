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

| Ex  | Ey  | Emotion      |
|-----|-----|--------------|
| -1  | -1  | Anger        |
| -1  |  0  | Disgust      |
| -1  |  1  | Love         |
|  0  | -1  | Sadness      |
|  0  |  0  | Neutral      |
|  0  |  1  | Joy          |
|  1  | -1  | Fear         |
|  1  |  0  | Anticipation |
|  1  |  1  | Trust        |

This table maps the 9 basic emotions to specific coordinates in the Emotion plane as described in your query. Each emotion is associated with a unique combination of Ex and Ey values, ranging from -1 to 1 in increments of 1. This representation allows for a simple yet effective way to encode emotions in a two-dimensional space, which can be useful for various machine learning and data analysis tasks related to emotion prediction and classification.
