LoViewer is simple neural network for predicting future episode's describings of any art.

Dataset tutorial:
X - Input (Season, Episode, chunk)
Y - Output (Ex,Ey,Px,Py)
Chunk = 10% of Episode(time)

every number (E-P x-y) in range [0.0 ; 1.0]

| Ex  | Ey  |       Describing       |
|-----|-----|------------------------|
|  0  |  0  | Hate                   |
|  0  | 0.5 | Nescience              |
|  0  |  1  | Love                   |
| 0.5 |  0  | Sadness                |
| 0.5 | 0.5 | Neutral                |
| 0.5 |  1  | Joy                    |
|  1  |  0  | Fear                   |
|  1  | 0.5 | Indifference           |
|  1  |  1  | Trust                  |

This table maps the 9 basic emotions to specific coordinates in the Emotion plane as described in your query. Each emotion is associated with a unique combination of Ex and Ey values, ranging from 0 to 1 in increments of 1. This representation allows for a simple yet effective way to encode emotions in a two-dimensional space, which can be useful for various machine learning and data analysis tasks related to emotion prediction and classification.

| Px  | Py  |              Describing                  |
|-----|-----|------------------------------------------|
|  0  |  0  | “New [minor information]!”               |
|  0  | 0.5 | "New [information]!"                     |
|  0  |  1  | "New [major information]!"               |
| 0.5 |  0  | "Just [minor information]."              |
| 0.5 | 0.5 | "Just [information]."                    |
| 0.5 |  1  | "Just [major information]."              |
|  1  |  0  | "Old [minor information]?"               |
|  1  | 0.5 | "Old [information]"                      |
|  1  |  1  | "Old [major information]"                |

This table encodes plot elements and their status using a Storytelling coordinate system. Each combination of Px and Py coordinates corresponds to a specific type of information, describing its importance to the overall narrative and its place within the story’s timeline. Px and Py values range from 0 to 1 in increments of 1, providing 9 distinct categories for classifying plot elements in games, literature, or other forms of storytelling. This system allows for structuring and analyzing information, facilitating the tracking of key moments and their relationships within the world.
