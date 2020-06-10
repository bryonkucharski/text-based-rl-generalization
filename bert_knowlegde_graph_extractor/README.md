# Goal
Goal of this model is to extract knowlegde graph tuples `(entity -> entity -> relation)` from a text description of a TextWorld state. These tuples are encoded with a Graph Neural Network and used downstream as part of the state in a reinforcement learning framework.

Example Input |
---   | 
DESCRIPTION:  You are in a kitchen. A standard one.  You make out a fridge. The fridge contains a red apple, a white onion, a raw pork chop, a parsley and some water. I mean, just wow! Isn't TextWorld just the best? You can see an oven. If you haven't noticed it already, there seems to be something there by the wall, it's a table. You see a knife on the table. You can make out a counter. The counter is vast. On the counter you see a raw red potato, a yellow bell pepper and a cookbook. I mean, just wow! Isn't TextWorld just the best? You see a stove. The stove is conventional. But the thing is empty, unfortunately.  There is an open plain door leading north. There is an exit to the east. Don't worry, there is no door.   INVENTORY: You are carrying: a red onion a fried red hot pepper a block of cheese    Recipe #1  Gather all following ingredients and follow the directions to prepare this tasty meal.  Ingredients: block of cheese red hot pepper red onion  Directions: chop the red hot pepper fry the red hot pepper prepare meal 

Example Output |
---   |
`player -> kitchen -> at`
`parsley -> fridge -> in`
`fridge -> kitchen -> at`
`red onion -> player -> in`
           
